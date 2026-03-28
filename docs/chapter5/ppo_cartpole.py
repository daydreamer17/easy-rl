import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gym
import numpy as np


ENV_NAME = "CartPole-v1"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ppo_cartpole"
VALID_MODES = (
    "policy_penalty",
    "policy_clip",
)


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    mode: str = "all"
    train_episodes: int = 150
    eval_episodes: int = 10
    max_steps: int = 500
    hidden_dim: int = 32
    actor_learning_rate: float = 0.003
    value_learning_rate: float = 0.01
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_epochs: int = 6
    clip_epsilon: float = 0.2
    penalty_beta: float = 0.5
    target_kl: float = 0.01
    seed: int = 42
    render: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class AdamOptimizer:
    def __init__(self, params, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {name: np.zeros_like(value) for name, value in params.items()}
        self.v = {name: np.zeros_like(value) for name, value in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for name in params:
            self.m[name] = self.beta1 * self.m[name] + (1.0 - self.beta1) * grads[name]
            self.v[name] = self.beta2 * self.v[name] + (1.0 - self.beta2) * (grads[name] ** 2)
            m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
            v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
            params[name] -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)


class PolicyNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim, seed):
        self.rng = np.random.default_rng(seed)
        self.params = {
            "W1": self.rng.normal(0.0, 1.0 / math.sqrt(state_dim), size=(state_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "W2": self.rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, action_dim)),
            "b2": np.zeros(action_dim, dtype=np.float64),
        }

    def forward(self, state):
        state = np.asarray(state, dtype=np.float64)
        z1 = state @ self.params["W1"] + self.params["b1"]
        hidden = np.tanh(z1)
        logits = hidden @ self.params["W2"] + self.params["b2"]
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits)
        cache = {
            "state": state,
            "hidden": hidden,
            "probs": probs,
        }
        return probs, cache

    def sample_action(self, state):
        probs, _ = self.forward(state)
        action = int(self.rng.choice(len(probs), p=probs))
        return action, float(probs[action]), probs

    def predict_action(self, state):
        probs, _ = self.forward(state)
        best_actions = np.flatnonzero(probs == np.max(probs))
        return int(self.rng.choice(best_actions))

    def loss_and_gradients(
        self,
        states,
        actions,
        old_action_probs,
        old_probs_batch,
        advantages,
        mode,
        clip_epsilon,
        penalty_beta,
    ):
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        loss = 0.0
        surrogate_values = []
        kl_values = []
        clip_count = 0

        for state, action, old_action_prob, old_probs, advantage in zip(
            states, actions, old_action_probs, old_probs_batch, advantages
        ):
            probs, cache = self.forward(state)
            old_probs = np.asarray(old_probs, dtype=np.float64)
            old_action_prob = max(float(old_action_prob), 1e-8)
            new_action_prob = max(float(cache["probs"][action]), 1e-8)
            ratio = new_action_prob / old_action_prob
            kl_value = float(
                np.sum(old_probs * (np.log(np.clip(old_probs, 1e-8, 1.0)) - np.log(np.clip(cache["probs"], 1e-8, 1.0))))
            )
            kl_values.append(kl_value)
            probs_minus_action = cache["probs"].copy()
            probs_minus_action[action] -= 1.0

            if mode == "policy_penalty":
                surrogate = ratio * float(advantage)
                dlogits = probs_minus_action * surrogate + penalty_beta * (cache["probs"] - old_probs)
                loss += -surrogate + penalty_beta * kl_value
                surrogate_values.append(surrogate)
            else:
                clipped_ratio = float(np.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon))
                unclipped = ratio * float(advantage)
                clipped = clipped_ratio * float(advantage)
                surrogate = min(unclipped, clipped)
                use_gradient = True
                if advantage > 0.0 and ratio > 1.0 + clip_epsilon:
                    use_gradient = False
                if advantage < 0.0 and ratio < 1.0 - clip_epsilon:
                    use_gradient = False

                if use_gradient:
                    dlogits = probs_minus_action * (ratio * float(advantage))
                else:
                    dlogits = np.zeros_like(cache["probs"])
                    clip_count += 1

                loss += -surrogate
                surrogate_values.append(surrogate)

            grads["W2"] += np.outer(cache["hidden"], dlogits)
            grads["b2"] += dlogits

            dhidden = dlogits @ self.params["W2"].T
            dz1 = dhidden * (1.0 - cache["hidden"] ** 2)

            grads["W1"] += np.outer(cache["state"], dz1)
            grads["b1"] += dz1

        scale = max(len(states), 1)
        for name in grads:
            grads[name] /= scale
        loss /= scale

        return loss, grads, {
            "mean_kl": float(np.mean(kl_values)) if kl_values else 0.0,
            "mean_surrogate": float(np.mean(surrogate_values)) if surrogate_values else 0.0,
            "clip_fraction": float(clip_count / scale),
        }


class ValueNetwork:
    def __init__(self, state_dim, hidden_dim, seed):
        self.rng = np.random.default_rng(seed + 1000)
        self.params = {
            "W1": self.rng.normal(0.0, 1.0 / math.sqrt(state_dim), size=(state_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "W2": self.rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, 1)),
            "b2": np.zeros(1, dtype=np.float64),
        }

    def forward(self, state):
        state = np.asarray(state, dtype=np.float64)
        z1 = state @ self.params["W1"] + self.params["b1"]
        hidden = np.tanh(z1)
        value = hidden @ self.params["W2"] + self.params["b2"]
        cache = {
            "state": state,
            "hidden": hidden,
            "value": float(value[0]),
        }
        return float(value[0]), cache

    def predict_value(self, state):
        value, _ = self.forward(state)
        return float(value)

    def loss_and_gradients(self, states, targets):
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        loss = 0.0

        for state, target in zip(states, targets):
            predicted, cache = self.forward(state)
            diff = predicted - float(target)
            loss += 0.5 * (diff ** 2)

            dvalue = np.array([diff], dtype=np.float64)
            grads["W2"] += np.outer(cache["hidden"], dvalue)
            grads["b2"] += dvalue

            dhidden = dvalue @ self.params["W2"].T
            dz1 = dhidden * (1.0 - cache["hidden"] ** 2)

            grads["W1"] += np.outer(cache["state"], dz1)
            grads["b1"] += dz1

        scale = max(len(states), 1)
        for name in grads:
            grads[name] /= scale
        loss /= scale
        return loss, grads


class PPOAgent:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        actor_learning_rate,
        value_learning_rate,
        gamma,
        gae_lambda,
        ppo_epochs,
        clip_epsilon,
        penalty_beta,
        target_kl,
        seed,
        mode,
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ppo_epochs = ppo_epochs
        self.clip_epsilon = clip_epsilon
        self.penalty_beta = penalty_beta
        self.target_kl = target_kl
        self.mode = mode
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            seed=seed,
        )
        self.value_network = ValueNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            seed=seed,
        )
        self.actor_optimizer = AdamOptimizer(self.policy.params, learning_rate=actor_learning_rate)
        self.value_optimizer = AdamOptimizer(self.value_network.params, learning_rate=value_learning_rate)

    def sample_action(self, state):
        action, action_prob, probs = self.policy.sample_action(state)
        value = self.value_network.predict_value(state)
        return action, action_prob, probs, value

    def predict_action(self, state):
        return self.policy.predict_action(state)

    def predict_value(self, state):
        return self.value_network.predict_value(state)

    def update(self, trajectory):
        states = trajectory["states"]
        actions = trajectory["actions"]
        old_action_probs = trajectory["old_action_probs"]
        old_probs_batch = trajectory["old_probs"]
        returns = trajectory["returns"]
        advantages = np.asarray(trajectory["advantages"], dtype=np.float64)
        advantages = normalize(advantages)

        actor_losses = []
        critic_losses = []
        kl_values = []
        surrogate_values = []
        clip_fractions = []

        for _ in range(self.ppo_epochs):
            actor_loss, actor_grads, actor_metrics = self.policy.loss_and_gradients(
                states=states,
                actions=actions,
                old_action_probs=old_action_probs,
                old_probs_batch=old_probs_batch,
                advantages=advantages,
                mode=self.mode,
                clip_epsilon=self.clip_epsilon,
                penalty_beta=self.penalty_beta,
            )
            critic_loss, critic_grads = self.value_network.loss_and_gradients(states, returns)
            self.actor_optimizer.step(self.policy.params, actor_grads)
            self.value_optimizer.step(self.value_network.params, critic_grads)

            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)
            kl_values.append(actor_metrics["mean_kl"])
            surrogate_values.append(actor_metrics["mean_surrogate"])
            clip_fractions.append(actor_metrics["clip_fraction"])

        mean_kl = float(np.mean(kl_values)) if kl_values else 0.0
        if self.mode == "policy_penalty":
            if mean_kl > self.target_kl * 1.5:
                self.penalty_beta *= 2.0
            elif mean_kl < self.target_kl / 1.5:
                self.penalty_beta *= 0.5
            self.penalty_beta = float(np.clip(self.penalty_beta, 1e-4, 64.0))

        return {
            "actor_loss": float(np.mean(actor_losses)) if actor_losses else 0.0,
            "critic_loss": float(np.mean(critic_losses)) if critic_losses else 0.0,
            "mean_kl": mean_kl,
            "mean_surrogate": float(np.mean(surrogate_values)) if surrogate_values else 0.0,
            "clip_fraction": float(np.mean(clip_fractions)) if clip_fractions else 0.0,
            "penalty_beta": float(self.penalty_beta),
        }


def create_env(seed=None, render_mode=None):
    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(ENV_NAME, disable_env_checker=True, **make_kwargs)
    except TypeError:
        env = gym.make(ENV_NAME, **make_kwargs)

    if seed is not None:
        reset_env(env, seed=seed)
        if hasattr(env, "action_space"):
            env.action_space.seed(seed)
    return env


def reset_env(env, seed=None):
    if seed is None:
        reset_result = env.reset()
    else:
        try:
            reset_result = env.reset(seed=seed)
        except TypeError:
            if hasattr(env, "seed"):
                env.seed(seed)
            reset_result = env.reset()

    if isinstance(reset_result, tuple):
        observation, _ = reset_result
        return np.asarray(observation, dtype=np.float64)
    return np.asarray(reset_result, dtype=np.float64)


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return np.asarray(next_state, dtype=np.float64), float(reward), done, info
    next_state, reward, done, info = step_result
    return np.asarray(next_state, dtype=np.float64), float(reward), done, info


def moving_average(previous_value, current_value, factor=0.9):
    return previous_value * factor + current_value * (1.0 - factor)


def normalize(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    std = float(np.std(values))
    if std < 1e-8:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / (std + 1e-8)


def compute_returns_and_advantages(rewards, values, dones, gamma, gae_lambda, last_value=0.0):
    rewards = np.asarray(rewards, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dones = np.asarray(dones, dtype=np.float64)
    advantages = np.zeros_like(rewards, dtype=np.float64)
    returns = np.zeros_like(rewards, dtype=np.float64)
    gae = 0.0

    extended_values = np.concatenate([values, np.array([float(last_value)], dtype=np.float64)])
    for index in range(len(rewards) - 1, -1, -1):
        non_terminal = 1.0 - dones[index]
        delta = rewards[index] + gamma * extended_values[index + 1] * non_terminal - extended_values[index]
        gae = delta + gamma * gae_lambda * non_terminal * gae
        advantages[index] = gae
        returns[index] = advantages[index] + extended_values[index]

    return returns, advantages


def collect_episode(env, agent, max_steps, seed=None, render=False):
    states = []
    actions = []
    rewards = []
    dones = []
    old_action_probs = []
    old_probs = []
    values = []
    state = reset_env(env, seed=seed)

    for _ in range(max_steps):
        if render:
            env.render()

        action, action_prob, probs, value = agent.sample_action(state)
        next_state, reward, done, _ = step_env(env, action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        old_action_probs.append(action_prob)
        old_probs.append(np.asarray(probs, dtype=np.float64))
        values.append(value)

        state = next_state
        if done:
            break

    bootstrap_value = 0.0 if (not rewards or dones[-1]) else agent.predict_value(state)
    returns, advantages = compute_returns_and_advantages(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        last_value=bootstrap_value,
    )

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "old_action_probs": old_action_probs,
        "old_probs": old_probs,
        "values": values,
        "returns": returns,
        "advantages": advantages,
        "episode_reward": float(sum(rewards)),
    }


def train(env, agent, train_episodes, max_steps, seed=None):
    rewards = []
    ma_rewards = []
    actor_losses = []
    critic_losses = []
    kl_values = []
    surrogate_values = []
    clip_fractions = []
    beta_values = []
    ma_reward = 0.0

    for episode in range(train_episodes):
        episode_seed = seed + episode if seed is not None else None
        trajectory = collect_episode(
            env=env,
            agent=agent,
            max_steps=max_steps,
            seed=episode_seed,
            render=False,
        )
        update_result = agent.update(trajectory)
        reward = trajectory["episode_reward"]
        rewards.append(reward)
        ma_reward = reward if episode == 0 else moving_average(ma_reward, reward)
        ma_rewards.append(ma_reward)
        actor_losses.append(update_result["actor_loss"])
        critic_losses.append(update_result["critic_loss"])
        kl_values.append(update_result["mean_kl"])
        surrogate_values.append(update_result["mean_surrogate"])
        clip_fractions.append(update_result["clip_fraction"])
        beta_values.append(update_result["penalty_beta"])

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "kl_values": kl_values,
        "surrogate_values": surrogate_values,
        "clip_fractions": clip_fractions,
        "beta_values": beta_values,
    }


def evaluate(env, agent, eval_episodes, max_steps, seed=None, render=False):
    rewards = []
    ma_rewards = []
    ma_reward = 0.0

    for episode in range(eval_episodes):
        episode_seed = seed + 10_000 + episode if seed is not None else None
        state = reset_env(env, seed=episode_seed)
        episode_reward = 0.0

        for _ in range(max_steps):
            if render:
                env.render()

            action = agent.predict_action(state)
            next_state, reward, done, _ = step_env(env, action)
            episode_reward += reward
            state = next_state
            if done:
                break

        rewards.append(float(episode_reward))
        ma_reward = episode_reward if episode == 0 else moving_average(ma_reward, episode_reward)
        ma_rewards.append(float(ma_reward))

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def ensure_plot_backend():
    try:
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def plot_rewards(rewards, title, output_path_without_extension):
    rewards = list(rewards)
    output_path_without_extension = Path(output_path_without_extension)
    plt = ensure_plot_backend()
    if plt is not None:
        output_path = output_path_without_extension.with_suffix(".png")
        figure = plt.figure(figsize=(8, 4.5))
        axis = figure.add_subplot(111)
        axis.plot(range(1, len(rewards) + 1), rewards, linewidth=1.8)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel("Reward")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return output_path

    output_path = output_path_without_extension.with_suffix(".svg")
    write_svg_curve(rewards, title, output_path)
    return output_path


def plot_series_map(series_map, title, output_path_without_extension):
    output_path_without_extension = Path(output_path_without_extension)
    plt = ensure_plot_backend()
    if plt is not None:
        output_path = output_path_without_extension.with_suffix(".png")
        figure = plt.figure(figsize=(8, 4.5))
        axis = figure.add_subplot(111)
        for label, series in series_map.items():
            axis.plot(range(1, len(series) + 1), series, linewidth=1.8, label=label)
        axis.set_title(title)
        axis.set_xlabel("Episode")
        axis.set_ylabel("Reward")
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_path)
        plt.close(figure)
        return output_path

    output_path = output_path_without_extension.with_suffix(".svg")
    write_svg_multi_curve(series_map, title, output_path)
    return output_path


def write_svg_curve(rewards, title, output_path):
    write_svg_multi_curve({"reward": rewards}, title, output_path, show_legend=False)


def write_svg_multi_curve(series_map, title, output_path, show_legend=True):
    width = 800
    height = 450
    margin_left = 70
    margin_right = 30
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    all_values = []
    max_length = 0
    for series in series_map.values():
        series = list(series)
        max_length = max(max_length, len(series))
        all_values.extend(series)

    if not all_values:
        all_values = [0.0, 1.0]
    y_min = min(all_values)
    y_max = max(all_values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    x_denominator = max(max_length - 1, 1)
    y_range = y_max - y_min

    def scale_x(index):
        return margin_left + plot_width * (index / x_denominator)

    def scale_y(value):
        normalized = (value - y_min) / y_range
        return margin_top + plot_height * (1.0 - normalized)

    colors = [
        "#1f77b4",
        "#d62728",
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#8c564b",
    ]

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        ),
        '<rect x="0" y="0" width="{0}" height="{1}" fill="white"/>'.format(width, height),
        '<text x="{0}" y="24" font-size="18" text-anchor="middle">{1}</text>'.format(width / 2, escape_xml(title)),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333" stroke-width="1.5"/>'.format(
            margin_left, height - margin_bottom, width - margin_right
        ),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333" stroke-width="1.5"/>'.format(
            margin_left, margin_top, height - margin_bottom
        ),
    ]

    for tick_index in range(5):
        y_value = y_min + y_range * tick_index / 4.0
        y_position = scale_y(y_value)
        lines.append(
            '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#ddd" stroke-width="1"/>'.format(
                margin_left, y_position, width - margin_right
            )
        )
        lines.append(
            '<text x="{0}" y="{1}" font-size="12" text-anchor="end">{2:.1f}</text>'.format(
                margin_left - 8, y_position + 4, y_value
            )
        )

    if max_length > 1:
        for tick_index in range(5):
            x_value = int(round((max_length - 1) * tick_index / 4.0)) + 1
            x_position = scale_x(max(0, x_value - 1))
            lines.append(
                '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#ddd" stroke-width="1"/>'.format(
                    x_position, margin_top, height - margin_bottom
                )
            )
            lines.append(
                '<text x="{0}" y="{1}" font-size="12" text-anchor="middle">{2}</text>'.format(
                    x_position, height - margin_bottom + 20, x_value
                )
            )

    for color_index, (label, series) in enumerate(series_map.items()):
        series = list(series)
        if not series:
            continue
        path_points = []
        for index, value in enumerate(series):
            path_points.append("{0:.2f},{1:.2f}".format(scale_x(index), scale_y(value)))
        lines.append(
            '<polyline fill="none" stroke="{0}" stroke-width="2" points="{1}"/>'.format(
                colors[color_index % len(colors)], " ".join(path_points)
            )
        )

    if show_legend:
        legend_x = width - margin_right - 170
        legend_y = margin_top + 10
        for color_index, label in enumerate(series_map.keys()):
            item_y = legend_y + color_index * 22
            color = colors[color_index % len(colors)]
            lines.append(
                '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="{3}" stroke-width="3"/>'.format(
                    legend_x, item_y, legend_x + 24, color
                )
            )
            lines.append(
                '<text x="{0}" y="{1}" font-size="13">{2}</text>'.format(
                    legend_x + 32, item_y + 4, escape_xml(label)
                )
            )

    lines.extend(
        [
            '<text x="{0}" y="{1}" font-size="14" text-anchor="middle">Episode</text>'.format(
                width / 2, height - 20
            ),
            '<text x="24" y="{0}" font-size="14" text-anchor="middle" transform="rotate(-90 24 {0})">Reward</text>'.format(
                height / 2
            ),
            "</svg>",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def escape_xml(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def save_parameters(output_dir, agent):
    output_dir = Path(output_dir)
    actor_path = output_dir / "actor_params.npz"
    value_path = output_dir / "value_params.npz"
    np.savez(actor_path, **agent.policy.params)
    np.savez(value_path, **agent.value_network.params)
    return actor_path, value_path


def save_summary(output_dir, mode, config, train_result, eval_result, parameter_paths):
    summary = {
        "mode": mode,
        "config": asdict(config),
        "train_rewards": train_result["rewards"],
        "train_ma_rewards": train_result["ma_rewards"],
        "train_actor_losses": train_result["actor_losses"],
        "train_critic_losses": train_result["critic_losses"],
        "train_kl_values": train_result["kl_values"],
        "train_surrogate_values": train_result["surrogate_values"],
        "train_clip_fractions": train_result["clip_fractions"],
        "penalty_beta_values": train_result["beta_values"],
        "eval_rewards": eval_result["rewards"],
        "eval_ma_rewards": eval_result["ma_rewards"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        "eval_mean_reward": eval_result["mean_reward"],
        "actor_params_path": str(parameter_paths["actor"]),
        "value_params_path": str(parameter_paths["value"]),
    }
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def run_mode(
    mode,
    output_dir,
    train_episodes,
    eval_episodes,
    max_steps,
    seed,
    hidden_dim,
    actor_learning_rate=0.003,
    value_learning_rate=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=6,
    clip_epsilon=0.2,
    penalty_beta=0.5,
    target_kl=0.01,
    render=False,
):
    mode_output_dir = Path(output_dir) / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)
    try:
        agent = PPOAgent(
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim,
            action_dim=env.action_space.n,
            actor_learning_rate=actor_learning_rate,
            value_learning_rate=value_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            penalty_beta=penalty_beta,
            target_kl=target_kl,
            seed=seed,
            mode=mode,
        )
        train_result = train(
            env=env,
            agent=agent,
            train_episodes=train_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        eval_result = evaluate(
            env=eval_env,
            agent=agent,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed,
            render=render,
        )

        train_reward_path = plot_rewards(
            train_result["rewards"],
            "{} train rewards".format(mode),
            mode_output_dir / "train_rewards",
        )
        train_ma_reward_path = plot_rewards(
            train_result["ma_rewards"],
            "{} train moving average rewards".format(mode),
            mode_output_dir / "train_moving_average_rewards",
        )
        eval_reward_path = plot_rewards(
            eval_result["rewards"],
            "{} eval rewards".format(mode),
            mode_output_dir / "eval_rewards",
        )
        actor_path, value_path = save_parameters(mode_output_dir, agent)
        parameter_paths = {
            "actor": actor_path,
            "value": value_path,
        }
        config = Config(
            mode=mode,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            hidden_dim=hidden_dim,
            actor_learning_rate=actor_learning_rate,
            value_learning_rate=value_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            penalty_beta=penalty_beta,
            target_kl=target_kl,
            seed=seed,
            render=render,
            output_dir=str(mode_output_dir),
        )
        summary_path = save_summary(mode_output_dir, mode, config, train_result, eval_result, parameter_paths)

        return {
            "mode": mode,
            "train_result": train_result,
            "eval_result": eval_result,
            "train_reward_path": train_reward_path,
            "train_ma_reward_path": train_ma_reward_path,
            "eval_reward_path": eval_reward_path,
            "summary_path": summary_path,
            "actor_params_path": parameter_paths["actor"],
            "value_params_path": parameter_paths["value"],
        }
    finally:
        env.close()
        eval_env.close()


def run_all_modes(
    output_dir,
    train_episodes,
    eval_episodes,
    max_steps,
    seed,
    hidden_dim,
    actor_learning_rate=0.003,
    value_learning_rate=0.01,
    gamma=0.99,
    gae_lambda=0.95,
    ppo_epochs=6,
    clip_epsilon=0.2,
    penalty_beta=0.5,
    target_kl=0.01,
):
    output_dir = Path(output_dir)
    results = {}
    for mode in VALID_MODES:
        print("\n=== Running mode: {} ===".format(mode))
        results[mode] = run_mode(
            mode=mode,
            output_dir=output_dir,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed,
            hidden_dim=hidden_dim,
            actor_learning_rate=actor_learning_rate,
            value_learning_rate=value_learning_rate,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ppo_epochs=ppo_epochs,
            clip_epsilon=clip_epsilon,
            penalty_beta=penalty_beta,
            target_kl=target_kl,
            render=False,
        )

    compare_train_path = plot_series_map(
        {mode: result["train_result"]["ma_rewards"] for mode, result in results.items()},
        "PPO Modes: Training Moving Average Rewards",
        output_dir / "compare_train_moving_average_rewards",
    )
    compare_eval_path = plot_series_map(
        {mode: result["eval_result"]["rewards"] for mode, result in results.items()},
        "PPO Modes: Evaluation Rewards",
        output_dir / "compare_eval_rewards",
    )
    summary = {
        "modes": {
            mode: {
                "eval_mean_reward": result["eval_result"]["mean_reward"],
                "train_mean_reward": float(np.mean(result["train_result"]["rewards"])),
                "summary_path": str(result["summary_path"]),
            }
            for mode, result in results.items()
        },
        "comparison_plots": {
            "train_moving_average": str(compare_train_path),
            "eval_rewards": str(compare_eval_path),
        },
    }
    summary_path = output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "results": results,
        "compare_train_path": compare_train_path,
        "compare_eval_path": compare_eval_path,
        "summary_path": summary_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 5 PPO on CartPole-v1")
    parser.add_argument("--mode", choices=("all",) + VALID_MODES, default=Config.mode)
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--actor-learning-rate", type=float, default=Config.actor_learning_rate)
    parser.add_argument("--value-learning-rate", type=float, default=Config.value_learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--gae-lambda", type=float, default=Config.gae_lambda)
    parser.add_argument("--ppo-epochs", type=int, default=Config.ppo_epochs)
    parser.add_argument("--clip-epsilon", type=float, default=Config.clip_epsilon)
    parser.add_argument("--penalty-beta", type=float, default=Config.penalty_beta)
    parser.add_argument("--target-kl", type=float, default=Config.target_kl)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.mode == "all":
        result = run_all_modes(
            output_dir=Path(args.output_dir),
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            actor_learning_rate=args.actor_learning_rate,
            value_learning_rate=args.value_learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            ppo_epochs=args.ppo_epochs,
            clip_epsilon=args.clip_epsilon,
            penalty_beta=args.penalty_beta,
            target_kl=args.target_kl,
        )
        summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
        print("\nSaved comparison artifacts:")
        print("  summary: {}".format(result["summary_path"]))
        print("  compare train moving average rewards: {}".format(result["compare_train_path"]))
        print("  compare eval rewards: {}".format(result["compare_eval_path"]))
        for mode, mode_summary in summary["modes"].items():
            print("  {} eval mean reward: {:.2f}".format(mode, mode_summary["eval_mean_reward"]))
        return

    result = run_mode(
        mode=args.mode,
        output_dir=Path(args.output_dir),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        actor_learning_rate=args.actor_learning_rate,
        value_learning_rate=args.value_learning_rate,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        clip_epsilon=args.clip_epsilon,
        penalty_beta=args.penalty_beta,
        target_kl=args.target_kl,
        render=args.render,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print("  train rewards: {}".format(result["train_reward_path"]))
    print("  train moving average rewards: {}".format(result["train_ma_reward_path"]))
    print("  eval rewards: {}".format(result["eval_reward_path"]))
    print("  actor params: {}".format(result["actor_params_path"]))
    print("  value params: {}".format(result["value_params_path"]))
    print("  summary: {}".format(result["summary_path"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
