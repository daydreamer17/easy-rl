import argparse
import json
import math
import random
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path

import gym
import numpy as np


ENV_NAME = "CartPole-v1"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "dqn_cartpole"


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    train_episodes: int = 220
    eval_episodes: int = 10
    max_steps: int = 500
    hidden_dim: int = 64
    learning_rate: float = 0.001
    gamma: float = 0.99
    buffer_capacity: int = 10000
    batch_size: int = 64
    warmup_size: int = 500
    train_every: int = 1
    target_update_steps: int = 100
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 3000
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


class QNetwork:
    def __init__(self, state_dim, hidden_dim, action_dim, seed):
        self.rng = np.random.default_rng(seed)
        self.params = {
            "W1": self.rng.normal(0.0, 1.0 / math.sqrt(state_dim), size=(state_dim, hidden_dim)),
            "b1": np.zeros(hidden_dim, dtype=np.float64),
            "W2": self.rng.normal(0.0, 1.0 / math.sqrt(hidden_dim), size=(hidden_dim, action_dim)),
            "b2": np.zeros(action_dim, dtype=np.float64),
        }

    def forward(self, states):
        states = np.asarray(states, dtype=np.float64)
        if states.ndim == 1:
            states = states.reshape(1, -1)
        z1 = states @ self.params["W1"] + self.params["b1"]
        hidden = np.tanh(z1)
        q_values = hidden @ self.params["W2"] + self.params["b2"]
        cache = {
            "states": states,
            "hidden": hidden,
            "q_values": q_values,
        }
        return q_values, cache

    def predict(self, state):
        q_values, _ = self.forward(state)
        return q_values[0]

    def predict_batch(self, states):
        q_values, _ = self.forward(states)
        return q_values

    def copy_from(self, other_network):
        for name in self.params:
            self.params[name] = other_network.params[name].copy()

    def loss_and_gradients(self, states, actions, targets):
        actions = np.asarray(actions, dtype=np.int64)
        targets = np.asarray(targets, dtype=np.float64)
        q_values, cache = self.forward(states)
        batch_size = max(len(actions), 1)
        predictions = q_values[np.arange(len(actions)), actions]
        diffs = predictions - targets
        loss = 0.5 * float(np.mean(diffs ** 2))

        d_q_values = np.zeros_like(q_values)
        d_q_values[np.arange(len(actions)), actions] = diffs / batch_size

        grads = {
            "W2": cache["hidden"].T @ d_q_values,
            "b2": np.sum(d_q_values, axis=0),
        }

        d_hidden = d_q_values @ self.params["W2"].T
        d_z1 = d_hidden * (1.0 - cache["hidden"] ** 2)
        grads["W1"] = cache["states"].T @ d_z1
        grads["b1"] = np.sum(d_z1, axis=0)
        return loss, grads


class ReplayBuffer:
    def __init__(self, capacity, seed):
        self.buffer = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append(
            (
                np.asarray(state, dtype=np.float64),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float64),
                float(done),
            )
        )

    def sample(self, batch_size):
        indices = self.rng.choice(len(self.buffer), size=batch_size, replace=False)
        batch = [self.buffer[index] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return {
            "states": np.asarray(states, dtype=np.float64),
            "actions": np.asarray(actions, dtype=np.int64),
            "rewards": np.asarray(rewards, dtype=np.float64),
            "next_states": np.asarray(next_states, dtype=np.float64),
            "dones": np.asarray(dones, dtype=np.float64),
        }

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        batch_size,
        warmup_size,
        train_every,
        target_update_steps,
        epsilon_start,
        epsilon_end,
        epsilon_decay_steps,
        seed,
    ):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.warmup_size = warmup_size
        self.train_every = train_every
        self.target_update_steps = target_update_steps
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0
        self.gradient_steps = 0
        self.rng = np.random.default_rng(seed + 2000)
        self.online_q = QNetwork(state_dim, hidden_dim, action_dim, seed)
        self.target_q = QNetwork(state_dim, hidden_dim, action_dim, seed + 1)
        self.target_q.copy_from(self.online_q)
        self.optimizer = AdamOptimizer(self.online_q.params, learning_rate=learning_rate)

    def epsilon(self):
        progress = min(self.total_steps / max(self.epsilon_decay_steps, 1), 1.0)
        return self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, greedy=False):
        if greedy:
            q_values = self.online_q.predict(state)
            best_actions = np.flatnonzero(q_values == np.max(q_values))
            return int(self.rng.choice(best_actions))

        epsilon = self.epsilon()
        if self.rng.random() < epsilon:
            return int(self.rng.integers(self.action_dim))

        q_values = self.online_q.predict(state)
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self.rng.choice(best_actions))

    def train_step(self, replay_buffer):
        if len(replay_buffer) < max(self.batch_size, self.warmup_size):
            return None
        if self.total_steps % self.train_every != 0:
            return None

        batch = replay_buffer.sample(self.batch_size)
        next_q_values = self.target_q.predict_batch(batch["next_states"])
        max_next_q = np.max(next_q_values, axis=1)
        targets = batch["rewards"] + self.gamma * max_next_q * (1.0 - batch["dones"])

        loss, grads = self.online_q.loss_and_gradients(
            states=batch["states"],
            actions=batch["actions"],
            targets=targets,
        )
        self.optimizer.step(self.online_q.params, grads)
        self.gradient_steps += 1

        target_updated = False
        if self.gradient_steps % self.target_update_steps == 0:
            self.target_q.copy_from(self.online_q)
            target_updated = True

        return {
            "loss": loss,
            "epsilon": self.epsilon(),
            "target_updated": target_updated,
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


def write_svg_curve(values, title, output_path):
    values = list(values)
    width = 800
    height = 450
    margin_left = 70
    margin_right = 30
    margin_top = 40
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if not values:
        values = [0.0, 1.0]
    y_min = min(values)
    y_max = max(values)
    if math.isclose(y_min, y_max):
        y_min -= 1.0
        y_max += 1.0

    x_denominator = max(len(values) - 1, 1)
    y_range = y_max - y_min

    def scale_x(index):
        return margin_left + plot_width * (index / x_denominator)

    def scale_y(value):
        normalized = (value - y_min) / y_range
        return margin_top + plot_height * (1.0 - normalized)

    points = []
    for index, value in enumerate(values):
        points.append("{0:.2f},{1:.2f}".format(scale_x(index), scale_y(value)))

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

    if len(values) > 1:
        for tick_index in range(5):
            x_value = int(round((len(values) - 1) * tick_index / 4.0)) + 1
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

    lines.extend(
        [
            '<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{0}"/>'.format(" ".join(points)),
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


def plot_rewards(rewards, title, output_path_without_extension):
    output_path = Path(output_path_without_extension).with_suffix(".svg")
    write_svg_curve(rewards, title, output_path)
    return output_path


def train(env, agent, replay_buffer, train_episodes, max_steps, seed=None):
    rewards = []
    ma_rewards = []
    losses = []
    epsilons = []
    ma_reward = 0.0

    for episode in range(train_episodes):
        episode_seed = seed + episode if seed is not None else None
        state = reset_env(env, seed=episode_seed)
        episode_reward = 0.0
        episode_losses = []

        for _ in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, _ = step_env(env, action)
            replay_buffer.add(state, action, reward, next_state, done)
            agent.total_steps += 1

            train_result = agent.train_step(replay_buffer)
            if train_result is not None:
                episode_losses.append(train_result["loss"])

            episode_reward += reward
            state = next_state
            if done:
                break

        rewards.append(float(episode_reward))
        ma_reward = episode_reward if episode == 0 else moving_average(ma_reward, episode_reward)
        ma_rewards.append(float(ma_reward))
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)
        epsilons.append(float(agent.epsilon()))

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "losses": losses,
        "epsilons": epsilons,
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

            action = agent.select_action(state, greedy=True)
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


def save_parameters(output_dir, agent):
    output_dir = Path(output_dir)
    online_path = output_dir / "online_q_params.npz"
    target_path = output_dir / "target_q_params.npz"
    np.savez(online_path, **agent.online_q.params)
    np.savez(target_path, **agent.target_q.params)
    return online_path, target_path


def save_summary(output_dir, config, train_result, eval_result, parameter_paths):
    summary = {
        "config": asdict(config),
        "train_rewards": train_result["rewards"],
        "train_ma_rewards": train_result["ma_rewards"],
        "train_losses": train_result["losses"],
        "train_epsilons": train_result["epsilons"],
        "eval_rewards": eval_result["rewards"],
        "eval_ma_rewards": eval_result["ma_rewards"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        "eval_mean_reward": eval_result["mean_reward"],
        "online_params_path": str(parameter_paths["online"]),
        "target_params_path": str(parameter_paths["target"]),
    }
    summary_path = Path(output_dir) / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def run_experiment(
    output_dir,
    train_episodes,
    eval_episodes,
    max_steps,
    seed,
    hidden_dim,
    learning_rate=0.001,
    gamma=0.99,
    buffer_capacity=10000,
    batch_size=64,
    warmup_size=500,
    train_every=1,
    target_update_steps=100,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay_steps=3000,
    render=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)
    try:
        agent = DQNAgent(
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim,
            action_dim=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            batch_size=batch_size,
            warmup_size=warmup_size,
            train_every=train_every,
            target_update_steps=target_update_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            seed=seed,
        )
        replay_buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        train_result = train(
            env=env,
            agent=agent,
            replay_buffer=replay_buffer,
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
            "DQN train rewards",
            output_dir / "train_rewards",
        )
        train_ma_reward_path = plot_rewards(
            train_result["ma_rewards"],
            "DQN train moving average rewards",
            output_dir / "train_moving_average_rewards",
        )
        eval_reward_path = plot_rewards(
            eval_result["rewards"],
            "DQN eval rewards",
            output_dir / "eval_rewards",
        )
        online_path, target_path = save_parameters(output_dir, agent)
        config = Config(
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            warmup_size=warmup_size,
            train_every=train_every,
            target_update_steps=target_update_steps,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay_steps=epsilon_decay_steps,
            seed=seed,
            render=render,
            output_dir=str(output_dir),
        )
        summary_path = save_summary(
            output_dir=output_dir,
            config=config,
            train_result=train_result,
            eval_result=eval_result,
            parameter_paths={"online": online_path, "target": target_path},
        )

        return {
            "train_result": train_result,
            "eval_result": eval_result,
            "train_reward_path": train_reward_path,
            "train_ma_reward_path": train_ma_reward_path,
            "eval_reward_path": eval_reward_path,
            "summary_path": summary_path,
            "online_params_path": online_path,
            "target_params_path": target_path,
        }
    finally:
        env.close()
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Chapter 6 simplest DQN on CartPole-v1")
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--buffer-capacity", type=int, default=Config.buffer_capacity)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--warmup-size", type=int, default=Config.warmup_size)
    parser.add_argument("--train-every", type=int, default=Config.train_every)
    parser.add_argument("--target-update-steps", type=int, default=Config.target_update_steps)
    parser.add_argument("--epsilon-start", type=float, default=Config.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=Config.epsilon_end)
    parser.add_argument("--epsilon-decay-steps", type=int, default=Config.epsilon_decay_steps)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    result = run_experiment(
        output_dir=Path(args.output_dir),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        warmup_size=args.warmup_size,
        train_every=args.train_every,
        target_update_steps=args.target_update_steps,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        render=args.render,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print("  train rewards: {}".format(result["train_reward_path"]))
    print("  train moving average rewards: {}".format(result["train_ma_reward_path"]))
    print("  eval rewards: {}".format(result["eval_reward_path"]))
    print("  online Q params: {}".format(result["online_params_path"]))
    print("  target Q params: {}".format(result["target_params_path"]))
    print("  summary: {}".format(result["summary_path"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
