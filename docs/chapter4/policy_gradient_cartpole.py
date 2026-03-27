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
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "policy_gradient_cartpole"
VALID_MODES = (
    "vanilla",
    "baseline",
    "reward_to_go",
    "baseline_reward_to_go",
)


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    mode: str = "all"
    train_episodes: int = 200
    eval_episodes: int = 10
    max_steps: int = 500
    hidden_dim: int = 16
    learning_rate: float = 0.002
    gamma: float = 0.99
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


class RunningBaseline:
    def __init__(self):
        self.count = 0
        self.value = 0.0

    def update(self, episode_return):
        self.count += 1
        self.value += (episode_return - self.value) / self.count


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
        return action, probs

    def predict_action(self, state):
        probs, _ = self.forward(state)
        best_actions = np.flatnonzero(probs == np.max(probs))
        return int(self.rng.choice(best_actions))

    def loss_and_gradients(self, states, actions, advantages):
        grads = {name: np.zeros_like(value) for name, value in self.params.items()}
        loss = 0.0

        for state, action, advantage in zip(states, actions, advantages):
            probs, cache = self.forward(state)
            chosen_prob = max(cache["probs"][action], 1e-8)
            loss -= float(advantage) * math.log(chosen_prob)

            dlogits = cache["probs"].copy()
            dlogits[action] -= 1.0
            dlogits *= float(advantage)

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
        return loss, grads


class REINFORCEAgent:
    def __init__(
        self,
        state_dim,
        hidden_dim,
        action_dim,
        learning_rate,
        gamma,
        seed,
        use_baseline=False,
        use_reward_to_go=False,
    ):
        self.gamma = gamma
        self.use_baseline = use_baseline
        self.use_reward_to_go = use_reward_to_go
        self.policy = PolicyNetwork(
            state_dim=state_dim,
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            seed=seed,
        )
        self.optimizer = AdamOptimizer(self.policy.params, learning_rate=learning_rate)
        self.baseline = RunningBaseline()

    def sample_action(self, state):
        return self.policy.sample_action(state)

    def predict_action(self, state):
        return self.policy.predict_action(state)

    def update(self, states, actions, rewards):
        raw_returns = compute_returns(rewards, self.gamma, reward_to_go=self.use_reward_to_go)
        episode_return = compute_returns(rewards, self.gamma, reward_to_go=False)[0] if rewards else 0.0
        baseline_value = self.baseline.value if self.use_baseline else 0.0
        advantages = raw_returns - baseline_value
        loss, grads = self.policy.loss_and_gradients(states, actions, advantages)
        self.optimizer.step(self.policy.params, grads)
        if self.use_baseline:
            self.baseline.update(episode_return)
        return loss, raw_returns, advantages, baseline_value


def mode_flags(mode):
    return {
        "use_baseline": mode in {"baseline", "baseline_reward_to_go"},
        "use_reward_to_go": mode in {"reward_to_go", "baseline_reward_to_go"},
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


def compute_returns(rewards, gamma, reward_to_go):
    if not rewards:
        return np.array([], dtype=np.float64)

    discounted = np.zeros(len(rewards), dtype=np.float64)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = rewards[index] + gamma * running
        discounted[index] = running

    if reward_to_go:
        return discounted

    return np.full(len(rewards), discounted[0], dtype=np.float64)


def collect_episode(env, agent, max_steps, seed=None, render=False):
    states = []
    actions = []
    rewards = []
    state = reset_env(env, seed=seed)

    for _ in range(max_steps):
        if render:
            env.render()
        action, _ = agent.sample_action(state)
        next_state, reward, done, _ = step_env(env, action)
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        if done:
            break

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
    }


def train(env, agent, train_episodes, max_steps, seed=None):
    rewards = []
    ma_rewards = []
    losses = []
    baselines = []

    for episode_idx in range(train_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        trajectory = collect_episode(env, agent, max_steps=max_steps, seed=episode_seed, render=False)
        loss, _, _, baseline_value = agent.update(
            trajectory["states"],
            trajectory["actions"],
            trajectory["rewards"],
        )
        episode_reward = float(sum(trajectory["rewards"]))
        rewards.append(episode_reward)
        losses.append(float(loss))
        baselines.append(float(baseline_value))
        if ma_rewards:
            ma_rewards.append(moving_average(ma_rewards[-1], episode_reward))
        else:
            ma_rewards.append(episode_reward)

        print(
            "Train Episode {}/{} | reward: {:.1f} | steps: {} | loss: {:.4f} | baseline: {:.2f}".format(
                episode_idx + 1,
                train_episodes,
                episode_reward,
                len(trajectory["rewards"]),
                loss,
                baseline_value,
            )
        )

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "losses": losses,
        "baselines": baselines,
    }


def evaluate(env, agent, eval_episodes, max_steps, seed=None, render=False):
    rewards = []
    ma_rewards = []
    steps = []

    for episode_idx in range(eval_episodes):
        episode_seed = None if seed is None else seed + 10_000 + episode_idx
        state = reset_env(env, seed=episode_seed)
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_steps):
            if render:
                env.render()
            action = agent.predict_action(state)
            next_state, reward, done, _ = step_env(env, action)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if done:
                break

        rewards.append(episode_reward)
        steps.append(episode_steps)
        if ma_rewards:
            ma_rewards.append(moving_average(ma_rewards[-1], episode_reward))
        else:
            ma_rewards.append(episode_reward)

        print(
            "Eval Episode {}/{} | reward: {:.1f} | steps: {}".format(
                episode_idx + 1,
                eval_episodes,
                episode_reward,
                episode_steps,
            )
        )

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "steps": steps,
        "mean_reward": float(np.mean(rewards)) if rewards else math.nan,
    }


def plot_rewards(values, title, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(8, 5))
        plt.plot(values)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        output_path = output_stem.with_suffix(".png")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
    except ModuleNotFoundError:
        output_path = output_stem.with_suffix(".svg")
        write_svg_curve(values, title, output_path)
        return output_path


def plot_series_map(series_map, title, output_stem):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(9, 5))
        for label, values in series_map.items():
            plt.plot(values, label=label)
        plt.title(title)
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(alpha=0.3)
        plt.legend()
        output_path = output_stem.with_suffix(".png")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
    except ModuleNotFoundError:
        output_path = output_stem.with_suffix(".svg")
        write_svg_multi_curve(series_map, title, output_path)
        return output_path


def write_svg_curve(values, title, output_path):
    width = 800
    height = 500
    margin = 60
    plot_width = width - margin * 2
    plot_height = height - margin * 2

    safe_values = values if values else [0.0]
    min_value = min(safe_values)
    max_value = max(safe_values)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    def scale_x(index):
        if len(safe_values) == 1:
            return margin + plot_width / 2
        return margin + (index / (len(safe_values) - 1)) * plot_width

    def scale_y(value):
        ratio = (value - min_value) / (max_value - min_value)
        return height - margin - ratio * plot_height

    points = " ".join(
        "{:.2f},{:.2f}".format(scale_x(index), scale_y(value))
        for index, value in enumerate(safe_values)
    )

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        ),
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="{0}" y="30" font-size="22" text-anchor="middle">{1}</text>'.format(
            width / 2, escape_xml(title)
        ),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333" stroke-width="2"/>'.format(
            margin, margin, height - margin
        ),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333" stroke-width="2"/>'.format(
            margin, height - margin, width - margin
        ),
        '<polyline fill="none" stroke="#0b84f3" stroke-width="2.5" points="{}"/>'.format(points),
        '<text x="{0}" y="{1}" font-size="14" text-anchor="middle">Episode</text>'.format(
            width / 2, height - 15
        ),
        '<text x="20" y="{0}" font-size="14" text-anchor="middle" transform="rotate(-90 20 {0})">Reward</text>'.format(
            height / 2
        ),
        "</svg>",
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_svg_multi_curve(series_map, title, output_path):
    width = 900
    height = 540
    margin = 70
    plot_width = width - margin * 2
    plot_height = height - margin * 2
    colors = ["#0b84f3", "#f97316", "#22c55e", "#ef4444"]

    safe_series = {label: values if values else [0.0] for label, values in series_map.items()}
    all_values = [value for values in safe_series.values() for value in values]
    min_value = min(all_values)
    max_value = max(all_values)
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    def scale_x(index, length):
        if length <= 1:
            return margin + plot_width / 2
        return margin + (index / (length - 1)) * plot_width

    def scale_y(value):
        ratio = (value - min_value) / (max_value - min_value)
        return height - margin - ratio * plot_height

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'.format(
            width, height
        ),
        '<rect width="100%" height="100%" fill="white" />',
        '<text x="{0}" y="35" font-size="24" text-anchor="middle">{1}</text>'.format(
            width / 2, escape_xml(title)
        ),
        '<line x1="{0}" y1="{1}" x2="{0}" y2="{2}" stroke="#333" stroke-width="2"/>'.format(
            margin, margin, height - margin
        ),
        '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#333" stroke-width="2"/>'.format(
            margin, height - margin, width - margin
        ),
    ]

    for idx, (label, values) in enumerate(safe_series.items()):
        points = " ".join(
            "{:.2f},{:.2f}".format(scale_x(index, len(values)), scale_y(value))
            for index, value in enumerate(values)
        )
        color = colors[idx % len(colors)]
        legend_x = width - margin - 140
        legend_y = 55 + idx * 22
        lines.append(
            '<polyline fill="none" stroke="{0}" stroke-width="2.5" points="{1}"/>'.format(
                color, points
            )
        )
        lines.append(
            '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="{3}" stroke-width="4"/>'.format(
                legend_x, legend_y, legend_x + 24, color
            )
        )
        lines.append(
            '<text x="{0}" y="{1}" font-size="13">{2}</text>'.format(
                legend_x + 32, legend_y + 4, escape_xml(label)
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


def save_summary(output_dir, mode, config, train_result, eval_result):
    summary = {
        "mode": mode,
        "config": asdict(config),
        "train_rewards": train_result["rewards"],
        "train_ma_rewards": train_result["ma_rewards"],
        "eval_rewards": eval_result["rewards"],
        "eval_ma_rewards": eval_result["ma_rewards"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        "eval_mean_reward": eval_result["mean_reward"],
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
    learning_rate=0.002,
    gamma=0.99,
    render=False,
):
    mode_output_dir = Path(output_dir) / mode
    mode_output_dir.mkdir(parents=True, exist_ok=True)
    flags = mode_flags(mode)

    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)
    try:
        agent = REINFORCEAgent(
            state_dim=env.observation_space.shape[0],
            hidden_dim=hidden_dim,
            action_dim=env.action_space.n,
            learning_rate=learning_rate,
            gamma=gamma,
            seed=seed,
            use_baseline=flags["use_baseline"],
            use_reward_to_go=flags["use_reward_to_go"],
        )
        train_result = train(
            env,
            agent,
            train_episodes=train_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        eval_result = evaluate(
            eval_env,
            agent,
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
        config = Config(
            mode=mode,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            seed=seed,
            render=render,
            output_dir=str(mode_output_dir),
        )
        summary_path = save_summary(mode_output_dir, mode, config, train_result, eval_result)

        return {
            "mode": mode,
            "train_result": train_result,
            "eval_result": eval_result,
            "train_reward_path": train_reward_path,
            "train_ma_reward_path": train_ma_reward_path,
            "eval_reward_path": eval_reward_path,
            "summary_path": summary_path,
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
    learning_rate=0.002,
    gamma=0.99,
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
            learning_rate=learning_rate,
            gamma=gamma,
            render=False,
        )

    compare_train_path = plot_series_map(
        {mode: result["train_result"]["ma_rewards"] for mode, result in results.items()},
        "Policy Gradient Modes: Training Moving Average Rewards",
        output_dir / "compare_train_moving_average_rewards",
    )
    compare_eval_path = plot_series_map(
        {mode: result["eval_result"]["rewards"] for mode, result in results.items()},
        "Policy Gradient Modes: Evaluation Rewards",
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
    parser = argparse.ArgumentParser(description="Chapter 4 policy gradient on CartPole-v1")
    parser.add_argument("--mode", choices=("all",) + VALID_MODES, default=Config.mode)
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
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
            learning_rate=args.learning_rate,
            gamma=args.gamma,
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
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        render=args.render,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print("  train rewards: {}".format(result["train_reward_path"]))
    print("  train moving average rewards: {}".format(result["train_ma_reward_path"]))
    print("  eval rewards: {}".format(result["eval_reward_path"]))
    print("  summary: {}".format(result["summary_path"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
