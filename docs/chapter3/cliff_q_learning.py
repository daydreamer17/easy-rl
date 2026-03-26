import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gym
import numpy as np


ENV_NAME = "CliffWalking-v0"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "cliff_q_learning"


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    train_episodes: int = 500
    eval_episodes: int = 20
    max_steps: int = 200
    learning_rate: float = 0.1
    gamma: float = 0.9
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: float = 0.99
    seed: int = 42
    render: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class QLearningAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate,
        gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.995,
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.q_table = np.zeros((state_dim, action_dim), dtype=np.float64)
        self.rng = np.random.default_rng()

    def set_seed(self, seed):
        self.rng = np.random.default_rng(seed)

    def predict(self, state):
        q_values = self.q_table[state]
        best_actions = np.flatnonzero(q_values == np.max(q_values))
        return int(self.rng.choice(best_actions))

    def sample(self, state):
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.action_dim))
        return self.predict(state)

    def update(self, state, action, reward, next_state, done):
        td_target = reward
        if not done:
            td_target += self.gamma * np.max(self.q_table[next_state])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def decay_epsilon_value(self):
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, file_path):
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(file_path, self.q_table)

    def load(self, file_path):
        self.q_table = np.load(file_path)


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
        return int(observation)
    return int(reset_result)


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return int(next_state), float(reward), done, info
    next_state, reward, done, info = step_result
    return int(next_state), float(reward), done, info


def moving_average(previous_value, current_value, factor=0.9):
    return previous_value * factor + current_value * (1.0 - factor)


def train(env, agent, train_episodes, max_steps, seed=None):
    rewards = []
    ma_rewards = []
    steps = []

    for episode_idx in range(train_episodes):
        episode_seed = None if seed is None else seed + episode_idx
        state = reset_env(env, seed=episode_seed)
        episode_reward = 0.0
        episode_steps = 0

        for _ in range(max_steps):
            action = agent.sample(state)
            next_state, reward, done, _ = step_env(env, action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            episode_steps += 1
            if done:
                break

        agent.decay_epsilon_value()
        rewards.append(episode_reward)
        steps.append(episode_steps)
        if ma_rewards:
            ma_rewards.append(moving_average(ma_rewards[-1], episode_reward))
        else:
            ma_rewards.append(episode_reward)

        print(
            "Train Episode {}/{} | reward: {:.1f} | steps: {} | epsilon: {:.4f}".format(
                episode_idx + 1,
                train_episodes,
                episode_reward,
                episode_steps,
                agent.epsilon,
            )
        )

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "steps": steps,
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
            action = agent.predict(state)
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

    y_ticks = np.linspace(min_value, max_value, num=5)
    x_axis_y = height - margin

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
            margin, x_axis_y, width - margin
        ),
    ]

    for tick in y_ticks:
        y = scale_y(float(tick))
        lines.append(
            '<line x1="{0}" y1="{1}" x2="{2}" y2="{1}" stroke="#ddd" stroke-width="1"/>'.format(
                margin, y, width - margin
            )
        )
        lines.append(
            '<text x="{0}" y="{1}" font-size="12" text-anchor="end">{2:.1f}</text>'.format(
                margin - 8, y + 4, tick
            )
        )

    lines.extend(
        [
            '<polyline fill="none" stroke="#0b84f3" stroke-width="2.5" points="{}"/>'.format(points),
            '<text x="{0}" y="{1}" font-size="14" text-anchor="middle">Episode</text>'.format(
                width / 2, height - 15
            ),
            '<text x="20" y="{0}" font-size="14" text-anchor="middle" transform="rotate(-90 20 {0})">Reward</text>'.format(
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


def save_summary(output_dir, config, train_result, eval_result):
    summary = {
        "config": asdict(config),
        "train": {
            "final_reward": train_result["rewards"][-1] if train_result["rewards"] else None,
            "best_reward": max(train_result["rewards"]) if train_result["rewards"] else None,
            "mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        },
        "eval": {
            "mean_reward": eval_result["mean_reward"],
            "best_reward": max(eval_result["rewards"]) if eval_result["rewards"] else None,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def parse_args():
    parser = argparse.ArgumentParser(description="Q-learning demo for CliffWalking-v0")
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--epsilon-start", type=float, default=Config.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=Config.epsilon_end)
    parser.add_argument("--epsilon-decay", type=float, default=Config.epsilon_decay)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def build_config(args):
    return Config(
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
        render=args.render,
        output_dir=args.output_dir,
    )


def main():
    args = parse_args()
    config = build_config(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(config.seed)
    np.random.seed(config.seed)

    env = create_env(seed=config.seed)
    eval_env = create_env(
        seed=config.seed,
        render_mode="human" if config.render else None,
    )
    try:
        agent = QLearningAgent(
            state_dim=env.observation_space.n,
            action_dim=env.action_space.n,
            learning_rate=config.learning_rate,
            gamma=config.gamma,
            epsilon_start=config.epsilon_start,
            epsilon_end=config.epsilon_end,
            epsilon_decay=config.epsilon_decay,
        )
        agent.set_seed(config.seed)

        train_result = train(
            env,
            agent,
            train_episodes=config.train_episodes,
            max_steps=config.max_steps,
            seed=config.seed,
        )
        eval_result = evaluate(
            eval_env,
            agent,
            eval_episodes=config.eval_episodes,
            max_steps=config.max_steps,
            seed=config.seed,
            render=config.render,
        )

        train_reward_path = plot_rewards(
            train_result["rewards"],
            "Training Rewards",
            output_dir / "train_rewards",
        )
        train_ma_reward_path = plot_rewards(
            train_result["ma_rewards"],
            "Training Moving Average Rewards",
            output_dir / "train_moving_average_rewards",
        )
        eval_reward_path = plot_rewards(
            eval_result["rewards"],
            "Evaluation Rewards",
            output_dir / "eval_rewards",
        )
        eval_ma_reward_path = plot_rewards(
            eval_result["ma_rewards"],
            "Evaluation Moving Average Rewards",
            output_dir / "eval_moving_average_rewards",
        )
        model_path = output_dir / "q_table.npy"
        agent.save(model_path)
        summary_path = save_summary(output_dir, config, train_result, eval_result)

        print("\nSaved artifacts:")
        print("  train rewards: {}".format(train_reward_path))
        print("  train moving average rewards: {}".format(train_ma_reward_path))
        print("  eval rewards: {}".format(eval_reward_path))
        print("  eval moving average rewards: {}".format(eval_ma_reward_path))
        print("  q table: {}".format(model_path))
        print("  summary: {}".format(summary_path))
        print("Average evaluation reward: {:.2f}".format(eval_result["mean_reward"]))
    finally:
        env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
