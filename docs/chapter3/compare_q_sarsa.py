import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "cliff_compare"


def load_module(file_name, module_name):
    module_path = ROOT_DIR / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def build_agent(module, env, learning_rate, gamma, epsilon_start, epsilon_end, epsilon_decay, seed):
    agent_class = getattr(module, "QLearningAgent", None)
    if agent_class is None:
        agent_class = getattr(module, "SarsaAgent")
    agent = agent_class(
        state_dim=env.observation_space.n,
        action_dim=env.action_space.n,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )
    agent.set_seed(seed)
    return agent


def save_algorithm_outputs(module, agent, output_dir, train_result, eval_result):
    output_dir.mkdir(parents=True, exist_ok=True)
    train_reward_path = module.plot_rewards(
        train_result["rewards"],
        "Training Rewards",
        output_dir / "train_rewards",
    )
    train_ma_path = module.plot_rewards(
        train_result["ma_rewards"],
        "Training Moving Average Rewards",
        output_dir / "train_moving_average_rewards",
    )
    eval_reward_path = module.plot_rewards(
        eval_result["rewards"],
        "Evaluation Rewards",
        output_dir / "eval_rewards",
    )
    eval_ma_path = module.plot_rewards(
        eval_result["ma_rewards"],
        "Evaluation Moving Average Rewards",
        output_dir / "eval_moving_average_rewards",
    )
    model_path = output_dir / "q_table.npy"
    agent.save(model_path)
    return {
        "train_reward_path": train_reward_path,
        "train_ma_path": train_ma_path,
        "eval_reward_path": eval_reward_path,
        "eval_ma_path": eval_ma_path,
        "model_path": model_path,
    }


def run_experiment(
    module,
    algorithm_name,
    output_dir,
    train_episodes,
    eval_episodes,
    max_steps,
    seed,
    learning_rate,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
):
    env = module.create_env(seed=seed)
    eval_env = module.create_env(seed=seed)
    try:
        agent = build_agent(
            module,
            env,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon_start=epsilon_start,
            epsilon_end=epsilon_end,
            epsilon_decay=epsilon_decay,
            seed=seed,
        )
        train_result = module.train(
            env,
            agent,
            train_episodes=train_episodes,
            max_steps=max_steps,
            seed=seed,
        )
        eval_result = module.evaluate(
            eval_env,
            agent,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed,
            render=False,
        )
        artifact_paths = save_algorithm_outputs(
            module,
            agent,
            output_dir / algorithm_name,
            train_result,
            eval_result,
        )
        return {
            "algorithm": algorithm_name,
            "train": train_result,
            "eval": eval_result,
            "artifacts": artifact_paths,
        }
    finally:
        env.close()
        eval_env.close()


def summarize_result(result):
    train_rewards = result["train"]["rewards"]
    eval_rewards = result["eval"]["rewards"]
    return {
        "train_mean_reward": float(np.mean(train_rewards)) if train_rewards else math.nan,
        "train_best_reward": max(train_rewards) if train_rewards else None,
        "eval_mean_reward": float(np.mean(eval_rewards)) if eval_rewards else math.nan,
        "eval_best_reward": max(eval_rewards) if eval_rewards else None,
        "output_dir": str(Path(result["artifacts"]["model_path"]).parent),
    }


def plot_comparison(series_map, title, output_stem, xlabel="Episode", ylabel="Reward"):
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt

        figure = plt.figure(figsize=(9, 5))
        for label, values in series_map.items():
            plt.plot(values, label=label)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.grid(alpha=0.3)
        plt.legend()
        output_path = output_stem.with_suffix(".png")
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return output_path
    except ModuleNotFoundError:
        output_path = output_stem.with_suffix(".svg")
        write_svg_multi_curve(series_map, title, output_path, xlabel=xlabel, ylabel=ylabel)
        return output_path


def write_svg_multi_curve(series_map, title, output_path, xlabel="Episode", ylabel="Reward"):
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

    max_len = max(len(values) for values in safe_series.values())

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

    y_ticks = np.linspace(min_value, max_value, num=5)
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

    for idx, (label, values) in enumerate(safe_series.items()):
        points = " ".join(
            "{:.2f},{:.2f}".format(scale_x(index, len(values)), scale_y(value))
            for index, value in enumerate(values)
        )
        color = colors[idx % len(colors)]
        lines.append(
            '<polyline fill="none" stroke="{0}" stroke-width="2.5" points="{1}"/>'.format(
                color, points
            )
        )
        legend_y = 55 + idx * 22
        legend_x = width - margin - 140
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
            '<text x="{0}" y="{1}" font-size="14" text-anchor="middle">{2}</text>'.format(
                width / 2, height - 20, escape_xml(xlabel)
            ),
            '<text x="24" y="{0}" font-size="14" text-anchor="middle" transform="rotate(-90 24 {0})">{1}</text>'.format(
                height / 2, escape_xml(ylabel)
            ),
            '<text x="{0}" y="{1}" font-size="12" text-anchor="end">{2}</text>'.format(
                width - margin, height - margin + 20, max_len
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


def run_comparison(
    output_dir,
    train_episodes=500,
    eval_episodes=20,
    max_steps=200,
    seed=42,
    learning_rate=0.1,
    gamma=0.9,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.99,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    q_module = load_module("cliff_q_learning.py", "cliff_q_learning_compare")
    sarsa_module = load_module("cliff_sarsa.py", "cliff_sarsa_compare")

    q_result = run_experiment(
        q_module,
        "q_learning",
        output_dir,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )
    sarsa_result = run_experiment(
        sarsa_module,
        "sarsa",
        output_dir,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        seed=seed,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
    )

    compare_train_ma_path = plot_comparison(
        {
            "Q-learning": q_result["train"]["ma_rewards"],
            "Sarsa": sarsa_result["train"]["ma_rewards"],
        },
        "Q-learning vs Sarsa Training Moving Average Rewards",
        output_dir / "compare_train_moving_average_rewards",
    )
    compare_eval_path = plot_comparison(
        {
            "Q-learning": q_result["eval"]["rewards"],
            "Sarsa": sarsa_result["eval"]["rewards"],
        },
        "Q-learning vs Sarsa Evaluation Rewards",
        output_dir / "compare_eval_rewards",
    )

    q_summary = summarize_result(q_result)
    sarsa_summary = summarize_result(sarsa_result)
    summary = {
        "q_learning": q_summary,
        "sarsa": sarsa_summary,
        "reward_gap": q_summary["eval_mean_reward"] - sarsa_summary["eval_mean_reward"],
        "comparison_plots": {
            "train_moving_average": str(compare_train_ma_path),
            "eval_rewards": str(compare_eval_path),
        },
        "shared_config": {
            "train_episodes": train_episodes,
            "eval_episodes": eval_episodes,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "epsilon_start": epsilon_start,
            "epsilon_end": epsilon_end,
            "epsilon_decay": epsilon_decay,
            "seed": seed,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "q_result": q_result,
        "sarsa_result": sarsa_result,
        "summary_path": summary_path,
        "compare_plot_path": compare_train_ma_path,
        "compare_eval_path": compare_eval_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Compare Q-learning and Sarsa on CliffWalking-v0")
    parser.add_argument("--train-episodes", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_comparison(
        output_dir=Path(args.output_dir),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))

    print("\nSaved comparison artifacts:")
    print("  summary: {}".format(result["summary_path"]))
    print("  compare train moving average rewards: {}".format(result["compare_plot_path"]))
    print("  compare eval rewards: {}".format(result["compare_eval_path"]))
    print("Q-learning eval mean reward: {:.2f}".format(summary["q_learning"]["eval_mean_reward"]))
    print("Sarsa eval mean reward: {:.2f}".format(summary["sarsa"]["eval_mean_reward"]))
    print("Reward gap (Q-learning - Sarsa): {:.2f}".format(summary["reward_gap"]))


if __name__ == "__main__":
    main()
