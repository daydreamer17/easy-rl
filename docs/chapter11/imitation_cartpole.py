import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


ENV_NAME = "CartPole-v1"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "imitation_cartpole"
EXPERT_OUTPUT_DIR = ROOT_DIR.parent / "chapter7" / "outputs" / "ddqn_cartpole"
EXPERT_SUMMARY_PATH = EXPERT_OUTPUT_DIR / "summary.json"
EXPERT_POLICY_PATH = EXPERT_OUTPUT_DIR / "policy_net.pth"
VALID_MODES = ("bc", "irl", "hybrid")


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    mode: str = "all"
    train_episodes: int = 20
    eval_episodes: int = 10
    max_steps: int = 500
    hidden_dim: int = 64
    policy_learning_rate: float = 0.001
    discriminator_learning_rate: float = 0.001
    gamma: float = 0.99
    entropy_coef: float = 0.01
    bc_batch_size: int = 128
    irl_batch_size: int = 128
    demos_episodes: int = 12
    bc_epochs: int = 4
    rollout_episodes: int = 4
    discriminator_updates: int = 4
    policy_updates: int = 2
    seed: int = 42
    render: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class Chapter7MLP(nn.Module):
    def __init__(self, n_states, hidden_dim, n_actions):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_states, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    def forward(self, x):
        return self.layers(x)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x):
        return self.layers(x)


class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions_one_hot):
        inputs = torch.cat([states, actions_one_hot], dim=-1)
        return self.layers(inputs).squeeze(-1)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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
        return np.asarray(observation, dtype=np.float32)
    return np.asarray(reset_result, dtype=np.float32)


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_state, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return np.asarray(next_state, dtype=np.float32), float(reward), done, info
    next_state, reward, done, info = step_result
    return np.asarray(next_state, dtype=np.float32), float(reward), done, info


def moving_average(previous_value, current_value, factor=0.9):
    return previous_value * factor + current_value * (1.0 - factor)


def action_one_hot(actions, action_dim, device):
    actions_tensor = torch.as_tensor(actions, dtype=torch.long, device=device)
    return F.one_hot(actions_tensor, num_classes=action_dim).to(dtype=torch.float32)


def discounted_returns(rewards, gamma):
    returns = []
    running = 0.0
    for reward in reversed(rewards):
        running = float(reward) + gamma * running
        returns.append(running)
    returns.reverse()
    return returns


def normalize_tensor(values):
    if values.numel() <= 1:
        return values
    std = values.std(unbiased=False)
    if float(std.item()) < 1e-8:
        return values - values.mean()
    return (values - values.mean()) / (std + 1e-8)


def read_expert_hidden_dim():
    if not EXPERT_SUMMARY_PATH.exists():
        raise FileNotFoundError("Expert summary not found: {}".format(EXPERT_SUMMARY_PATH))
    summary = json.loads(EXPERT_SUMMARY_PATH.read_text(encoding="utf-8"))
    return int(summary["config"]["hidden_dim"])


def load_expert_policy(state_dim, action_dim, device):
    if not EXPERT_POLICY_PATH.exists():
        raise FileNotFoundError("Expert checkpoint not found: {}".format(EXPERT_POLICY_PATH))
    hidden_dim = read_expert_hidden_dim()
    model = Chapter7MLP(state_dim, hidden_dim, action_dim).to(device)
    state_dict = torch.load(EXPERT_POLICY_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def expert_predict_action(expert_policy, state, device):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        q_values = expert_policy(state_tensor)
    return int(torch.argmax(q_values, dim=1).item())


def collect_or_load_expert_demos(output_dir, demos_episodes, max_steps, seed, state_dim, action_dim, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    demos_path = output_dir / "expert_demos.npz"

    if demos_path.exists():
        cached = np.load(demos_path, allow_pickle=False)
        metadata = {
            "seed": int(cached["seed"][0]),
            "episodes": int(cached["episodes"][0]),
            "max_steps": int(cached["max_steps"][0]),
            "state_dim": int(cached["state_dim"][0]),
            "action_dim": int(cached["action_dim"][0]),
        }
        if metadata == {
            "seed": int(seed),
            "episodes": int(demos_episodes),
            "max_steps": int(max_steps),
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
        }:
            return {
                "states": cached["states"].astype(np.float32),
                "actions": cached["actions"].astype(np.int64),
                "episode_returns": cached["episode_returns"].astype(np.float32),
                "episode_lengths": cached["episode_lengths"].astype(np.int64),
                "path": demos_path,
            }

    env = create_env(seed=seed)
    expert_policy = load_expert_policy(state_dim=state_dim, action_dim=action_dim, device=device)
    states = []
    actions = []
    episode_returns = []
    episode_lengths = []

    try:
        for episode_idx in range(demos_episodes):
            state = reset_env(env, seed=seed + episode_idx)
            total_reward = 0.0
            steps = 0
            for _ in range(max_steps):
                action = expert_predict_action(expert_policy, state, device=device)
                next_state, reward, done, _ = step_env(env, action)
                states.append(state)
                actions.append(action)
                total_reward += reward
                steps += 1
                state = next_state
                if done:
                    break
            episode_returns.append(total_reward)
            episode_lengths.append(steps)
    finally:
        env.close()

    np.savez(
        demos_path,
        states=np.asarray(states, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.int64),
        episode_returns=np.asarray(episode_returns, dtype=np.float32),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        seed=np.asarray([seed], dtype=np.int64),
        episodes=np.asarray([demos_episodes], dtype=np.int64),
        max_steps=np.asarray([max_steps], dtype=np.int64),
        state_dim=np.asarray([state_dim], dtype=np.int64),
        action_dim=np.asarray([action_dim], dtype=np.int64),
    )
    return {
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.int64),
        "episode_returns": np.asarray(episode_returns, dtype=np.float32),
        "episode_lengths": np.asarray(episode_lengths, dtype=np.int64),
        "path": demos_path,
    }


def sample_policy_action(policy, state, device):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    logits = policy(state_tensor)
    distribution = torch.distributions.Categorical(logits=logits)
    action = distribution.sample()
    return int(action.item())


def predict_policy_action(policy, state, device):
    state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        logits = policy(state_tensor)
    return int(torch.argmax(logits, dim=1).item())


def evaluate_policy(env, policy, eval_episodes, max_steps, seed, device, render=False):
    rewards = []
    ma_rewards = []
    steps = []

    for episode_idx in range(eval_episodes):
        state = reset_env(env, seed=seed + 10_000 + episode_idx)
        episode_reward = 0.0
        episode_steps = 0
        for step_idx in range(max_steps):
            if render:
                env.render()
            action = predict_policy_action(policy, state, device=device)
            next_state, reward, done, _ = step_env(env, action)
            episode_reward += reward
            episode_steps = step_idx + 1
            state = next_state
            if done:
                break

        rewards.append(float(episode_reward))
        steps.append(int(episode_steps))
        if ma_rewards:
            ma_rewards.append(float(moving_average(ma_rewards[-1], episode_reward)))
        else:
            ma_rewards.append(float(episode_reward))

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "steps": steps,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def collect_policy_rollouts(env, policy, rollout_episodes, max_steps, seed, device):
    trajectories = []
    env_returns = []
    env_lengths = []

    for episode_idx in range(rollout_episodes):
        state = reset_env(env, seed=seed + episode_idx)
        trajectory_states = []
        trajectory_actions = []
        episode_reward = 0.0
        episode_steps = 0

        for step_idx in range(max_steps):
            action = sample_policy_action(policy, state, device=device)
            next_state, reward, done, _ = step_env(env, action)
            trajectory_states.append(state)
            trajectory_actions.append(action)
            episode_reward += reward
            episode_steps = step_idx + 1
            state = next_state
            if done:
                break

        trajectories.append(
            {
                "states": np.asarray(trajectory_states, dtype=np.float32),
                "actions": np.asarray(trajectory_actions, dtype=np.int64),
            }
        )
        env_returns.append(float(episode_reward))
        env_lengths.append(int(episode_steps))

    return {
        "trajectories": trajectories,
        "env_returns": env_returns,
        "env_lengths": env_lengths,
        "mean_env_return": float(np.mean(env_returns)) if env_returns else 0.0,
    }


def flatten_state_actions(trajectories):
    state_chunks = []
    action_chunks = []
    for trajectory in trajectories:
        if len(trajectory["states"]) == 0:
            continue
        state_chunks.append(trajectory["states"])
        action_chunks.append(trajectory["actions"])
    if not state_chunks:
        return (
            np.zeros((0, 4), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.concatenate(state_chunks, axis=0).astype(np.float32),
        np.concatenate(action_chunks, axis=0).astype(np.int64),
    )


def run_bc_epoch(policy, optimizer, expert_states, expert_actions, batch_size, device, rng):
    indices = rng.permutation(len(expert_states))
    losses = []

    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start : start + batch_size]
        states = torch.tensor(expert_states[batch_indices], dtype=torch.float32, device=device)
        actions = torch.tensor(expert_actions[batch_indices], dtype=torch.long, device=device)
        logits = policy(states)
        loss = F.cross_entropy(logits, actions)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def train_behavior_cloning(
    env,
    policy,
    optimizer,
    expert_states,
    expert_actions,
    train_episodes,
    bc_epochs,
    bc_batch_size,
    rollout_episodes,
    max_steps,
    seed,
    device,
):
    rng = np.random.default_rng(seed)
    rewards = []
    ma_rewards = []
    bc_losses = []

    for episode_idx in range(train_episodes):
        epoch_losses = []
        for _ in range(bc_epochs):
            epoch_losses.append(
                run_bc_epoch(
                    policy=policy,
                    optimizer=optimizer,
                    expert_states=expert_states,
                    expert_actions=expert_actions,
                    batch_size=bc_batch_size,
                    device=device,
                    rng=rng,
                )
            )

        train_eval = evaluate_policy(
            env=env,
            policy=policy,
            eval_episodes=rollout_episodes,
            max_steps=max_steps,
            seed=seed + 500 * episode_idx,
            device=device,
            render=False,
        )
        reward = train_eval["mean_reward"]
        rewards.append(float(reward))
        bc_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        bc_losses.append(bc_loss)
        if ma_rewards:
            ma_rewards.append(float(moving_average(ma_rewards[-1], reward)))
        else:
            ma_rewards.append(float(reward))

        print(
            "BC Episode {}/{} | reward: {:.1f} | loss: {:.4f}".format(
                episode_idx + 1,
                train_episodes,
                reward,
                bc_loss,
            )
        )

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "bc_losses": bc_losses,
    }


def train_discriminator(
    discriminator,
    optimizer,
    expert_states,
    expert_actions,
    learner_states,
    learner_actions,
    action_dim,
    batch_size,
    updates,
    device,
    rng,
):
    if len(learner_states) == 0:
        return 0.0

    losses = []
    for _ in range(updates):
        current_batch_size = min(batch_size, len(expert_states), len(learner_states))
        expert_indices = rng.integers(0, len(expert_states), size=current_batch_size)
        learner_indices = rng.integers(0, len(learner_states), size=current_batch_size)

        expert_state_tensor = torch.tensor(expert_states[expert_indices], dtype=torch.float32, device=device)
        expert_action_tensor = action_one_hot(expert_actions[expert_indices], action_dim=action_dim, device=device)
        learner_state_tensor = torch.tensor(learner_states[learner_indices], dtype=torch.float32, device=device)
        learner_action_tensor = action_one_hot(learner_actions[learner_indices], action_dim=action_dim, device=device)

        expert_logits = discriminator(expert_state_tensor, expert_action_tensor)
        learner_logits = discriminator(learner_state_tensor, learner_action_tensor)

        expert_loss = F.binary_cross_entropy_with_logits(expert_logits, torch.ones_like(expert_logits))
        learner_loss = F.binary_cross_entropy_with_logits(learner_logits, torch.zeros_like(learner_logits))
        loss = expert_loss + learner_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    return float(np.mean(losses)) if losses else 0.0


def update_policy_from_discriminator(
    policy,
    optimizer,
    discriminator,
    trajectories,
    action_dim,
    gamma,
    entropy_coef,
    policy_updates,
    device,
):
    flat_states = []
    flat_actions = []
    flat_returns = []
    imitation_rewards = []

    for trajectory in trajectories:
        if len(trajectory["states"]) == 0:
            continue

        state_tensor = torch.tensor(trajectory["states"], dtype=torch.float32, device=device)
        action_tensor = torch.tensor(trajectory["actions"], dtype=torch.long, device=device)
        action_tensor_one_hot = action_one_hot(action_tensor, action_dim=action_dim, device=device)

        with torch.no_grad():
            logits = discriminator(state_tensor, action_tensor_one_hot)
            rewards_tensor = F.softplus(logits)
            rewards = rewards_tensor.cpu().numpy().astype(np.float32)

        returns = discounted_returns(rewards.tolist(), gamma=gamma)
        flat_states.append(trajectory["states"])
        flat_actions.append(trajectory["actions"])
        flat_returns.append(np.asarray(returns, dtype=np.float32))
        imitation_rewards.extend(rewards.tolist())

    if not flat_states:
        return {
            "policy_loss": 0.0,
            "imitation_reward_mean": 0.0,
            "imitation_return_mean": 0.0,
        }

    states = np.concatenate(flat_states, axis=0).astype(np.float32)
    actions = np.concatenate(flat_actions, axis=0).astype(np.int64)
    returns = np.concatenate(flat_returns, axis=0).astype(np.float32)

    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    actions_tensor = torch.tensor(actions, dtype=torch.long, device=device)
    returns_tensor = normalize_tensor(torch.tensor(returns, dtype=torch.float32, device=device))

    losses = []
    for _ in range(policy_updates):
        logits = policy(states_tensor)
        distribution = torch.distributions.Categorical(logits=logits)
        log_probs = distribution.log_prob(actions_tensor)
        entropy = distribution.entropy().mean()
        loss = -(log_probs * returns_tensor).mean() - entropy_coef * entropy
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.item()))

    return {
        "policy_loss": float(np.mean(losses)) if losses else 0.0,
        "imitation_reward_mean": float(np.mean(imitation_rewards)) if imitation_rewards else 0.0,
        "imitation_return_mean": float(np.mean(returns)) if len(returns) else 0.0,
    }


def train_adversarial_imitation(
    env,
    policy,
    policy_optimizer,
    discriminator,
    discriminator_optimizer,
    expert_states,
    expert_actions,
    train_episodes,
    rollout_episodes,
    max_steps,
    irl_batch_size,
    discriminator_updates,
    policy_updates,
    gamma,
    entropy_coef,
    seed,
    device,
    action_dim,
    warm_start_epochs=0,
    bc_batch_size=128,
):
    rng = np.random.default_rng(seed)
    warm_start_losses = []
    if warm_start_epochs > 0:
        for _ in range(warm_start_epochs):
            warm_start_losses.append(
                run_bc_epoch(
                    policy=policy,
                    optimizer=policy_optimizer,
                    expert_states=expert_states,
                    expert_actions=expert_actions,
                    batch_size=bc_batch_size,
                    device=device,
                    rng=rng,
                )
            )

    rewards = []
    ma_rewards = []
    discriminator_losses = []
    policy_losses = []
    imitation_reward_means = []
    rollout_reward_means = []

    for episode_idx in range(train_episodes):
        rollout_result = collect_policy_rollouts(
            env=env,
            policy=policy,
            rollout_episodes=rollout_episodes,
            max_steps=max_steps,
            seed=seed + 1_000 * episode_idx,
            device=device,
        )
        learner_states, learner_actions = flatten_state_actions(rollout_result["trajectories"])
        discriminator_loss = train_discriminator(
            discriminator=discriminator,
            optimizer=discriminator_optimizer,
            expert_states=expert_states,
            expert_actions=expert_actions,
            learner_states=learner_states,
            learner_actions=learner_actions,
            action_dim=action_dim,
            batch_size=irl_batch_size,
            updates=discriminator_updates,
            device=device,
            rng=rng,
        )
        policy_update_result = update_policy_from_discriminator(
            policy=policy,
            optimizer=policy_optimizer,
            discriminator=discriminator,
            trajectories=rollout_result["trajectories"],
            action_dim=action_dim,
            gamma=gamma,
            entropy_coef=entropy_coef,
            policy_updates=policy_updates,
            device=device,
        )
        train_eval = evaluate_policy(
            env=env,
            policy=policy,
            eval_episodes=rollout_episodes,
            max_steps=max_steps,
            seed=seed + 20_000 + episode_idx,
            device=device,
            render=False,
        )

        reward = train_eval["mean_reward"]
        rewards.append(float(reward))
        rollout_reward_means.append(float(rollout_result["mean_env_return"]))
        discriminator_losses.append(float(discriminator_loss))
        policy_losses.append(float(policy_update_result["policy_loss"]))
        imitation_reward_means.append(float(policy_update_result["imitation_reward_mean"]))
        if ma_rewards:
            ma_rewards.append(float(moving_average(ma_rewards[-1], reward)))
        else:
            ma_rewards.append(float(reward))

        print(
            "IRL Episode {}/{} | reward: {:.1f} | rollout: {:.1f} | d_loss: {:.4f} | p_loss: {:.4f} | r_hat: {:.4f}".format(
                episode_idx + 1,
                train_episodes,
                reward,
                rollout_result["mean_env_return"],
                discriminator_loss,
                policy_update_result["policy_loss"],
                policy_update_result["imitation_reward_mean"],
            )
        )

    return {
        "rewards": rewards,
        "ma_rewards": ma_rewards,
        "discriminator_losses": discriminator_losses,
        "policy_losses": policy_losses,
        "imitation_reward_means": imitation_reward_means,
        "rollout_reward_means": rollout_reward_means,
        "warm_start_bc_loss": float(np.mean(warm_start_losses)) if warm_start_losses else None,
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


def save_summary(
    output_dir,
    mode,
    config,
    demos,
    train_result,
    eval_result,
    policy_path,
    discriminator_path,
):
    summary = {
        "mode": mode,
        "environment": ENV_NAME,
        "expert_policy_path": str(EXPERT_POLICY_PATH),
        "expert_demo_path": str(demos["path"]),
        "expert_demo_count": int(len(demos["actions"])),
        "expert_demo_mean_reward": float(np.mean(demos["episode_returns"])) if len(demos["episode_returns"]) else 0.0,
        "config": asdict(config),
        "train_rewards": train_result["rewards"],
        "train_ma_rewards": train_result["ma_rewards"],
        "eval_rewards": eval_result["rewards"],
        "eval_ma_rewards": eval_result["ma_rewards"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else 0.0,
        "eval_mean_reward": eval_result["mean_reward"],
        "policy_path": str(policy_path),
        "discriminator_path": str(discriminator_path) if discriminator_path is not None else None,
    }
    for key, value in train_result.items():
        if key in {"rewards", "ma_rewards"}:
            continue
        summary[key] = value

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
    policy_learning_rate=Config.policy_learning_rate,
    discriminator_learning_rate=Config.discriminator_learning_rate,
    gamma=Config.gamma,
    entropy_coef=Config.entropy_coef,
    bc_batch_size=Config.bc_batch_size,
    irl_batch_size=Config.irl_batch_size,
    demos_episodes=Config.demos_episodes,
    bc_epochs=Config.bc_epochs,
    rollout_episodes=Config.rollout_episodes,
    discriminator_updates=Config.discriminator_updates,
    policy_updates=Config.policy_updates,
    render=False,
):
    if mode not in VALID_MODES:
        raise ValueError("Unsupported mode: {}".format(mode))

    base_output_dir = Path(output_dir)
    mode_output_dir = base_output_dir / mode
    base_output_dir.mkdir(parents=True, exist_ok=True)
    mode_output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)

    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        demos = collect_or_load_expert_demos(
            output_dir=base_output_dir,
            demos_episodes=demos_episodes,
            max_steps=max_steps,
            seed=seed,
            state_dim=state_dim,
            action_dim=action_dim,
            device=device,
        )

        policy = PolicyNetwork(state_dim, hidden_dim, action_dim).to(device)
        policy_optimizer = optim.Adam(policy.parameters(), lr=policy_learning_rate)
        discriminator = None

        if mode == "bc":
            train_result = train_behavior_cloning(
                env=env,
                policy=policy,
                optimizer=policy_optimizer,
                expert_states=demos["states"],
                expert_actions=demos["actions"],
                train_episodes=train_episodes,
                bc_epochs=bc_epochs,
                bc_batch_size=bc_batch_size,
                rollout_episodes=rollout_episodes,
                max_steps=max_steps,
                seed=seed,
                device=device,
            )
        else:
            discriminator = Discriminator(state_dim, action_dim, hidden_dim).to(device)
            discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=discriminator_learning_rate)
            warm_start_epochs = bc_epochs if mode == "hybrid" else 0
            train_result = train_adversarial_imitation(
                env=env,
                policy=policy,
                policy_optimizer=policy_optimizer,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                expert_states=demos["states"],
                expert_actions=demos["actions"],
                train_episodes=train_episodes,
                rollout_episodes=rollout_episodes,
                max_steps=max_steps,
                irl_batch_size=irl_batch_size,
                discriminator_updates=discriminator_updates,
                policy_updates=policy_updates,
                gamma=gamma,
                entropy_coef=entropy_coef,
                seed=seed,
                device=device,
                action_dim=action_dim,
                warm_start_epochs=warm_start_epochs,
                bc_batch_size=bc_batch_size,
            )

        eval_result = evaluate_policy(
            env=eval_env,
            policy=policy,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            seed=seed,
            device=device,
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

        policy_path = mode_output_dir / "policy_net.pth"
        torch.save(policy.state_dict(), policy_path)
        discriminator_path = None
        if discriminator is not None:
            discriminator_path = mode_output_dir / "discriminator.pth"
            torch.save(discriminator.state_dict(), discriminator_path)

        config = Config(
            mode=mode,
            train_episodes=train_episodes,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            hidden_dim=hidden_dim,
            policy_learning_rate=policy_learning_rate,
            discriminator_learning_rate=discriminator_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
            bc_batch_size=bc_batch_size,
            irl_batch_size=irl_batch_size,
            demos_episodes=demos_episodes,
            bc_epochs=bc_epochs,
            rollout_episodes=rollout_episodes,
            discriminator_updates=discriminator_updates,
            policy_updates=policy_updates,
            seed=seed,
            render=render,
            output_dir=str(mode_output_dir),
        )
        summary_path = save_summary(
            output_dir=mode_output_dir,
            mode=mode,
            config=config,
            demos=demos,
            train_result=train_result,
            eval_result=eval_result,
            policy_path=policy_path,
            discriminator_path=discriminator_path,
        )

        return {
            "mode": mode,
            "train_result": train_result,
            "eval_result": eval_result,
            "train_reward_path": train_reward_path,
            "train_ma_reward_path": train_ma_reward_path,
            "eval_reward_path": eval_reward_path,
            "policy_path": policy_path,
            "discriminator_path": discriminator_path,
            "summary_path": summary_path,
            "expert_demo_path": demos["path"],
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
    policy_learning_rate=Config.policy_learning_rate,
    discriminator_learning_rate=Config.discriminator_learning_rate,
    gamma=Config.gamma,
    entropy_coef=Config.entropy_coef,
    bc_batch_size=Config.bc_batch_size,
    irl_batch_size=Config.irl_batch_size,
    demos_episodes=Config.demos_episodes,
    bc_epochs=Config.bc_epochs,
    rollout_episodes=Config.rollout_episodes,
    discriminator_updates=Config.discriminator_updates,
    policy_updates=Config.policy_updates,
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
            policy_learning_rate=policy_learning_rate,
            discriminator_learning_rate=discriminator_learning_rate,
            gamma=gamma,
            entropy_coef=entropy_coef,
            bc_batch_size=bc_batch_size,
            irl_batch_size=irl_batch_size,
            demos_episodes=demos_episodes,
            bc_epochs=bc_epochs,
            rollout_episodes=rollout_episodes,
            discriminator_updates=discriminator_updates,
            policy_updates=policy_updates,
            render=False,
        )

    compare_train_path = plot_series_map(
        {mode: result["train_result"]["ma_rewards"] for mode, result in results.items()},
        "Chapter 11 Imitation Learning: Training Moving Average Rewards",
        output_dir / "compare_train_moving_average_rewards",
    )
    compare_eval_path = plot_series_map(
        {mode: result["eval_result"]["rewards"] for mode, result in results.items()},
        "Chapter 11 Imitation Learning: Evaluation Rewards",
        output_dir / "compare_eval_rewards",
    )
    summary = {
        "modes": {
            mode: {
                "train_mean_reward": float(np.mean(result["train_result"]["rewards"]))
                if result["train_result"]["rewards"]
                else 0.0,
                "eval_mean_reward": result["eval_result"]["mean_reward"],
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
    parser = argparse.ArgumentParser(description="Chapter 11 imitation learning on CartPole-v1")
    parser.add_argument("--mode", choices=("all",) + VALID_MODES, default=Config.mode)
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--policy-learning-rate", type=float, default=Config.policy_learning_rate)
    parser.add_argument("--discriminator-learning-rate", type=float, default=Config.discriminator_learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--entropy-coef", type=float, default=Config.entropy_coef)
    parser.add_argument("--bc-batch-size", type=int, default=Config.bc_batch_size)
    parser.add_argument("--irl-batch-size", type=int, default=Config.irl_batch_size)
    parser.add_argument("--demos-episodes", type=int, default=Config.demos_episodes)
    parser.add_argument("--bc-epochs", type=int, default=Config.bc_epochs)
    parser.add_argument("--rollout-episodes", type=int, default=Config.rollout_episodes)
    parser.add_argument("--discriminator-updates", type=int, default=Config.discriminator_updates)
    parser.add_argument("--policy-updates", type=int, default=Config.policy_updates)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.mode == "all":
        result = run_all_modes(
            output_dir=Path(args.output_dir),
            train_episodes=args.train_episodes,
            eval_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            hidden_dim=args.hidden_dim,
            policy_learning_rate=args.policy_learning_rate,
            discriminator_learning_rate=args.discriminator_learning_rate,
            gamma=args.gamma,
            entropy_coef=args.entropy_coef,
            bc_batch_size=args.bc_batch_size,
            irl_batch_size=args.irl_batch_size,
            demos_episodes=args.demos_episodes,
            bc_epochs=args.bc_epochs,
            rollout_episodes=args.rollout_episodes,
            discriminator_updates=args.discriminator_updates,
            policy_updates=args.policy_updates,
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
        policy_learning_rate=args.policy_learning_rate,
        discriminator_learning_rate=args.discriminator_learning_rate,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        bc_batch_size=args.bc_batch_size,
        irl_batch_size=args.irl_batch_size,
        demos_episodes=args.demos_episodes,
        bc_epochs=args.bc_epochs,
        rollout_episodes=args.rollout_episodes,
        discriminator_updates=args.discriminator_updates,
        policy_updates=args.policy_updates,
        render=args.render,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print("  expert demos: {}".format(result["expert_demo_path"]))
    print("  train rewards: {}".format(result["train_reward_path"]))
    print("  train moving average rewards: {}".format(result["train_ma_reward_path"]))
    print("  eval rewards: {}".format(result["eval_reward_path"]))
    print("  policy: {}".format(result["policy_path"]))
    if result["discriminator_path"] is not None:
        print("  discriminator: {}".format(result["discriminator_path"]))
    print("  summary: {}".format(result["summary_path"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
