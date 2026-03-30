import argparse
import copy
import json
import random
from collections import deque, namedtuple
from dataclasses import asdict, dataclass
from pathlib import Path

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


ENV_NAME = "CartPole-v0"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ddqn_cartpole"
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    train_episodes: int = 220
    eval_episodes: int = 10
    max_steps: int = 500
    hidden_dim: int = 128
    learning_rate: float = 0.0005
    gamma: float = 0.99
    memory_capacity: int = 10000
    batch_size: int = 64
    min_memory_size: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay: int = 5000
    target_update: int = 4
    seed: int = 1
    render: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                int(action),
                float(reward),
                np.asarray(next_state, dtype=np.float32),
                bool(done),
            )
        )

    def sample(self, batch_size):
        transitions = random.sample(self.memory, batch_size)
        batch = Transition(*zip(*transitions))
        return {
            "states": np.asarray(batch.state, dtype=np.float32),
            "actions": np.asarray(batch.action, dtype=np.int64),
            "rewards": np.asarray(batch.reward, dtype=np.float32),
            "next_states": np.asarray(batch.next_state, dtype=np.float32),
            "dones": np.asarray(batch.done, dtype=np.float32),
        }

    def __len__(self):
        return len(self.memory)


class MLP(nn.Module):
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


class DDQNAgent:
    def __init__(self, n_states, n_actions, cfg, device):
        self.n_actions = n_actions
        self.device = device
        self.gamma = cfg.gamma
        self.batch_size = cfg.batch_size
        self.min_memory_size = cfg.min_memory_size
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.sample_count = 0
        self.epsilon = cfg.epsilon_start
        self.memory = ReplayMemory(cfg.memory_capacity)

        self.policy_net = MLP(n_states, cfg.hidden_dim, n_actions).to(device)
        self.target_net = MLP(n_states, cfg.hidden_dim, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.learning_rate)
        self.loss_fn = nn.SmoothL1Loss()

    def _update_epsilon(self):
        self.sample_count += 1
        decay_ratio = min(self.sample_count / max(self.epsilon_decay, 1), 1.0)
        self.epsilon = self.epsilon_start + decay_ratio * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state, greedy=False):
        state_tensor = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        if greedy:
            with torch.no_grad():
                action = self.policy_net(state_tensor).argmax(dim=1).item()
            return int(action)

        self._update_epsilon()
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)

        with torch.no_grad():
            action = self.policy_net(state_tensor).argmax(dim=1).item()
        return int(action)

    def update(self):
        if len(self.memory) < max(self.batch_size, self.min_memory_size):
            return None

        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(batch["states"], device=self.device, dtype=torch.float32)
        actions = torch.tensor(batch["actions"], device=self.device, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(batch["rewards"], device=self.device, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], device=self.device, dtype=torch.float32)
        dones = torch.tensor(batch["dones"], device=self.device, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1.0 - dones)

        loss = self.loss_fn(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 5.0)
        self.optimizer.step()
        return float(loss.item())


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


def plot_series(values, title, ylabel, output_path):
    output_path = Path(output_path)
    figure = plt.figure(figsize=(8, 4.5))
    axis = figure.add_subplot(111)
    axis.plot(range(1, len(values) + 1), values, linewidth=1.8)
    axis.set_title(title)
    axis.set_xlabel("Episode")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)
    return output_path


def maybe_create_writer(output_dir):
    if SummaryWriter is None:
        return None
    try:
        return SummaryWriter(log_dir=str(Path(output_dir) / "tensorboard"))
    except Exception:
        return None


def train(env, agent, cfg, writer=None):
    rewards = []
    moving_average_rewards = []
    steps = []
    losses = []
    best_ma_reward = float("-inf")
    best_policy_state = copy.deepcopy(agent.policy_net.state_dict())
    best_target_state = copy.deepcopy(agent.target_net.state_dict())

    for i_episode in range(1, cfg.train_episodes + 1):
        state = reset_env(env, seed=cfg.seed + i_episode)
        ep_reward = 0.0
        episode_steps = 0
        episode_losses = []

        for i_step in range(1, cfg.max_steps + 1):
            action = agent.select_action(state)
            next_state, reward, done, _ = step_env(env, action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)
            episode_steps = i_step
            if done:
                break

        if i_episode % cfg.target_update == 0:
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        steps.append(int(episode_steps))
        rewards.append(float(ep_reward))
        if i_episode == 1:
            moving_average_rewards.append(float(ep_reward))
        else:
            moving_average_rewards.append(float(moving_average(moving_average_rewards[-1], ep_reward)))
        losses.append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        if moving_average_rewards[-1] > best_ma_reward:
            best_ma_reward = moving_average_rewards[-1]
            best_policy_state = copy.deepcopy(agent.policy_net.state_dict())
            best_target_state = copy.deepcopy(agent.target_net.state_dict())

        if writer is not None:
            writer.add_scalar("train/reward", ep_reward, i_episode)
            writer.add_scalar("train/moving_average_reward", moving_average_rewards[-1], i_episode)
            writer.add_scalar("train/steps", episode_steps, i_episode)
            writer.add_scalar("train/epsilon", agent.epsilon, i_episode)
            writer.add_scalar("train/loss", losses[-1], i_episode)

        print(
            "Episode:",
            i_episode,
            "Reward: %i" % int(ep_reward),
            "n_steps:",
            episode_steps,
            "done:",
            done,
            "Explore: %.2f" % agent.epsilon,
        )

    return {
        "rewards": rewards,
        "moving_average_rewards": moving_average_rewards,
        "steps": steps,
        "losses": losses,
        "best_ma_reward": best_ma_reward,
        "best_policy_state": best_policy_state,
        "best_target_state": best_target_state,
    }


def evaluate(env, agent, cfg, writer=None):
    rewards = []
    moving_average_rewards = []
    steps = []

    for i_episode in range(1, cfg.eval_episodes + 1):
        state = reset_env(env, seed=cfg.seed + 10_000 + i_episode)
        ep_reward = 0.0
        episode_steps = 0

        for i_step in range(1, cfg.max_steps + 1):
            if cfg.render:
                env.render()
            action = agent.select_action(state, greedy=True)
            next_state, reward, done, _ = step_env(env, action)
            ep_reward += reward
            state = next_state
            episode_steps = i_step
            if done:
                break

        steps.append(int(episode_steps))
        rewards.append(float(ep_reward))
        if i_episode == 1:
            moving_average_rewards.append(float(ep_reward))
        else:
            moving_average_rewards.append(float(moving_average(moving_average_rewards[-1], ep_reward)))

        if writer is not None:
            writer.add_scalar("eval/reward", ep_reward, i_episode)
            writer.add_scalar("eval/moving_average_reward", moving_average_rewards[-1], i_episode)
            writer.add_scalar("eval/steps", episode_steps, i_episode)

    return {
        "rewards": rewards,
        "moving_average_rewards": moving_average_rewards,
        "steps": steps,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
    }


def save_summary(output_dir, cfg, train_result, eval_result, policy_path, target_path, tensorboard_dir):
    summary = {
        "environment": ENV_NAME,
        "config": asdict(cfg),
        "train_rewards": train_result["rewards"],
        "train_moving_average_rewards": train_result["moving_average_rewards"],
        "train_steps": train_result["steps"],
        "train_losses": train_result["losses"],
        "best_train_moving_average_reward": train_result["best_ma_reward"],
        "eval_rewards": eval_result["rewards"],
        "eval_moving_average_rewards": eval_result["moving_average_rewards"],
        "eval_steps": eval_result["steps"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        "eval_mean_reward": eval_result["mean_reward"],
        "policy_net_path": str(policy_path),
        "target_net_path": str(target_path),
        "tensorboard_dir": str(tensorboard_dir) if tensorboard_dir is not None else None,
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
    memory_capacity=10000,
    batch_size=64,
    min_memory_size=500,
    epsilon_start=1.0,
    epsilon_end=0.01,
    epsilon_decay=2000,
    target_update=4,
    render=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cfg = Config(
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        gamma=gamma,
        memory_capacity=memory_capacity,
        batch_size=batch_size,
        min_memory_size=min_memory_size,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay=epsilon_decay,
        target_update=target_update,
        seed=seed,
        render=render,
        output_dir=str(output_dir),
    )

    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)
    writer = maybe_create_writer(output_dir)
    try:
        n_states = env.observation_space.shape[0]
        n_actions = env.action_space.n
        agent = DDQNAgent(
            n_states=n_states,
            n_actions=n_actions,
            cfg=cfg,
            device=device,
        )

        train_result = train(env=env, agent=agent, cfg=cfg, writer=writer)
        agent.policy_net.load_state_dict(train_result["best_policy_state"])
        agent.target_net.load_state_dict(train_result["best_target_state"])
        eval_result = evaluate(env=eval_env, agent=agent, cfg=cfg, writer=writer)

        train_rewards_path = plot_series(
            train_result["rewards"],
            "DDQN Train Rewards",
            "Reward",
            output_dir / "rewards_train.png",
        )
        train_ma_rewards_path = plot_series(
            train_result["moving_average_rewards"],
            "DDQN Train Moving Average Rewards",
            "Reward",
            output_dir / "moving_average_rewards_train.png",
        )
        train_steps_path = plot_series(
            train_result["steps"],
            "DDQN Train Steps",
            "Steps",
            output_dir / "steps_train.png",
        )
        eval_rewards_path = plot_series(
            eval_result["rewards"],
            "DDQN Eval Rewards",
            "Reward",
            output_dir / "rewards_eval.png",
        )
        eval_ma_rewards_path = plot_series(
            eval_result["moving_average_rewards"],
            "DDQN Eval Moving Average Rewards",
            "Reward",
            output_dir / "moving_average_rewards_eval.png",
        )
        eval_steps_path = plot_series(
            eval_result["steps"],
            "DDQN Eval Steps",
            "Steps",
            output_dir / "steps_eval.png",
        )

        policy_path = output_dir / "policy_net.pth"
        target_path = output_dir / "target_net.pth"
        torch.save(agent.policy_net.state_dict(), policy_path)
        torch.save(agent.target_net.state_dict(), target_path)

        tensorboard_dir = output_dir / "tensorboard" if writer is not None else None
        summary_path = save_summary(
            output_dir=output_dir,
            cfg=cfg,
            train_result=train_result,
            eval_result=eval_result,
            policy_path=policy_path,
            target_path=target_path,
            tensorboard_dir=tensorboard_dir,
        )

        return {
            "train_result": train_result,
            "eval_result": eval_result,
            "train_rewards_path": train_rewards_path,
            "train_ma_rewards_path": train_ma_rewards_path,
            "train_steps_path": train_steps_path,
            "eval_rewards_path": eval_rewards_path,
            "eval_ma_rewards_path": eval_ma_rewards_path,
            "eval_steps_path": eval_steps_path,
            "policy_path": policy_path,
            "target_path": target_path,
            "summary_path": summary_path,
            "tensorboard_dir": tensorboard_dir,
        }
    finally:
        if writer is not None:
            writer.flush()
            writer.close()
        env.close()
        eval_env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Project2 Double DQN on CartPole-v0")
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--memory-capacity", type=int, default=Config.memory_capacity)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--min-memory-size", type=int, default=Config.min_memory_size)
    parser.add_argument("--epsilon-start", type=float, default=Config.epsilon_start)
    parser.add_argument("--epsilon-end", type=float, default=Config.epsilon_end)
    parser.add_argument("--epsilon-decay", type=int, default=Config.epsilon_decay)
    parser.add_argument("--target-update", type=int, default=Config.target_update)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--output-dir", type=str, default=Config.output_dir)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    result = run_experiment(
        output_dir=Path(args.output_dir),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        gamma=args.gamma,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        min_memory_size=args.min_memory_size,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update=args.target_update,
        render=args.render,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print("  train rewards: {}".format(result["train_rewards_path"]))
    print("  train moving average rewards: {}".format(result["train_ma_rewards_path"]))
    print("  train steps: {}".format(result["train_steps_path"]))
    print("  eval rewards: {}".format(result["eval_rewards_path"]))
    print("  eval moving average rewards: {}".format(result["eval_ma_rewards_path"]))
    print("  eval steps: {}".format(result["eval_steps_path"]))
    print("  policy net: {}".format(result["policy_path"]))
    print("  target net: {}".format(result["target_path"]))
    print("  summary: {}".format(result["summary_path"]))
    if result["tensorboard_dir"] is not None:
        print("  tensorboard: {}".format(result["tensorboard_dir"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
