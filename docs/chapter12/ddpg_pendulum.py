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
from gym import spaces

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:
    SummaryWriter = None


ENV_NAME = "Pendulum-v1"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ddpg_pendulum"
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))


if not hasattr(np, "bool8"):
    np.bool8 = np.bool_


@dataclass
class Config:
    train_episodes: int = 120
    eval_episodes: int = 10
    max_steps: int = 200
    hidden_dim: int = 128
    actor_learning_rate: float = 0.001
    critic_learning_rate: float = 0.001
    gamma: float = 0.99
    tau: float = 0.005
    memory_capacity: int = 100000
    batch_size: int = 128
    min_memory_size: int = 1000
    initial_random_steps: int = 1000
    ou_theta: float = 0.15
    ou_sigma: float = 0.2
    noise_scale_start: float = 1.0
    noise_scale_end: float = 0.1
    seed: int = 1
    render: bool = False
    output_dir: str = str(DEFAULT_OUTPUT_DIR)


class NormalizedActions(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.low = self.env.action_space.low.astype(np.float32)
        self.high = self.env.action_space.high.astype(np.float32)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=self.env.action_space.shape,
            dtype=np.float32,
        )

    def action(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        scaled = self.low + (action + 1.0) * 0.5 * (self.high - self.low)
        return np.clip(scaled, self.low, self.high)

    def reverse_action(self, action):
        action = np.asarray(action, dtype=np.float32)
        normalized = 2.0 * (action - self.low) / (self.high - self.low) - 1.0
        return np.clip(normalized, self.action_space.low, self.action_space.high)


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.memory.append(
            Transition(
                np.asarray(state, dtype=np.float32),
                np.asarray(action, dtype=np.float32),
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
            "actions": np.asarray(batch.action, dtype=np.float32),
            "rewards": np.asarray(batch.reward, dtype=np.float32),
            "next_states": np.asarray(batch.next_state, dtype=np.float32),
            "dones": np.asarray(batch.done, dtype=np.float32),
        }

    def __len__(self):
        return len(self.memory)


class OUNoise:
    def __init__(self, action_dim, mu=0.0, theta=0.15, sigma=0.2):
        self.action_dim = int(action_dim)
        self.mu = float(mu)
        self.theta = float(theta)
        self.sigma = float(sigma)
        self.state = np.full(self.action_dim, self.mu, dtype=np.float32)

    def reset(self):
        self.state.fill(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = (self.state + dx).astype(np.float32)
        return self.state.copy()


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),
        )

    def forward(self, states):
        return self.net(states)


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, states, actions):
        return self.net(torch.cat([states, actions], dim=-1))


class DDPGAgent:
    def __init__(self, state_dim, action_dim, cfg, device):
        self.device = device
        self.action_dim = action_dim
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.min_memory_size = cfg.min_memory_size
        self.initial_random_steps = cfg.initial_random_steps
        self.sample_count = 0

        self.memory = ReplayBuffer(cfg.memory_capacity)
        self.noise = OUNoise(action_dim, theta=cfg.ou_theta, sigma=cfg.ou_sigma)

        self.actor = ActorNetwork(state_dim, cfg.hidden_dim, action_dim).to(device)
        self.critic = CriticNetwork(state_dim, cfg.hidden_dim, action_dim).to(device)
        self.target_actor = ActorNetwork(state_dim, cfg.hidden_dim, action_dim).to(device)
        self.target_critic = CriticNetwork(state_dim, cfg.hidden_dim, action_dim).to(device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_learning_rate)
        self.critic_loss_fn = nn.MSELoss()

    def reset_noise(self):
        self.noise.reset()

    def select_action(self, state, action_space, noise_scale=0.0, greedy=False):
        self.sample_count += 1
        if not greedy and self.sample_count <= self.initial_random_steps:
            return np.asarray(action_space.sample(), dtype=np.float32)

        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        if not greedy and noise_scale > 0.0:
            action = action + noise_scale * self.noise.sample()
        return np.clip(action, action_space.low, action_space.high).astype(np.float32)

    def update(self):
        if len(self.memory) < max(self.batch_size, self.min_memory_size):
            return None

        batch = self.memory.sample(self.batch_size)
        states = torch.tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.tensor(batch["actions"], dtype=torch.float32, device=self.device)
        rewards = torch.tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q = self.target_critic(next_states, next_actions)
            target_values = rewards + self.gamma * target_q * (1.0 - dones)

        current_q = self.critic(states, actions)
        critic_loss = self.critic_loss_fn(current_q, target_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 5.0)
        self.critic_optimizer.step()

        actor_actions = self.actor(states)
        actor_loss = -self.critic(states, actor_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 5.0)
        self.actor_optimizer.step()

        self.soft_update(self.actor, self.target_actor)
        self.soft_update(self.critic, self.target_critic)
        return {
            "actor_loss": float(actor_loss.item()),
            "critic_loss": float(critic_loss.item()),
        }

    def soft_update(self, source, target):
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1.0 - self.tau) * target_param.data)


def create_env(seed=None, render_mode=None):
    make_kwargs = {}
    if render_mode is not None:
        make_kwargs["render_mode"] = render_mode

    try:
        env = gym.make(ENV_NAME, disable_env_checker=True, **make_kwargs)
    except TypeError:
        env = gym.make(ENV_NAME, **make_kwargs)
    env = NormalizedActions(env)

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
    figure.savefig(output_path, dpi=150)
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
    actor_losses = []
    critic_losses = []
    noise_scales = []
    best_ma_reward = float("-inf")
    best_states = {
        "actor": copy.deepcopy(agent.actor.state_dict()),
        "critic": copy.deepcopy(agent.critic.state_dict()),
        "target_actor": copy.deepcopy(agent.target_actor.state_dict()),
        "target_critic": copy.deepcopy(agent.target_critic.state_dict()),
    }

    for i_episode in range(1, cfg.train_episodes + 1):
        state = reset_env(env, seed=cfg.seed + i_episode)
        agent.reset_noise()
        progress = (i_episode - 1) / max(cfg.train_episodes - 1, 1)
        noise_scale = cfg.noise_scale_start + progress * (cfg.noise_scale_end - cfg.noise_scale_start)
        ep_reward = 0.0
        episode_steps = 0
        episode_actor_losses = []
        episode_critic_losses = []

        for i_step in range(1, cfg.max_steps + 1):
            action = agent.select_action(
                state=state,
                action_space=env.action_space,
                noise_scale=noise_scale,
                greedy=False,
            )
            next_state, reward, done, _ = step_env(env, action)
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            state = next_state
            update_result = agent.update()
            if update_result is not None:
                episode_actor_losses.append(update_result["actor_loss"])
                episode_critic_losses.append(update_result["critic_loss"])
            episode_steps = i_step
            if done:
                break

        steps.append(int(episode_steps))
        rewards.append(float(ep_reward))
        noise_scales.append(float(noise_scale))
        if i_episode == 1:
            moving_average_rewards.append(float(ep_reward))
        else:
            moving_average_rewards.append(float(moving_average(moving_average_rewards[-1], ep_reward)))

        actor_loss = float(np.mean(episode_actor_losses)) if episode_actor_losses else 0.0
        critic_loss = float(np.mean(episode_critic_losses)) if episode_critic_losses else 0.0
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)

        if moving_average_rewards[-1] > best_ma_reward:
            best_ma_reward = moving_average_rewards[-1]
            best_states = {
                "actor": copy.deepcopy(agent.actor.state_dict()),
                "critic": copy.deepcopy(agent.critic.state_dict()),
                "target_actor": copy.deepcopy(agent.target_actor.state_dict()),
                "target_critic": copy.deepcopy(agent.target_critic.state_dict()),
            }

        if writer is not None:
            writer.add_scalar("train/reward", ep_reward, i_episode)
            writer.add_scalar("train/moving_average_reward", moving_average_rewards[-1], i_episode)
            writer.add_scalar("train/steps", episode_steps, i_episode)
            writer.add_scalar("train/noise_scale", noise_scale, i_episode)
            writer.add_scalar("train/actor_loss", actor_loss, i_episode)
            writer.add_scalar("train/critic_loss", critic_loss, i_episode)

        print(
            "Episode:",
            i_episode,
            "Reward: %.2f" % ep_reward,
            "n_steps:",
            episode_steps,
            "done:",
            done,
            "Noise: %.2f" % noise_scale,
        )

    return {
        "rewards": rewards,
        "moving_average_rewards": moving_average_rewards,
        "steps": steps,
        "actor_losses": actor_losses,
        "critic_losses": critic_losses,
        "noise_scales": noise_scales,
        "best_ma_reward": float(best_ma_reward),
        "best_states": best_states,
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
            action = agent.select_action(
                state=state,
                action_space=env.action_space,
                noise_scale=0.0,
                greedy=True,
            )
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


def save_summary(
    output_dir,
    cfg,
    train_result,
    eval_result,
    actor_path,
    critic_path,
    target_actor_path,
    target_critic_path,
    tensorboard_dir,
):
    summary = {
        "environment": ENV_NAME,
        "config": asdict(cfg),
        "train_rewards": train_result["rewards"],
        "train_moving_average_rewards": train_result["moving_average_rewards"],
        "train_steps": train_result["steps"],
        "train_actor_losses": train_result["actor_losses"],
        "train_critic_losses": train_result["critic_losses"],
        "train_noise_scales": train_result["noise_scales"],
        "best_train_moving_average_reward": train_result["best_ma_reward"],
        "eval_rewards": eval_result["rewards"],
        "eval_moving_average_rewards": eval_result["moving_average_rewards"],
        "eval_steps": eval_result["steps"],
        "train_mean_reward": float(np.mean(train_result["rewards"])) if train_result["rewards"] else None,
        "eval_mean_reward": eval_result["mean_reward"],
        "actor_path": str(actor_path),
        "critic_path": str(critic_path),
        "target_actor_path": str(target_actor_path),
        "target_critic_path": str(target_critic_path),
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
    actor_learning_rate=0.001,
    critic_learning_rate=0.001,
    gamma=0.99,
    tau=0.005,
    memory_capacity=100000,
    batch_size=128,
    min_memory_size=1000,
    initial_random_steps=1000,
    ou_theta=0.15,
    ou_sigma=0.2,
    noise_scale_start=1.0,
    noise_scale_end=0.1,
    render=False,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cfg = Config(
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        max_steps=max_steps,
        hidden_dim=hidden_dim,
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        gamma=gamma,
        tau=tau,
        memory_capacity=memory_capacity,
        batch_size=batch_size,
        min_memory_size=min_memory_size,
        initial_random_steps=initial_random_steps,
        ou_theta=ou_theta,
        ou_sigma=ou_sigma,
        noise_scale_start=noise_scale_start,
        noise_scale_end=noise_scale_end,
        seed=seed,
        render=render,
        output_dir=str(output_dir),
    )

    env = create_env(seed=seed)
    eval_env = create_env(seed=seed, render_mode="human" if render else None)
    writer = maybe_create_writer(output_dir)
    try:
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        agent = DDPGAgent(state_dim=state_dim, action_dim=action_dim, cfg=cfg, device=device)

        train_result = train(env=env, agent=agent, cfg=cfg, writer=writer)
        agent.actor.load_state_dict(train_result["best_states"]["actor"])
        agent.critic.load_state_dict(train_result["best_states"]["critic"])
        agent.target_actor.load_state_dict(train_result["best_states"]["target_actor"])
        agent.target_critic.load_state_dict(train_result["best_states"]["target_critic"])
        eval_result = evaluate(env=eval_env, agent=agent, cfg=cfg, writer=writer)

        train_rewards_path = plot_series(
            train_result["rewards"],
            "DDPG Train Rewards",
            "Reward",
            output_dir / "rewards_train.png",
        )
        train_ma_rewards_path = plot_series(
            train_result["moving_average_rewards"],
            "DDPG Train Moving Average Rewards",
            "Reward",
            output_dir / "moving_average_rewards_train.png",
        )
        train_steps_path = plot_series(
            train_result["steps"],
            "DDPG Train Steps",
            "Steps",
            output_dir / "steps_train.png",
        )
        eval_rewards_path = plot_series(
            eval_result["rewards"],
            "DDPG Eval Rewards",
            "Reward",
            output_dir / "rewards_eval.png",
        )
        eval_ma_rewards_path = plot_series(
            eval_result["moving_average_rewards"],
            "DDPG Eval Moving Average Rewards",
            "Reward",
            output_dir / "moving_average_rewards_eval.png",
        )
        eval_steps_path = plot_series(
            eval_result["steps"],
            "DDPG Eval Steps",
            "Steps",
            output_dir / "steps_eval.png",
        )

        actor_path = output_dir / "actor.pth"
        critic_path = output_dir / "critic.pth"
        target_actor_path = output_dir / "target_actor.pth"
        target_critic_path = output_dir / "target_critic.pth"
        torch.save(agent.actor.state_dict(), actor_path)
        torch.save(agent.critic.state_dict(), critic_path)
        torch.save(agent.target_actor.state_dict(), target_actor_path)
        torch.save(agent.target_critic.state_dict(), target_critic_path)

        tensorboard_dir = output_dir / "tensorboard" if writer is not None else None
        summary_path = save_summary(
            output_dir=output_dir,
            cfg=cfg,
            train_result=train_result,
            eval_result=eval_result,
            actor_path=actor_path,
            critic_path=critic_path,
            target_actor_path=target_actor_path,
            target_critic_path=target_critic_path,
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
            "actor_path": actor_path,
            "critic_path": critic_path,
            "target_actor_path": target_actor_path,
            "target_critic_path": target_critic_path,
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
    parser = argparse.ArgumentParser(description="Project3 DDPG on Pendulum-v1")
    parser.add_argument("--train-episodes", type=int, default=Config.train_episodes)
    parser.add_argument("--eval-episodes", type=int, default=Config.eval_episodes)
    parser.add_argument("--max-steps", type=int, default=Config.max_steps)
    parser.add_argument("--hidden-dim", type=int, default=Config.hidden_dim)
    parser.add_argument("--actor-learning-rate", type=float, default=Config.actor_learning_rate)
    parser.add_argument("--critic-learning-rate", type=float, default=Config.critic_learning_rate)
    parser.add_argument("--gamma", type=float, default=Config.gamma)
    parser.add_argument("--tau", type=float, default=Config.tau)
    parser.add_argument("--memory-capacity", type=int, default=Config.memory_capacity)
    parser.add_argument("--batch-size", type=int, default=Config.batch_size)
    parser.add_argument("--min-memory-size", type=int, default=Config.min_memory_size)
    parser.add_argument("--initial-random-steps", type=int, default=Config.initial_random_steps)
    parser.add_argument("--ou-theta", type=float, default=Config.ou_theta)
    parser.add_argument("--ou-sigma", type=float, default=Config.ou_sigma)
    parser.add_argument("--noise-scale-start", type=float, default=Config.noise_scale_start)
    parser.add_argument("--noise-scale-end", type=float, default=Config.noise_scale_end)
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
        actor_learning_rate=args.actor_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        gamma=args.gamma,
        tau=args.tau,
        memory_capacity=args.memory_capacity,
        batch_size=args.batch_size,
        min_memory_size=args.min_memory_size,
        initial_random_steps=args.initial_random_steps,
        ou_theta=args.ou_theta,
        ou_sigma=args.ou_sigma,
        noise_scale_start=args.noise_scale_start,
        noise_scale_end=args.noise_scale_end,
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
    print("  actor: {}".format(result["actor_path"]))
    print("  critic: {}".format(result["critic_path"]))
    print("  target actor: {}".format(result["target_actor_path"]))
    print("  target critic: {}".format(result["target_critic_path"]))
    print("  summary: {}".format(result["summary_path"]))
    if result["tensorboard_dir"] is not None:
        print("  tensorboard: {}".format(result["tensorboard_dir"]))
    print("Average evaluation reward: {:.2f}".format(summary["eval_mean_reward"]))


if __name__ == "__main__":
    main()
