"""
Chapter 9: A2C (Advantage Actor-Critic) on CartPole-v1

Algorithm (equation 9.2 from the textbook):
    TD target:  y_t = r_t + γ·V(s_{t+1})·(1 - terminated_t)
    Advantage:  A_t = y_t - V(s_t)
    Actor loss: L_actor  = -E[A_t · log π(a_t|s_t)] - entropy_coef · H(π)
    Critic loss:L_critic = MSE(V(s_t), y_t)

The advantage is detached before the actor loss so actor gradients do
not flow through the critic.  V(s_{t+1}) is computed under no_grad so
the TD target is a fixed bootstrap (no double-gradient through Critic).
"""
import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import gym

# numpy 2.x removed np.bool8 – add shim for gym compatibility
if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

ENV_NAME = "CartPole-v1"
ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "a2c_cartpole"


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class Actor(nn.Module):
    """Policy network: s -> action probabilities."""

    def __init__(self, state_dim: int, hidden_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.net(x), dim=-1)


class Critic(nn.Module):
    """Value network: s -> V(s)."""

    def __init__(self, state_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Environment helpers (gym 0.26.2 compatible)
# ---------------------------------------------------------------------------

def make_env(seed: int | None = None) -> gym.Env:
    env = gym.make(ENV_NAME)
    if seed is not None and hasattr(env, "action_space"):
        env.action_space.seed(seed)
    return env


def reset_env(env: gym.Env, seed: int | None = None) -> np.ndarray:
    result = env.reset(seed=seed) if seed is not None else env.reset()
    obs = result[0] if isinstance(result, tuple) else result
    return np.asarray(obs, dtype=np.float32)


def step_env(env: gym.Env, action: int):
    """Returns (next_obs, reward, terminated, info).

    gym 0.26.2 returns a 5-tuple (obs, reward, terminated, truncated, info).
    We use only `terminated` for the bootstrap mask; `truncated` is ignored
    so that episodes cut short by the step limit don't zero out the bootstrap.
    """
    result = env.step(action)
    if len(result) == 5:
        obs, reward, terminated, _truncated, info = result
        return np.asarray(obs, dtype=np.float32), float(reward), bool(terminated), info
    obs, reward, done, info = result
    return np.asarray(obs, dtype=np.float32), float(reward), bool(done), info


# ---------------------------------------------------------------------------
# A2C Agent
# ---------------------------------------------------------------------------

class A2CAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 64,
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
    ):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.actor = Actor(state_dim, hidden_dim, action_dim)
        self.critic = Critic(state_dim, hidden_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state: np.ndarray) -> int:
        """Sample action from current policy (no grad)."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(s).squeeze(0)
        return int(torch.distributions.Categorical(probs).sample().item())

    def predict_action(self, state: np.ndarray) -> int:
        """Greedy action for evaluation (no grad)."""
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            probs = self.actor(s).squeeze(0)
        return int(probs.argmax().item())

    def collect_episode(self, env: gym.Env, max_steps: int, seed: int | None = None):
        """Roll out one full episode and return transition lists."""
        states, actions, rewards, next_states, dones = [], [], [], [], []
        state = reset_env(env, seed=seed)
        for _ in range(max_steps):
            action = self.select_action(state)
            next_state, reward, terminated, _ = step_env(env, action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(terminated)
            state = next_state
            if terminated:
                break
        return states, actions, rewards, next_states, dones

    def update(self, states, actions, rewards, next_states, dones):
        """One A2C update over a collected episode.

        Advantage A_t = r_t + γ·V(s_{t+1})·mask - V(s_t)
        Critic: minimise MSE(V(s_t), td_target)
        Actor:  minimise -E[A_t.detach() · log π(a_t|s_t)] - entropy_coef·H(π)
        """
        s      = torch.tensor(np.array(states),      dtype=torch.float32)
        a      = torch.tensor(actions,               dtype=torch.long)
        r      = torch.tensor(rewards,               dtype=torch.float32)
        s_next = torch.tensor(np.array(next_states), dtype=torch.float32)
        # mask = 0 at terminal so bootstrap is zeroed out
        mask   = torch.tensor([1.0 - float(d) for d in dones], dtype=torch.float32)

        # TD target (bootstrap masked at terminal; no_grad = fixed target)
        with torch.no_grad():
            v_next = self.critic(s_next)
        td_target = r + self.gamma * v_next * mask

        # Critic loss: MSE between prediction and fixed TD target
        v = self.critic(s)
        critic_loss = nn.functional.mse_loss(v, td_target.detach())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Advantage (detached — actor gradients must not flow through critic)
        advantage = (td_target - v.detach()).detach()

        # Actor loss: policy gradient + entropy bonus (subtracted to maximise H)
        probs     = self.actor(s)
        dist      = torch.distributions.Categorical(probs)
        log_prob  = dist.log_prob(a)
        entropy   = dist.entropy().mean()
        actor_loss = -(log_prob * advantage).mean() - self.entropy_coef * entropy
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return float(actor_loss.item()), float(critic_loss.item())


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def _moving_average(prev: float, curr: float, factor: float = 0.9) -> float:
    return prev * factor + curr * (1.0 - factor)


def train(env, agent: A2CAgent, train_episodes: int, max_steps: int, seed: int | None = None):
    rewards, ma_rewards = [], []
    ma = 0.0
    for ep in range(train_episodes):
        ep_seed = seed + ep if seed is not None else None
        states, actions, ep_rewards, next_states, dones = agent.collect_episode(
            env, max_steps, seed=ep_seed
        )
        agent.update(states, actions, ep_rewards, next_states, dones)
        total = float(sum(ep_rewards))
        rewards.append(total)
        ma = total if ep == 0 else _moving_average(ma, total)
        ma_rewards.append(ma)
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep + 1:4d}/{train_episodes}  reward={total:.1f}  ma={ma:.1f}")
    return rewards, ma_rewards


def evaluate(env, agent: A2CAgent, eval_episodes: int, max_steps: int, seed: int | None = None):
    rewards = []
    for ep in range(eval_episodes):
        ep_seed = seed + 10_000 + ep if seed is not None else None
        state = reset_env(env, seed=ep_seed)
        total = 0.0
        for _ in range(max_steps):
            action = agent.predict_action(state)
            state, reward, terminated, _ = step_env(env, action)
            total += reward
            if terminated:
                break
        rewards.append(total)
    return rewards


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def _write_svg(rewards, title: str, output_path: Path):
    """Write a minimal SVG line chart (no matplotlib dependency)."""
    n = len(rewards)
    if n == 0:
        output_path.write_text("<svg/>", encoding="utf-8")
        return
    w, h = 800, 400
    ml, mr, mt, mb = 60, 20, 30, 50
    pw, ph = w - ml - mr, h - mt - mb
    y_min, y_max = min(rewards), max(rewards)
    if math.isclose(y_min, y_max):
        y_min -= 1; y_max += 1

    def sx(i): return ml + pw * i / max(n - 1, 1)
    def sy(v): return mt + ph * (1 - (v - y_min) / (y_max - y_min))

    pts = " ".join(f"{sx(i):.1f},{sy(v):.1f}" for i, v in enumerate(rewards))
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}">',
        f'<rect width="{w}" height="{h}" fill="white"/>',
        f'<text x="{w/2}" y="22" text-anchor="middle" font-size="16">{title}</text>',
        f'<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{pts}"/>',
        f'<text x="{w/2}" y="{h - 10}" text-anchor="middle" font-size="13">Episode</text>',
        f'<text x="14" y="{h/2}" text-anchor="middle" font-size="13" '
        f'transform="rotate(-90 14 {h/2})">Reward</text>',
        "</svg>",
    ]
    output_path.write_text("\n".join(svg), encoding="utf-8")


# ---------------------------------------------------------------------------
# run_experiment (public API used by tests)
# ---------------------------------------------------------------------------

def run_experiment(
    output_dir=DEFAULT_OUTPUT_DIR,
    train_episodes: int = 500,
    eval_episodes: int = 10,
    max_steps: int = 500,
    hidden_dim: int = 64,
    actor_lr: float = 3e-4,
    critic_lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    seed: int = 42,
) -> dict:
    """Run A2C training + evaluation and save artifacts.

    Returns dict with keys:
        train_rewards, eval_rewards,
        summary_path, train_reward_path, train_ma_reward_path, eval_reward_path
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = make_env(seed=seed)
    eval_env = make_env(seed=seed)
    try:
        state_dim  = env.observation_space.shape[0]
        action_dim = env.action_space.n
        agent = A2CAgent(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
            gamma=gamma,
            entropy_coef=entropy_coef,
        )
        train_rewards, train_ma = train(env, agent, train_episodes, max_steps, seed=seed)
        eval_rewards = evaluate(eval_env, agent, eval_episodes, max_steps, seed=seed)
    finally:
        env.close()
        eval_env.close()

    train_reward_path    = output_dir / "train_rewards.svg"
    train_ma_reward_path = output_dir / "train_ma_rewards.svg"
    eval_reward_path     = output_dir / "eval_rewards.svg"
    _write_svg(train_rewards, "A2C Train Rewards",                train_reward_path)
    _write_svg(train_ma,      "A2C Train Moving Average Rewards", train_ma_reward_path)
    _write_svg(eval_rewards,  "A2C Eval Rewards",                 eval_reward_path)

    summary = {
        "train_rewards":    train_rewards,
        "train_ma_rewards": train_ma,
        "eval_rewards":     eval_rewards,
        "train_mean_reward": float(np.mean(train_rewards)) if train_rewards else 0.0,
        "eval_mean_reward":  float(np.mean(eval_rewards))  if eval_rewards  else 0.0,
        "config": {
            "train_episodes": train_episodes,
            "eval_episodes":  eval_episodes,
            "max_steps":      max_steps,
            "hidden_dim":     hidden_dim,
            "actor_lr":       actor_lr,
            "critic_lr":      critic_lr,
            "gamma":          gamma,
            "entropy_coef":   entropy_coef,
            "seed":           seed,
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    return {
        "train_rewards":       train_rewards,
        "eval_rewards":        eval_rewards,
        "summary_path":        summary_path,
        "train_reward_path":   train_reward_path,
        "train_ma_reward_path": train_ma_reward_path,
        "eval_reward_path":    eval_reward_path,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Chapter 9: A2C on CartPole-v1")
    parser.add_argument("--train-episodes", type=int,   default=500)
    parser.add_argument("--eval-episodes",  type=int,   default=10)
    parser.add_argument("--max-steps",      type=int,   default=500)
    parser.add_argument("--hidden-dim",     type=int,   default=64)
    parser.add_argument("--actor-lr",       type=float, default=3e-4)
    parser.add_argument("--critic-lr",      type=float, default=1e-3)
    parser.add_argument("--gamma",          type=float, default=0.99)
    parser.add_argument("--entropy-coef",   type=float, default=0.01)
    parser.add_argument("--seed",           type=int,   default=42)
    parser.add_argument("--output-dir",     type=str,   default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    result = run_experiment(
        output_dir=Path(args.output_dir),
        train_episodes=args.train_episodes,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        hidden_dim=args.hidden_dim,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        seed=args.seed,
    )
    summary = json.loads(result["summary_path"].read_text(encoding="utf-8"))
    print("\nSaved artifacts:")
    print(f"  train rewards:    {result['train_reward_path']}")
    print(f"  moving average:   {result['train_ma_reward_path']}")
    print(f"  eval rewards:     {result['eval_reward_path']}")
    print(f"  summary:          {result['summary_path']}")
    print(f"Eval mean reward: {summary['eval_mean_reward']:.2f}")


if __name__ == "__main__":
    main()
