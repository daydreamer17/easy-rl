import gym
import numpy as np


ENV_NAME = "MountainCar-v0"


class SimpleAgent:
    def __init__(self, env):
        pass

    def decide(self, observation):  # 决策
        position, velocity = observation
        lb = min(
            -0.09 * (position + 0.25) ** 2 + 0.03,
            0.3 * (position + 0.9) ** 4 - 0.008,
        )
        ub = -0.07 * (position + 0.38) ** 2 + 0.07
        if lb < velocity < ub:
            action = 2
        else:
            action = 0
        return action  # 返回动作

    def learn(self, *args):  # 学习
        pass


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
        return observation
    return reset_result


def step_env(env, action):
    step_result = env.step(action)
    if len(step_result) == 5:
        next_observation, reward, terminated, truncated, info = step_result
        done = terminated or truncated
        return next_observation, reward, done, info
    return step_result


def play(env, agent, render=False, train=False, seed=None):
    episode_reward = 0.0  # 记录回合总奖励，初始化为0
    observation = reset_env(env, seed=seed)  # 重置游戏环境，开始新回合
    while True:  # 不断循环，直到回合结束
        if render:  # 判断是否显示
            env.render()  # 显示图形界面，图形界面可以用 env.close() 语句关闭
        action = agent.decide(observation)
        next_observation, reward, done, _ = step_env(env, action)  # 执行动作
        episode_reward += reward  # 收集回合奖励
        if train:  # 判断是否训练智能体
            agent.learn(observation, action, reward, done)  # 学习
        if done:  # 回合结束，跳出循环
            break
        observation = next_observation
    return episode_reward  # 返回回合总奖励


def main():
    env = create_env(render_mode="human")
    agent = SimpleAgent(env)
    print("观测空间 = {}".format(env.observation_space))
    print("动作空间 = {}".format(env.action_space))
    print(
        "观测范围 = {} ~ {}".format(
            env.observation_space.low,
            env.observation_space.high,
        )
    )
    print("动作数 = {}".format(env.action_space.n))

    episode_reward = play(env, agent, render=True, seed=3)
    print("回合奖励 = {}".format(episode_reward))

    episode_rewards = [play(env, agent, seed=3 + i) for i in range(100)]
    print("平均回合奖励 = {}".format(np.mean(episode_rewards)))

    env.close()


if __name__ == "__main__":
    main()
