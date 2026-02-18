import minigrid
import gymnasium as gym

# U-shaped maze — directly mirrors the paper's topology
env = gym.make("MiniGrid-FourRooms-v0", render_mode="human")

obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # random action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()