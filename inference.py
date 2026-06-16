import os

import gymnasium as gym
import torch
from gymnasium.wrappers import RecordVideo

from args import args
from network import PolicyNetwork


def run_inference():
    video_dir = f"videos/{args.model_name}"

    if args.save_video:
        os.makedirs(video_dir, exist_ok=True)

        env = gym.make(args.env, render_mode="rgb_array")

        env = RecordVideo(
            env,
            video_folder=video_dir,
            episode_trigger=lambda episode_id: True,
            name_prefix=args.model_name
        )
    else:
        env = gym.make(args.env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor = PolicyNetwork(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=args.policy_hidden
    )

    model_path = f"{args.model_name}_best_actor.pth"

    actor.load_state_dict(torch.load(model_path, map_location="cpu"))
    actor.eval()

    num_episodes = 5

    for ep in range(num_episodes):
        state, info = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                mean, std = actor(state_tensor)

                # deterministic inference: use mean action
                raw_action = mean
                action = torch.tanh(raw_action)
            print("mean:", mean.cpu().numpy(), "action:", action.cpu().numpy())

            # with torch.no_grad():
            #     mean, std = actor(state_tensor)
            #     dist = torch.distributions.Normal(mean, std)
            #     raw_action = dist.sample()
            #     action = torch.tanh(raw_action)

            action_np = action.cpu().numpy()

            next_state, reward, terminated, truncated, info = env.step(action_np)

            done = terminated or truncated
            total_reward += reward
            state = next_state

        print(f"Episode {ep + 1}: Return = {total_reward:.2f}")

    env.close()

    if args.save_video:
        print(f"Videos saved in: {video_dir}")


if __name__ == "__main__":
    run_inference()