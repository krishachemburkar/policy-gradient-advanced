import gymnasium as gym
import torch
import matplotlib.pyplot as plt
# from network import PolicyNetwork
import numpy as np
from torch.distributions import Normal

def compute_gae(rollout_len, buffer, last_value, gamma, gae_lambda):
    advantages = np.zeros(rollout_len)

    gae = 0.0 
    _, _, rewards, _, values, dones = buffer.get_arrays()
    for t in range(rollout_len-1, -1, -1):
        if t == rollout_len-1:
            next_value = last_value
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma*(1 - dones[t])* next_value - values[t]

        gae = delta + gamma*gae_lambda*(1-dones[t])* gae

        advantages[t] = gae

    returns = advantages+ values
    return advantages, returns


        

def collect_rollout(rollout_len, buffer, actor, critic, env, state):
    buffer.reset()

    for _ in range(rollout_len):
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        with torch.no_grad():
            mean, std = actor(state_tensor)
            dist = Normal(mean, std)

            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            value = critic(state_tensor).squeeze()
        # mean, std = actor(state_tensor)
        # dist = Normal(mean, std)

        # action = dist.sample()
        # log_prob = dist.log_prob(action).sum(dim=-1)
        raw_action = dist.sample()
        log_prob = dist.log_prob(raw_action).sum(dim=-1)

        action_np = raw_action.cpu().numpy()
        action_np = np.clip(
            action_np,
            env.action_space.low,
            env.action_space.high
        )

        next_state, reward, terminated, truncated, info = env.step(action_np)
        current_done = terminated or truncated

        # value = critic(state)

        buffer.add(
            state=state,
            reward=reward,
            action=raw_action.cpu().numpy(),
            log_prob=log_prob.item(),
            value=value.item(),
            done=current_done
        )
        state = next_state
        # done = current_done

    with torch.no_grad():
        last_state_tensor = torch.tensor(state, dtype=torch.float32)
        last_value = critic(last_state_tensor).squeeze().detach()

    return state, last_value



def normalize(x):
    if not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)

    return (x - x.mean()) / (x.std() + 1e-8)


def evaluate_policy(policy, env_name, n_eval_episodes=5):
    eval_env = gym.make(env_name)

    policy.eval()
    episode_returns = []

    with torch.no_grad():
        for _ in range(n_eval_episodes):
            state, info = eval_env.reset()
            done = False
            total_reward = 0

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32)

                mean, std = policy(state_tensor)

                action = mean.cpu().numpy()
                action = np.clip(
                    action,
                    eval_env.action_space.low,
                    eval_env.action_space.high
                )

                next_state, reward, terminated, truncated, info = eval_env.step(action)

                total_reward += reward
                state = next_state
                done = terminated or truncated

            episode_returns.append(total_reward)

    eval_env.close()
    policy.train()

    return sum(episode_returns) / len(episode_returns)

def log_training(
    log_file,
    episode,
    episode_return,
    train_avg,
    avg_eval_return,
    actor_loss=None,
    critic_loss=None
):
    log_line = (
        f"Episode: {episode}, "
        f"Train Return: {episode_return:.2f}, "
        f"Train Avg Return: {train_avg:.2f}, "
        f"Eval Avg Return: {avg_eval_return}, "
        f"Actor Loss: {actor_loss}, "
        f"Critic Loss: {critic_loss}\n"
    )

    print(log_line.strip())
    log_file.write(log_line)
    log_file.flush()


# def plot_returns(train_avg_returns, eval_episodes, eval_rewards, n_episode):
#     plt.figure()

#     plt.plot(
#         range(1, n_episode + 1),
#         train_avg_returns,
#         label="Train Avg Return"
#     )

#     plt.plot(
#         eval_episodes,
#         eval_rewards,
#         label="Eval Avg Return"
#     )

#     plt.xlabel("Episode")
#     plt.ylabel("Average Return")
#     plt.title("Train vs Eval Return")
#     plt.legend()
#     plt.grid(True)

#     plt.savefig("train_eval_returns.png")
#     plt.show()

def plot_returns(train_steps, train_avg_returns, eval_steps, eval_rewards):
    plt.figure()

    if len(train_avg_returns) > 0:
        plt.plot(
            train_steps,
            train_avg_returns,
            label="Train Avg Return"
        )

    if len(eval_rewards) > 0:
        plt.plot(
            eval_steps,
            eval_rewards,
            label="Eval Avg Return"
        )

    plt.xlabel("Environment Steps")
    plt.ylabel("Average Return")
    plt.title("Train vs Eval Return")
    plt.legend()
    plt.grid(True)

    plt.savefig("train_eval_returns.png")
    plt.show()