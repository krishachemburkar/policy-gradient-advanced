import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import torch
import numpy as np

from torch.distributions import Normal
from torch.optim import Adam

from args import args
from network import PolicyNetwork, Baseline

from algorithms.gae import train_gae_actor_critic

from utils import (
    collect_rollout,
    compute_gae,
    normalize,
    evaluate_policy,
    log_training,
    plot_returns
)

from buffer import RolloutBuffer


env = gym.make(
    args.env,
    # render_mode="rgb_array"
)

# env = RecordVideo(
#     env,
#     video_folder="videos",
#     episode_trigger=lambda episode_id: episode_id % 100 == 0
# )

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

actor = PolicyNetwork(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_size=args.policy_hidden
)

critic = Baseline(
    state_dim=state_dim,
    hidden_size=args.nn_baseline_hidden
)

actor_optimizer = Adam(actor.parameters(), lr=args.actor_lr)
critic_optimizer = Adam(critic.parameters(), lr=args.critic_lr)

critic_criterion = torch.nn.MSELoss()

buffer = RolloutBuffer()

train_returns = []
train_avg_returns = []
train_steps = []

eval_rewards = []
eval_steps = []

actor_losses = []
critic_losses = []

best_eval_return = -float("inf")

log_file = open(args.log_file, "w")

state, info = env.reset()

total_steps = 0
episode_count = 0
episode_return = 0


while total_steps < args.max_steps:

    state, last_value = collect_rollout(
        rollout_len=args.rollout_steps,
        buffer=buffer,
        actor=actor,
        critic=critic,
        env=env,
        state=state
    )

    total_steps += args.rollout_steps

    states, actions, rewards, old_log_probs, values, dones = buffer.get_arrays()

    for reward, done in zip(rewards, dones):
        episode_return += reward

        if done:
            train_returns.append(episode_return)
            train_avg_returns.append(np.mean(train_returns[-50:]))
            train_steps.append(total_steps)

            episode_count += 1
            episode_return = 0

    advantages, returns = compute_gae(
        rollout_len=args.rollout_steps,
        buffer=buffer,
        last_value=last_value.item() if torch.is_tensor(last_value) else last_value,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda
    )

    states = torch.tensor(states, dtype=torch.float32)
    actions = torch.tensor(actions, dtype=torch.float32)
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)

    advantages = normalize(advantages)

    actor_loss_value, critic_loss_value = train_gae_actor_critic(
        actor=actor,
        critic=critic,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        critic_criterion=critic_criterion,
        states=states,
        actions=actions,
        advantages=advantages,
        returns=returns
    )
    actor_losses.append(actor_loss_value)
    critic_losses.append(critic_loss_value)

    avg_actor_loss = np.mean(actor_losses[-100:]) if len(actor_losses) > 0 else 0
    avg_critic_loss = np.mean(critic_losses[-100:]) if len(critic_losses) > 0 else 0

    avg_eval_return = None

    if total_steps % args.eval_freq < args.rollout_steps:
        avg_eval_return = evaluate_policy(
            actor,
            args.env,
            n_eval_episodes=5
        )

        eval_rewards.append(avg_eval_return)
        eval_steps.append(total_steps)

        if avg_eval_return > best_eval_return:
            best_eval_return = avg_eval_return

            torch.save(
                actor.state_dict(),
                f"{args.model_name}_best_actor.pth"
            )

            torch.save(
                critic.state_dict(),
                f"{args.model_name}_best_critic.pth"
            )

            print(f"New best eval return: {best_eval_return:.2f}")

    avg_actor_loss = np.mean(actor_losses[-100:])
    avg_critic_loss = np.mean(critic_losses[-100:])

    latest_return = train_returns[-1] if len(train_returns) > 0 else 0
    latest_avg = train_avg_returns[-1] if len(train_avg_returns) > 0 else 0

    log_training(
        log_file=log_file,
        episode=episode_count,
        episode_return=latest_return,
        train_avg=latest_avg,
        avg_eval_return=avg_eval_return,
        actor_loss=avg_actor_loss,
        critic_loss=avg_critic_loss
    )


env.close()
log_file.close()

torch.save(
    actor.state_dict(),
    f"{args.model_name}_final_actor.pth"
)

torch.save(
    critic.state_dict(),
    f"{args.model_name}_final_critic.pth"
)

plot_returns(
    train_steps=train_steps,
    train_avg_returns=train_avg_returns,
    eval_steps=eval_steps,
    eval_rewards=eval_rewards
)