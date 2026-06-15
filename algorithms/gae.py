import torch
from torch.distributions import Normal


def train_gae_actor_critic(
    actor,
    critic,
    actor_optimizer,
    critic_optimizer,
    critic_criterion,
    states,
    actions,
    advantages,
    returns
):
    mean, std = actor(states)
    dist = Normal(mean, std)

    log_probs = dist.log_prob(actions).sum(dim=-1)

    print("mean nan:", torch.isnan(mean).any())
    print("std nan:", torch.isnan(std).any())
    print("actions nan:", torch.isnan(actions).any())
    print("advantages nan:", torch.isnan(advantages).any())
    print("returns nan:", torch.isnan(returns).any())

    actor_loss = -(log_probs * advantages).mean()

    predicted_values = critic(states).squeeze()
    critic_loss = critic_criterion(predicted_values, returns)

    actor_optimizer.zero_grad()
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5)
    actor_optimizer.step()

    critic_optimizer.zero_grad()
    critic_loss.backward()
    torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5)
    critic_optimizer.step()

    return actor_loss.item(), critic_loss.item()