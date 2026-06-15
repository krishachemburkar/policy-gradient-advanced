import argparse

parser = argparse.ArgumentParser()

# Environment
parser.add_argument(
    "--env",
    type=str,
    default="MountainCarContinuous-v0"
)

# Training
parser.add_argument(
    "--max_steps",
    type=int,
    default=200000
)

parser.add_argument(
    "--rollout_steps",
    type=int,
    default=128
)

parser.add_argument(
    "--eval_freq",
    type=int,
    default=5000
)

# Discounting
parser.add_argument(
    "--gamma",
    type=float,
    default=0.99
)

parser.add_argument(
    "--gae_lambda",
    type=float,
    default=0.95
)

# Learning rates
parser.add_argument(
    "--actor_lr",
    type=float,
    default=3e-4
)

parser.add_argument(
    "--critic_lr",
    type=float,
    default=1e-3
)

# Network
parser.add_argument(
    "--policy_hidden",
    type=int,
    default=64
)

parser.add_argument(
    "--nn_baseline_hidden",
    type=int,
    default=64
)

# Logging
parser.add_argument(
    "--model_name",
    type=str,
    default="gae_ac"
)

parser.add_argument(
    "--log_file",
    type=str,
    default="training_log.txt"
)

# Reproducibility
parser.add_argument(
    "--seed",
    type=int,
    default=42
)

args = parser.parse_args()