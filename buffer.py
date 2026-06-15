import numpy as np


class RolloutBuffer:
    def __init__(self):
        self.reset()

    def add(
        self,
        state,
        action,
        reward,
        log_prob,
        value,
        done
    ):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def reset(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def __len__(self):
        return len(self.rewards)

    def get_arrays(self):
        """
        Returns everything as numpy arrays.
        Useful for GAE computation.
        """

        return (
            np.array(self.states, dtype=np.float32),
            np.array(self.actions, dtype=np.float32),
            np.array(self.rewards, dtype=np.float32),
            np.array(self.log_probs, dtype=np.float32),
            np.array(self.values, dtype=np.float32),
            np.array(self.dones, dtype=np.float32)
        )