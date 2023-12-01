import numpy as np
import scipy

def _accumulate_sum(arr, param):
    # res = np.zeros(shape=(arr.shape[0], 1), dtype=np.float32)
    # for i, a in enumerate(arr):
    #     for j, e in enumerate(arr[i:]):
    #         res[i, 0] += (param ** j) * e
    # return res
    return scipy.signal.lfilter([1], [1, float(-param)], arr[::-1], axis=0)[::-1]


class Buffer:

    def __init__(self, max_steps, state_space):
        self.size = max_steps
        self.n_states = state_space
        self.states, self.actions, self.log_probs, self.rewards, self.values, self.returns, self.advantages = \
            None, None, None, None, None, None, None

        self._pointer, self._trajectory_start_index = None, None

        self._build()

    def add_data(self, state, action, log_prob, reward, value):
        self.states[self._pointer] = state
        self.actions[self._pointer] = action
        self.log_probs[self._pointer] = log_prob
        self.rewards[self._pointer] = reward
        self.values[self._pointer] = value

        self._pointer += 1

    def _build(self):
        self.states = np.ndarray(shape=(self.size, self.n_states), dtype=np.float32)
        self.actions = np.ndarray(shape=(self.size,), dtype=np.uint8)
        self.log_probs = np.ndarray(shape=(self.size,), dtype=np.float32)
        self.rewards = np.ndarray(shape=(self.size,), dtype=np.float32)
        self.values = np.ndarray(shape=(self.size,), dtype=np.float32)
        self.returns = np.ndarray(shape=(self.size,), dtype=np.float32)
        self.advantages = np.ndarray(shape=(self.size,), dtype=np.float32)
        self._pointer = 0
        self._trajectory_start_index = 0

    def finish_trajectory(self, gamma, lmbd, last_value):
        path_slice = slice(self._trajectory_start_index, self._pointer)
        rewards = np.append(self.rewards[path_slice], last_value)
        values = np.append(self.values[path_slice], last_value)

        deltas = rewards[:-1] + gamma * values[1:] - values[:-1]

        self.advantages[path_slice] = _accumulate_sum(
            deltas, gamma * lmbd
        )
        self.returns[path_slice] = _accumulate_sum(
            rewards, gamma
        )[:-1]

        self._trajectory_start_index = self._pointer

    def reset(self):
        self._build()

    def get(self):
        adv = (self.advantages - np.mean(self.advantages)) / np.std(self.advantages)
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.values,
            self.returns,
            adv
        )
