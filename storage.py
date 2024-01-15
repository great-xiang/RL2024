import numpy as np

class ReplayMemory:
    def __init__(self, ENV_NUM=5, STEP_NUMS=1000):

        self.cur_step = 0
        self.step_nums = STEP_NUMS
        self.obs_space = (3, 84, 84)
        self.act_space = (1,)

        self.obs = np.zeros((STEP_NUMS, ENV_NUM) + self.obs_space, dtype='float32')
        self.actions = np.zeros((STEP_NUMS, ENV_NUM) + self.act_space, dtype='float32')
        self.logprobs = np.zeros((STEP_NUMS, ENV_NUM), dtype='float32')
        self.rewards = np.zeros((STEP_NUMS, ENV_NUM), dtype='float32')
        self.dones = np.zeros((STEP_NUMS, ENV_NUM), dtype='float32')
        self.values = np.zeros((STEP_NUMS, ENV_NUM), dtype='float32')

    def append(self, obs, action, logprob, reward, done, value):
        self.obs[self.cur_step] = obs
        self.actions[self.cur_step] = action
        self.logprobs[self.cur_step] = logprob
        self.rewards[self.cur_step] = reward
        self.dones[self.cur_step] = done
        self.values[self.cur_step] = value

        self.cur_step = (self.cur_step + 1) % self.step_nums

    def compute_returns(self, value, done, gamma=0.99, gae_lambda=0.95):
        # gamma: discounting factor
        # gae_lambda: Lambda parameter for calculating N-step advantage
        advantages = np.zeros_like(self.rewards)
        lastgaelam = 0
        for t in reversed(range(self.step_nums)):
            if t == self.step_nums - 1:
                nextnonterminal = 1.0 - done
                nextvalues = value.reshape(1, -1)
            else:
                nextnonterminal = 1.0 - self.dones[t + 1]
                nextvalues = self.values[t + 1]
            delta = self.rewards[t] + gamma * nextvalues * nextnonterminal - self.values[t]
            advantages[t] = lastgaelam = delta + gamma * gae_lambda * nextnonterminal * lastgaelam
            # print("advantages[t]", advantages[t])
            # print("lastgaelam", lastgaelam)
        returns = advantages + self.values
        self.returns = returns
        self.advantages = advantages
        return advantages, returns

    def sample_batch(self, idx):
        # flatten rollout
        b_obs = self.obs.reshape((-1,) + self.obs_space)
        b_actions = self.actions.reshape((-1,) + self.act_space)
        b_logprobs = self.logprobs.reshape(-1)
        b_advantages = self.advantages.reshape(-1)
        b_returns = self.returns.reshape(-1)
        b_values = self.values.reshape(-1)

        return b_obs[idx], b_actions[idx], b_logprobs[idx], b_advantages[idx], b_returns[idx], b_values[idx]

    def __call__(self, *args, **kwargs):
        print("is_done", self.dones)
