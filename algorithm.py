import numpy as np
import paddle
from paddle import nn
from paddle.distribution import Categorical
from parl import Algorithm
import paddle.nn.functional as F
from parl.utils import LinearDecayScheduler
from parl.utils import summary


device = paddle.CUDAPlace(0)


class PPO(Algorithm):
    def __init__(self, model, lr, betas, gamma, K_epochs, eps_clip=0.1):
        self.model = model
        self.lr = lr
        self.eps = 1e-5
        self.betas = betas
        self.max_grad_norm = 0.5
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.policy = model.to(device)
        self.mse_loss = paddle.nn.MSELoss(reduction='mean')
        clip = nn.ClipGradByValue(self.max_grad_norm)
        self.optimizer = paddle.optimizer.Adam(learning_rate=lr, parameters=self.policy.get_params(), epsilon=self.eps,
                                               grad_clip=clip)
        self.lr_scheduler = LinearDecayScheduler(lr, int(1000))

    def sample(self, obs):
        value = self.model.value(obs)
        logits = self.model.policy(obs)
        dist = Categorical(logits=logits)
        action = dist.sample([1])
        act_dim = logits.shape[-1]
        actions_onehot = F.one_hot(action, act_dim)
        action_log_probs = paddle.sum(F.log_softmax(logits) * actions_onehot, axis=-1)
        action_entropy = dist.entropy()

        return value, action, action_log_probs, action_entropy

    def predict(self, obs):
        logits = self.model.policy(obs)
        probs = F.softmax(logits)
        action = paddle.argmax(probs, 1)

        return action

    def value(self, obs):
        return self.model.value(obs)

    def learn(self,
              batch_obs,
              batch_action,
              batch_value,
              batch_return,
              batch_logprob,
              batch_adv):
        self.lr = self.lr_scheduler.step(step_num=1)
        values = self.model.value(batch_obs)
        logits = self.model.policy(batch_obs)

        dist = Categorical(logits=logits)
        # actions = dist.sample([1])

        batch_action = paddle.to_tensor(batch_action, dtype='int64')
        actions_onehot = F.one_hot(batch_action, 3)

        # print("logits", F.log_softmax(logits).shape)

        action_log_probs = paddle.sum(F.log_softmax(logits) * actions_onehot, axis=-1)
        dist_entropy = dist.entropy()

        entropy_loss = dist_entropy.mean()

        batch_adv = batch_adv

        batch_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)

        ratio = paddle.exp(action_log_probs - batch_logprob)
        surr1 = ratio * batch_adv
        surr2 = paddle.clip(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * batch_adv

        action_loss = -paddle.minimum(surr1, surr2).mean()

        values = values.reshape([-1])

        value_pred_clipped = batch_value + paddle.clip(
            values - batch_value, -0.1, 0.1)
        value_losses = (values - batch_return).pow(2)
        value_losses_clipped = (value_pred_clipped - batch_return).pow(2)
        value_loss = 0.5 * paddle.maximum(value_losses, value_losses_clipped).mean()

        loss = 0.1 * value_loss + action_loss - 0.01 * entropy_loss

        self.optimizer.set_lr(self.lr)
        self.optimizer.clear_grad()
        loss.mean().backward()
        self.optimizer.step()

        return value_loss.item(), action_loss.item(), entropy_loss.item()
