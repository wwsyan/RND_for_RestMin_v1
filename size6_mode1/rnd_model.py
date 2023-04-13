import random
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Union, Tuple, Dict, List, Optional

standard_RND_net_config = dict(
    hidden_size_list=[16, 8],
    learning_rate=1e-3,
    batch_size=64,
    epoch=20,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1,
    device="gpu"
)

class FCEncoder(nn.Module):
    def __init__(
            self,
            obs_shape: int,
            hidden_size_list,
            activation: Optional[nn.Module] = nn.ReLU(),
    ) -> None:
        super(FCEncoder, self).__init__()
        self.obs_shape = obs_shape
        self.act = activation
        self.init = nn.Linear(obs_shape, hidden_size_list[0])

        layers = []
        for i in range(len(hidden_size_list) - 1):
            layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i + 1]))
            layers.append(self.act)
        self.main = nn.Sequential(*layers)

    def forward(self, x: th.Tensor) -> th.Tensor:
        x = self.act(self.init(x))
        x = self.main(x)
        return x
    
class RndNetwork(nn.Module):
    def __init__(self, obs_shape: Union[int, list], hidden_size_list: list) -> None:
        super(RndNetwork, self).__init__()
        self.target = FCEncoder(obs_shape, hidden_size_list)
        self.predictor = FCEncoder(obs_shape, hidden_size_list)

        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, obs: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        predict_feature = self.predictor(obs)
        with th.no_grad():
            target_feature = self.target(obs)
        return predict_feature, target_feature

class RunningMeanStd(object):
    def __init__(self, epsilon=1e-4, shape=(), device=th.device('cpu')):
        self._epsilon = epsilon
        self._shape = shape
        self._device = device
        self.reset()

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        new_count = batch_count + self._count
        mean_delta = batch_mean - self._mean
        new_mean = self._mean + mean_delta * batch_count / new_count
        # this method for calculating new variable might be numerically unstable
        m_a = self._var * self._count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(mean_delta) * self._count * batch_count / new_count
        new_var = m2 / new_count
        self._mean = new_mean
        self._var = new_var
        self._count = new_count

    def reset(self):
        if len(self._shape) > 0:
            self._mean = np.zeros(self._shape, 'float32')
            self._var = np.ones(self._shape, 'float32')
        else:
            self._mean, self._var = 0., 1.
        self._count = self._epsilon

    @property
    def mean(self) -> np.ndarray:
        if np.isscalar(self._mean):
            return self._mean
        else:
            return th.FloatTensor(self._mean).to(self._device)

    @property
    def std(self) -> np.ndarray:
        std = np.sqrt(self._var + 1e-8)
        if np.isscalar(std):
            return std
        else:
            return th.FloatTensor(std).to(self._device)
        
class RndRewardModel():

    def __init__(self, config) -> None:  # noqa
        self.cfg = config
        self.device = th.device("cuda:0") if config["device"] == "gpu" and th.cuda.is_available() else th.device("cpu")
        
        self.reward_model = RndNetwork(
            obs_shape=config["observation_shape"], hidden_size_list=config["hidden_size_list"]
        ).to(self.device)

        self.optim = optim.Adam(self.reward_model.predictor.parameters(), config["learning_rate"])

        if self.cfg["obs_norm"]:
            self._running_mean_std_rnd_obs = RunningMeanStd(epsilon=1e-4, device=self.device)

    def train(self, data: np.ndarray) -> None:
        batch_num, obs_dim = data.shape
        
        for epoch in range(self.cfg["epoch"]):
            if batch_num > self.cfg["batch_size"]:
                rand_index = np.random.randint(0, batch_num, size=self.cfg["batch_size"])
                train_data = data[rand_index]
            else:
                train_data = data
            
            train_data: th.Tensor = th.FloatTensor(train_data).to(self.device)
            if self.cfg["obs_norm"]:
                # Note: observation normalization: transform obs to mean 0, std 1
                self._running_mean_std_rnd_obs.update(train_data.cpu().numpy())
                train_data = (train_data - self._running_mean_std_rnd_obs.mean) / self._running_mean_std_rnd_obs.std
                train_data = th.clamp(
                    train_data, min=self.cfg["obs_norm_clamp_min"], max=self.cfg["obs_norm_clamp_max"]
                )

            predict_feature, target_feature = self.reward_model(train_data)
            loss = F.mse_loss(predict_feature, target_feature.detach())
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

    def estimate(self, data: np.ndarray) -> np.ndarray:
        """
        estimate the rnd intrinsic reward
        """

        obs = th.FloatTensor(data).to(self.device)
        if self.cfg["obs_norm"]:
            # Note: observation normalization: transform obs to mean 0, std 1
            obs = (obs - self._running_mean_std_rnd_obs.mean) / self._running_mean_std_rnd_obs.std
            obs = th.clamp(obs, min=self.cfg["obs_norm_clamp_min"], max=self.cfg["obs_norm_clamp_max"])

        with th.no_grad():
            predict_feature, target_feature = self.reward_model(obs)
            mse = F.mse_loss(predict_feature, target_feature, reduction='none').mean(dim=1) # mean by batch

            # Note: according to the min-max normalization, transform rnd reward to [0,1]
            rnd_reward = (mse - mse.min()) / (mse.max() - mse.min() + 1e-11)
            
            rnd_reward = rnd_reward.cpu().numpy() * self.cfg["reward_mse_ratio"]
        
        return rnd_reward


if __name__ == "__main__":
    batch_num, env_num, obs_dim = 100, 1, 3
    data = np.random.rand(batch_num, env_num, obs_dim)
    data_flat = data.reshape(-1, obs_dim)
    print(data,"\n")
    
    # 测试1
    MeanStd = RunningMeanStd()
    MeanStd.update(data_flat)
    if False:
        print(data_flat)
        print(MeanStd.mean)
        print(MeanStd.std)
    
    # 测试2
    config = standard_RND_net_config
    config["observation_shape"] = obs_dim
    rnd_model = RndRewardModel(config)
    rnd_model.train(data_flat)
    rnd_reward = rnd_model.estimate(data_flat).reshape(batch_num, env_num)
    if True:
        print(rnd_reward)
            

















