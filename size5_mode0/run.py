import numpy as np
import torch as th

from mppo_rnd import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from callback_da import DataAugmentCallback
from env import RestMinEnv_v1

standard_RND_net_config = dict(
    hidden_size_list=[16, 8],
    learning_rate=1e-3,
    batch_size=128,
    epoch=10,
    obs_norm=True,
    obs_norm_clamp_min=-1,
    obs_norm_clamp_max=1,
    reward_mse_ratio=1,
    device="gpu"
)

def mask_fn(env):
    return env.get_action_mask()

if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=0)
    env = ActionMasker(env, mask_fn)
    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                          net_arch=dict(pi=[64, 64], vf=[64, 64]))
    model = MaskablePPO(policy="MlpPolicy", 
                        env=env, 
                        learning_rate=3e-4,
                        n_steps=1024, 
                        batch_size=64,
                        gamma=0.99,
                        gae_lambda=0.95,
                        clip_range=0.2,
                        normalize_advantage=False,
                        ent_coef=0.01,
                        vf_coef=0.5,
                        max_grad_norm=0.5,
                        target_kl=None,
                        tensorboard_log="log_size6_mode0",
                        policy_kwargs=policy_kwargs,
                        seed=16, 
                        verbose=2,
                        rnd_config=standard_RND_net_config
                        )
    
    DACallback = DataAugmentCallback(env=env,
                                     model=model, 
                                     rollout_buffer=model.rollout_buffer,
                                     drop_episode=False,
                                     use_DA=True,
                                     print_buffer_data=False
                                     )
    
    model.learn(50e4, callback=DACallback)
    model.save("ppo")
    del model
    
    
    
    
    
    
    
    
    
    
    
    
    
    


