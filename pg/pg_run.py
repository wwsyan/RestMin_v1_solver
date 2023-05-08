import numpy as np
import torch as th

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import obs_as_tensor

from env import RestMinEnv_v1
from pg import PolicyGradient
from callback_da import DataAugmentCallback

def mask_fn(env):
    return env.get_action_mask()

if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=1, use_cluster_reward=True)
    env = ActionMasker(env, mask_fn)
    
    model = PolicyGradient.load("pg_75", env=env)
    
    # policy_kwargs = dict(activation_fn=th.nn.Tanh,
    #                       net_arch=dict(pi=[64, 64], vf=[64, 64]))
    # model = PolicyGradient(policy="MlpPolicy", 
    #                     env=env, 
    #                     learning_rate=3e-4,
    #                     n_steps=2048, 
    #                     batch_size=512,
    #                     n_epochs=8,
    #                     gamma=0.99,
    #                     gae_lambda=0.95,
    #                     ent_coef=0.01,
    #                     vf_coef=0.5,
    #                     max_grad_norm=0.5,
    #                     target_kl=None,
    #                     tensorboard_log=None,
    #                     policy_kwargs=policy_kwargs,
    #                     seed=16, 
    #                     verbose=2,
    #                     )
    
    DACallback = DataAugmentCallback(env=env, model=model, use_DA=True)
    
    model.learn(50e4, callback=DACallback)
    model.save("pg")
    del model
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    
    
    
    
    
    
    

