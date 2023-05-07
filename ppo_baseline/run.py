import numpy as np
import torch as th

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from env import RestMinEnv_v1

def mask_fn(env):
    return env.get_action_mask()

if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=0)
    env = ActionMasker(env, mask_fn)
    
    policy_kwargs = dict(activation_fn=th.nn.Tanh,
                          net_arch=dict(pi=[64, 64], vf=[64, 64]))
    
    try:
        model = MaskablePPO.load(path="ppo", env=env)
    except:
        model = MaskablePPO(policy="MlpPolicy", 
                            env=env, 
                            learning_rate=3e-4,
                            n_steps=2048, 
                            batch_size=512,
                            n_epochs=10,
                            gamma=0.99,
                            gae_lambda=0.95,
                            clip_range=0.2,
                            clip_range_vf=None,
                            normalize_advantage=False,
                            ent_coef=0.01,
                            vf_coef=0.5,
                            max_grad_norm=0.5,
                            target_kl=0.5,
                            tensorboard_log="tf-logs",
                            policy_kwargs=policy_kwargs,
                            seed=16, 
                            verbose=2,
                            )
    
    model.learn(100e4)
    model.save("ppo")
    del model
    
    
    
    
    
    
    
    
    
    
    
    
    
    


