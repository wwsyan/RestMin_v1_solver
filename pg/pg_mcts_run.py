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
    env = RestMinEnv_v1()
    env = ActionMasker(env, mask_fn)
    
    # 需要为MCTS单独创建一个环境引用, 因为env被wrapper处理后, 只能调用接口reset和step 
    env_for_mcts = RestMinEnv_v1()
    
    model = PolicyGradient.load(path="pg", env=env, n_steps=2048, n_epochs=8)
    model.tensorboard_log = None
    model.set_mcts_run(env=env_for_mcts)
    
    DACallback = DataAugmentCallback(env=env,
                                      model=model, 
                                      rollout_buffer=model.rollout_buffer,
                                      drop_episode=True,
                                      use_DA=True,
                                      print_buffer_data=False)
    
    model.learn(3e4, callback=DACallback)
    model.save("pg")
    del model

