import numpy as np
import torch as th

from ppo_mcts import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from callback_da import DataAugmentCallback
from env import RestMinEnv_v1


def mask_fn(env):
    return env.get_action_mask()

if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=0)
    env = ActionMasker(env, mask_fn)
    # 需要为MCTS单独创建一个环境实例, 因为env被wrapper处理后, 只能调用接口reset和step 
    env_for_mcts = RestMinEnv_v1(size=6, mode=0)
    
    model = MaskablePPO.load(path="ppo", env=env, target_kl=None)
    model.set_mcts(use_mcts=True, env_for_mcts=env_for_mcts, c_puct=5, n_playout=50, is_train=True)
    
    DACallback = DataAugmentCallback(env=env, model=model, use_DA=True)
                                     
    model.learn(3e4, callback=DACallback)
    model.save("ppo")
    del model
    

