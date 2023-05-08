import numpy as np
import torch as th

from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.save_util import load_from_zip_file

from env import RestMinEnv_v1
from pg import PolicyGradient


config = dict(
    learning_rate=3e-4,
    n_steps=2048,
    n_epochs=4,
    gamma=0.99,
    gae_lambda=0.95,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=1
)

def mask_fn(env):
    return env.get_action_mask()

if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=1, use_cluster_reward=True)
    env = ActionMasker(env, mask_fn)
    model = PolicyGradient.load(path="pg_75", env=env)
    
    print("learning_rate:", model.learning_rate)
    print("n_steps:", model.n_steps)
    print("n_epochs", model.n_epochs)
    print("gamma:", model.gamma)
    print("gae_lambda:", model.gae_lambda)
    print("ent_coef:", model.ent_coef)
    print("vf_coef:", model.vf_coef)
    print("max_grad_norm:", model.max_grad_norm)
    print("use_sde:", model.use_sde)
    print("sde_sample_freq:", model.sde_sample_freq)
    print("policy_kwargs:", model.policy_kwargs)
    print("target_kl:", model.target_kl)
    print("seed:", model.seed)
    
    
    # model.policy.set_training_mode(False)
    # for i in range(1):
    #     obs = env.reset()
    #     while True:
    #         print(env.state["obs"], "\n")
    #         actions, _ = model.predict(obs_as_tensor(obs, "cpu"), action_masks=env.get_action_mask())
    #         _, values, __ = model.policy(obs_as_tensor(np.expand_dims(obs, axis=0), "cuda"))
    #         print(values, "\n")
    #         obs, reward, done, info = env.step(actions)
    #         if done:
    #             break
    

