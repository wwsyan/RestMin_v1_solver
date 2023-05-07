import numpy as np
import torch as th
from typing import Any, Dict, Generator, List, Optional, Union
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor

class DataAugmentCallback(BaseCallback):
    def __init__(self, 
                 verbose=0,
                 env=None,
                 model=None,
                 drop_episode=False, # Whether to drop undone episode
                 use_DA=True, # Whether to use data augment
                 use_returns_as_adv=False, # Whether to replace advantages with returns 
                 print_buffer_data=False # Whether to print buffer data for debugging
    ):
        super().__init__(verbose)
        self.env = env
        self.model = model
        self.rollout_buffer = model.rollout_buffer
        self.drop_episode = drop_episode
        self.use_DA = use_DA
        self.print_buffer_data = print_buffer_data


    def _on_step(self) -> bool:
        # This is an abstractmethod, you need to redefine
        return True
    
    
    def _print_buffer_data(self) -> None:
        print("Rollout_buffer data:")
        for key, value in self.rollout_buffer.__dict__.items():
            print(key, "=")
            print(value, "\n")
    
    
    def _data_augment(self) -> None:
        """
        Data augment by rotating and flipping, which generates extra 7 batch data:
            rotate 90, rotate 90 + fliplr
            rotate 180, rotate 180 + fliplr
            rotate 270, rotate 270 + fliplr
            rotate 360, rotate 360 + fliplr
                
        :method direction_value_trans: 
        :method grid_trans: 
        
        """
        def direction_trans(direction, trans_type, rot_times=0) -> int:
            # 方向变换
            UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
            def step(direction, trans_type) -> int:
                if trans_type == "rot90":
                    if direction == UP: return LEFT
                    if direction == DOWN:  return RIGHT 
                    if direction == LEFT:  return DOWN 
                    if direction == RIGHT:  return UP
                if trans_type == 'fliplr':
                    if direction == UP: return UP
                    if direction == DOWN: return DOWN
                    if direction == LEFT: return RIGHT
                    if direction == RIGHT: return LEFT
            
            if trans_type == "rot90":
                direction_new = direction
                for i in range(rot_times):
                    direction_new = step(direction_new, "rot90")
            if trans_type == "fliplr":
                direction_new = step(direction, "fliplr")
            
            return direction_new
               
        def grid_trans(self, observation, action, action_mask, trans_type, rot_times=0) -> List[np.array]:
            # Grid类数据的变换
            ROW, COL = self.env.SIZE, self.env.SIZE
            MODE = self.env.mode
            # Preprocess data
            action = action.squeeze()
            action_mask = action_mask.squeeze()
            
            # Generate observation grid data
            obs_grid = observation.reshape(1 + MODE, ROW, COL)
            
            # Generate action grid data
            action_grid = np.zeros((4, ROW, COL))
            temp, direction = action // 4, action % 4
            x, y = int(temp // COL), int(temp % COL)
            direction_new = direction_trans(direction, trans_type, rot_times)
            action_grid[direction_new, x, y] = 1
            
            # Generate action_mask grid data
            action_mask_grid = np.zeros((4, ROW, COL))
            for i, bool_value in enumerate(action_mask):
                temp, direction = i // 4, i % 4
                x, y = temp // COL, temp % COL
                if bool_value:
                    direction_new = direction_trans(direction, trans_type, rot_times)
                    action_mask_grid[direction_new, x, y] = bool_value
            
            # Generate total grid data
            total_grid = np.concatenate((obs_grid, action_grid, action_mask_grid), axis=0)
            if trans_type == "rot90":
                total_grid = np.array([np.rot90(s, rot_times) for s in total_grid])
            if trans_type == "fliplr":
                total_grid = np.array([np.fliplr(s) for s in total_grid])
                
            obs_new, action_grid_new, action_mask_grid_new = total_grid[0:MODE+1], total_grid[MODE+1:MODE+5], total_grid[MODE+5::]  
            
            # Convert obs grid to standard observation
            obs_new = obs_new.flatten()
            
            # Convert action grid to standard discreate action
            index = np.where(action_grid_new > 0)
            k, i, j = list(zip(*index))[0]
            action_new = (i*COL + j)*4 + k
            
            # Convert action_mask grid to standard action_mask
            action_mask_new = np.zeros(4*ROW*COL)
            for k in range(4):
                for i in range(ROW):
                    for j in range(COL):
                            action_mask_new[(i*COL + j)*4 + k] = action_mask_grid_new[k, i, j]
            
            return obs_new, action_new, action_mask_new
        
        def exchange_trans(self, observation) -> List[np.array]:
            # 模式1下, 两子状态可以互相替换, 如下
            """
            [1 1 2 2 2]     [2 2 1 1 1]
            [1 1 1 1 1]     [2 2 2 2 2]
            [1 1 1 1 1] <=> [2 2 2 2 2]
            [1 1 1 1 1]     [2 2 2 2 2]
            [2 2 2 2 2]     [1 1 1 1 1]
            """
            # 但该函数实际的observation输入是one-hot格式, 所以需要进行转换
            ROW, COL = self.env.SIZE, self.env.SIZE
            MODE = self.env.mode
            obs_grid = observation.reshape(1 + MODE, ROW, COL)
            obs_new = np.zeros(obs_grid.shape)
            obs_new[0] = obs_grid[1]
            obs_new[1] = obs_grid[0]
            return obs_new
        
        # Data Augment main:
        # Create augmented data
        n_steps = self.rollout_buffer.episode_starts.size
        n_env = 16 if self.env.mode == 1 else 8
        DA_obs = np.zeros((n_steps, n_env, self.env.observation_space.n))
        DA_actions = np.zeros((n_steps, n_env, 1))
        DA_action_masks = np.zeros((n_steps, n_env, self.env.action_space.n))
        
        for i in range(n_steps):
            for k in [1, 2, 3, 4]:
                # Rotate 90
                obs_new, action_new, action_mask_new = grid_trans(self,
                                                                  self.rollout_buffer.observations[i], 
                                                                  self.rollout_buffer.actions[i], 
                                                                  self.rollout_buffer.action_masks[i], 
                                                                  trans_type="rot90",
                                                                  rot_times=k)
                DA_obs[i, 2*k-2] = obs_new.reshape(-1)
                DA_actions[i, 2*k-2, 0] = action_new
                DA_action_masks[i, 2*k-2] = action_mask_new.reshape(-1)
                if n_env == 16:
                    obs_new_exchange = exchange_trans(self, obs_new)
                    DA_obs[i, 2*k+6] = obs_new_exchange.reshape(-1)
                    DA_actions[i, 2*k+6, 0] = action_new
                    DA_action_masks[i, 2*k+6] = action_mask_new.reshape(-1)
                
                # Flip left-right
                obs_new, action_new, action_mask_new = grid_trans(self,
                                                                  obs_new, 
                                                                  action_new, 
                                                                  action_mask_new, 
                                                                  trans_type='fliplr')
                DA_obs[i, 2*k-1] = obs_new.reshape(1, -1)
                DA_actions[i, 2*k-1, 0] = action_new
                DA_action_masks[i, 2*k-1] = action_mask_new.reshape(1, -1)
                if n_env == 16:
                    obs_new_exchange = exchange_trans(self, obs_new)
                    DA_obs[i, 2*k+7] = obs_new_exchange.reshape(-1)
                    DA_actions[i, 2*k+7, 0] = action_new
                    DA_action_masks[i, 2*k+7] = action_mask_new.reshape(-1)
        
        # Check validity of augmented data (For debugging)
        check_DA = False
        if check_DA:
            self._check_augmented_data(DA_obs, DA_actions, DA_action_masks)
        
        # Rebuild 1.observations, 2.actions and 3.action_masks
        self.rollout_buffer.observations = DA_obs
        self.rollout_buffer.actions = DA_actions
        self.rollout_buffer.action_masks = DA_action_masks
        
        # Recompute 4.values and 5.logprobs 
        # Please check method compute_returns_and_advantage() in stable_baselines3.common.RolloutBuffer
        DA_values = np.zeros((n_steps, n_env))
        DA_log_probs = np.zeros((n_steps, n_env))
        with th.no_grad():
            DA_obs_tensor = obs_as_tensor(DA_obs, self.model.device)
            DA_actions_tensor = th.tensor(DA_actions, device=self.model.device).long()
            for batch_rank in range(n_env):
                values, log_prob, entropy = self.model.policy.evaluate_actions(DA_obs_tensor[:, batch_rank], 
                                                                               actions=DA_actions_tensor[:, batch_rank].flatten(),  
                                                                               action_masks=DA_action_masks[:, batch_rank]
                                                                               )
                DA_values[:, batch_rank] = values.cpu().numpy().reshape(-1)
                DA_log_probs[:, batch_rank] = log_prob.cpu().numpy()
        
        self.rollout_buffer.values = DA_values
        self.rollout_buffer.log_probs = DA_log_probs
        
        # Rebuild 8.episode_starts and 9.rewards
        self.rollout_buffer.buffer_size = n_steps
        self.rollout_buffer.n_envs = n_env
        self.rollout_buffer.episode_starts = np.tile(self.rollout_buffer.episode_starts, (1, n_env))
        self.rollout_buffer.rewards = np.tile(self.rollout_buffer.rewards, (1, n_env))
        
        # Recompute 6.returns and 7.advantages
        self.rollout_buffer.advantages = np.zeros((self.rollout_buffer.buffer_size, self.rollout_buffer.n_envs), dtype=np.float32)
        last_value, last_done = th.zeros(1), 1
        self.rollout_buffer.compute_returns_and_advantage(last_values=last_value, dones=last_done)
        
        self.rollout_buffer.returns = self.rollout_buffer.returns.astype(np.float32)
        self.rollout_buffer.values = self.rollout_buffer.values.astype(np.float32)
        self.rollout_buffer.log_probs = self.rollout_buffer.log_probs.astype(np.float32)
        
    
    def _check_augmented_data(self, DA_obs, DA_actions, DA_action_masks) -> None:
        n_steps = self.rollout_buffer.episode_starts.size
        ROW, COL = self.env.ROW, self.env.COL
        
        batch_rank = 6 # 0~7
        print("Check augmented data in batch rank:", batch_rank)
        for i in range(n_steps):
            print(DA_obs[i, batch_rank].reshape(ROW, COL))
            legal_action_index = np.where(DA_action_masks[i, batch_rank] == 1)[0]
            legal_actions = [self.env.std_action_to_raw(std_action) for std_action in legal_action_index]
            print("legal actions:", legal_actions)
            print("choose action:", self.env.std_action_to_raw(DA_actions[i, batch_rank, 0]))
    
    
    def _on_rollout_start(self) -> None:
        """
        This event is triggered before collecting new samples.
        In order to apply data augment, we rebuild the buffer, which will raise unfull mistake: assert self.fill "".
        Recover the buffer size and reset the buffer will solve this.
        """
        self.rollout_buffer.buffer_size = self.model.n_steps
        self.rollout_buffer.n_envs = 1
        self.rollout_buffer.reset()
        
        
    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        
        In this case, above datas are delivered to training phase for calculation:
            (please check method _get_samples() in: sb3_contrib.common.maskable.buffers
             and method train() in: sb3_contrib.ppo_mask.ppo_mask.MaskablePPO)
            1. observations
            2. actions
            3. action_masks
            4. old_values/values
            5. old_log_prob/log_probs
            6. returns
            7. advantages
            
        In order to recompute returns and advantages, done information is also needed:
            8. episode_starts
            9. rewards
        """
        # Print data for debugging
        # Remember: keep your augmented data in the same form
        if self.print_buffer_data:
            self._print_buffer_data()
        
        if self.drop_episode:
            for i in reversed(range(self.rollout_buffer.buffer_size)):
                if self.rollout_buffer.episode_starts[i, 0] == 1:
                    drop_flag = i
                    break
            # Drop undone data
            self.rollout_buffer.buffer_size = drop_flag
            self.rollout_buffer.observations = self.rollout_buffer.observations[:drop_flag]
            self.rollout_buffer.actions = self.rollout_buffer.actions[:drop_flag]
            self.rollout_buffer.action_masks = self.rollout_buffer.action_masks[:drop_flag]
            self.rollout_buffer.values = self.rollout_buffer.values[:drop_flag]
            self.rollout_buffer.log_probs = self.rollout_buffer.log_probs[:drop_flag]
            self.rollout_buffer.returns = self.rollout_buffer.returns[:drop_flag]
            self.rollout_buffer.advantages = self.rollout_buffer.advantages[:drop_flag] 
            self.rollout_buffer.episode_starts = self.rollout_buffer.episode_starts[:drop_flag] 
            self.rollout_buffer.rewards = self.rollout_buffer.rewards[:drop_flag] 
        
        if self.use_DA:
            self._data_augment()

