"""
import gym
import numpy as np
from gym import spaces

class CustomEnv(gym.Env):
    '''Custom Environment that follows gym interface.'''

    metadata = {"render.modes": ["human"]}

    def __init__(self, arg1, arg2, ...):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

    def step(self, action):
        ...
        return observation, reward, done, info

    def reset(self):
        ...
        return observation  # reward, done, info can't be included

    def render(self, mode="human"):
        ...

    def close(self):
        ...
"""
import gym
import pygame
import random
import numpy as np
from gym import spaces

from copy import deepcopy
from utils import grid_trans, get_cluster_num


class RestMinEnv_v1(gym.Env):
    metadata = {"render_mode": ["human"]}
    UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3
    
    # Since X is full of challanges, I only use H for training  
    MODE_1_TYPE = {
        "X": np.array([
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [1, 1, 1, 2, 2, 2],
            [2, 2, 2, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
            [2, 2, 2, 1, 1, 1],
        ]),
        "H": np.array([
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2],
            [2, 2, 2, 2, 2, 2]
        ])
    }
    
    TRANS_TYPE = ["none", "rot90", "rot180", "rot270", 
                  "fliplr", "rot90 + fliplr", "rot180 + fliplr", "rot270 + fliplr"] 
    
    TILE_SIZE = 40
    
    def __init__(self, 
                 size=6, 
                 mode=1, 
                 render_mode=None, 
                 use_cluster_reward=False
        ):
        super().__init__()
        assert mode in [0, 1], "Env: input mode in [0, 1]"
        assert render_mode in [None, "human"], "Env: input render_mode in [None, 'human']"
        
        self.SIZE = size
        self.mode = mode
        self.use_cluster_reward = use_cluster_reward
        
        if mode == 0:    
            self.observation_space = spaces.MultiBinary(size**2)
        elif mode == 1:
            self.observation_space = spaces.MultiBinary(2 * size**2)
        self.action_space = spaces.Discrete(size**2*4)
        
        self.render_mode = render_mode
        if render_mode == 'human':
            self.render_mode = render_mode
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.SIZE*self.TILE_SIZE, self.SIZE*self.TILE_SIZE))
            self.clock = pygame.time.Clock()
            self.FPS = 2
            
    def reset(self):
        assert self.SIZE >= 4, "Env: input size >= 4"
        assert self.mode in [0, 1], "Env: input mode in [0, 1]"
        
        if self.mode == 0:
            self.state = {"obs":np.ones((self.SIZE, self.SIZE)), "legal_actions":None, "n_step":0}
            if self.SIZE == 4:
                randx, randy = random.choice([
                (0, 1), (0, 2), (1, 0), (2, 0), (3, 1), (3, 2), (1, 3), (2, 3)
                ])
            elif self.SIZE == 5:
                randx, randy = random.choice([
                (0, 2), (1, 2), (3, 2), (4, 2), (2, 0), (2, 1), (2, 3), (2, 4)
                ]) 
            else:
                randx, randy = np.random.randint(0, self.SIZE), np.random.randint(0, self.SIZE)
            self.state["obs"][randx, randy] = 0
            self.state["legal_actions"] = self._get_legal_action()
            
        elif self.mode == 1:
            assert self.SIZE == 6, "Env: mode 1 only support size == 6"
            
            self.state = {"obs": grid_trans(self.MODE_1_TYPE["H"], random.choice(self.TRANS_TYPE)), 
                          "legal_actions": None, 
                          "n_step": 0}
            randx, randy = np.random.randint(0, self.SIZE), np.random.randint(0, self.SIZE)
            self.state["obs"][randx, randy] = 0
            self.state["legal_actions"] = self._get_legal_action()
        
        if self.render_mode == "human":
            self.render()
        
        return self._get_obs()
    
    def _get_obs(self):
        if self.mode == 0:
            obs = self.state["obs"].flatten()
        elif self.mode == 1:
            channel_0 = np.zeros((self.SIZE, self.SIZE))
            channel_1 = np.zeros((self.SIZE, self.SIZE)) 
            
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if self.state["obs"][i, j] == 1:
                        channel_0[i, j] = 1
                    elif self.state["obs"][i, j] == 2:
                        channel_1[i, j] = 1
            
            obs = np.concatenate((channel_0.flatten(), channel_1.flatten()))
        
        return obs
    
    def _get_legal_action(self):
        legal_action = []
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.state['obs'][i, j] == 0: 
                    continue 
                if 0 <= i-2 < self.SIZE:
                    if self.state['obs'][i-1, j] != 0 and self.state['obs'][i-2, j] == 0: # Jump UP
                        # (x*self.COL + y)*4 + direc_value
                        legal_action.append((i*self.SIZE+j)*4 + self.UP)           
                if 0 <= i+2 < self.SIZE:
                    if self.state['obs'][i+1, j] != 0 and self.state['obs'][i+2, j] == 0: # Jump DOWN
                        legal_action.append((i*self.SIZE+j)*4 + self.DOWN)
                if 0 <= j-2 < self.SIZE:
                    if self.state['obs'][i, j-1] != 0 and self.state['obs'][i, j-2] == 0: # Jump LEFT
                        legal_action.append((i*self.SIZE+j)*4 + self.LEFT)
                if 0 <= j+2 < self.SIZE:
                    if self.state['obs'][i, j+1] != 0 and self.state['obs'][i, j+2] == 0: # Jump RIGHT
                        legal_action.append((i*self.SIZE+j)*4 + self.RIGHT)
        
        return legal_action
    
    def get_action_mask(self):
        action_mask = np.zeros(self.action_space.n)
        action_mask[self._get_legal_action()] = 1
        
        return action_mask.astype(bool)
    
    def _is_done(self):
        return self.state["legal_actions"] == [] or self.state["n_step"] >= 50
    
    def _get_reward(self):
        count = 0
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if self.state["obs"][i, j] > 0: 
                    count += 1
        if self.mode == 0:
            if count == 1:
                reward = 100
            else:
                reward = 2*(self.SIZE - count)
        elif self.mode == 1:
            if count == 2:
                reward = 100
            else:
                reward = 15 - count
        return reward
    
    def std_action_to_raw(self, std_action):
        direction = int(std_action % 4)
        temp = std_action // 4
        x, y = int(temp // self.SIZE), int(temp % self.SIZE)
        return x, y, direction
    
    def _get_info(self):
        return {}
    
    def step(self, action):
        assert action in self.state["legal_actions"], "Env: step() input an ilegal action."
        
        x, y, direction = self.std_action_to_raw(action)
        Gid = self.state["obs"][x, y]
        self.state["obs"][x, y] = 0
        
        if direction == self.UP:
            if self.state["obs"][x-1, y] == Gid:
                self.state["obs"][x-1, y] = 0
                self.state["obs"][x-2, y] = Gid
            else:
                self.state["obs"][x-2, y] = Gid
                
        elif direction == self.DOWN:
            if self.state["obs"][x+1, y] == Gid:
                self.state["obs"][x+1, y] = 0
                self.state["obs"][x+2, y] = Gid
            else:
                self.state["obs"][x+2, y] = Gid
                
        elif direction == self.LEFT:
            if self.state["obs"][x, y-1] == Gid:
                self.state["obs"][x, y-1] = 0
                self.state["obs"][x, y-2] = Gid
            else:
                self.state["obs"][x, y-2] = Gid
                
        elif direction == self.RIGHT:
            if self.state["obs"][x, y+1] == Gid:
                self.state["obs"][x, y+1] = 0
                self.state["obs"][x, y+2] = Gid
            else:
                self.state["obs"][x, y+2] = Gid
        
        self.state["legal_actions"] = self._get_legal_action()
        self.state["n_step"] += 1
        obs = self._get_obs()
        done = self._is_done()
        
        reward = self._get_reward() if done else 0
        if self.use_cluster_reward:
            cluster_num = get_cluster_num(self.state)
            if cluster_num == 2:
                reward += 1
            else:
                reward += 0.1 * (5 - cluster_num)
        
        info = self._get_info()
        
        if self.render_mode == "human":
            self.render()
        
        return obs, reward, done, info
    
    def random_step(self):
        action = np.random.choice(self.state["legal_actions"])
        return self.step(action)
    
    def render(self):
        canvas = pygame.Surface((self.SIZE*self.TILE_SIZE, self.SIZE*self.TILE_SIZE))
        canvas.fill((255, 255, 255))
        
        # Draw some gridlines
        for x in range(self.SIZE + 1):
            pygame.draw.line(
                canvas,
                (120, 120, 120),
                (0, self.TILE_SIZE * x),
                (self.TILE_SIZE*self.SIZE, self.TILE_SIZE * x),
                width=2,
            )
            pygame.draw.line(
                canvas,
                (120, 120, 120),
                (self.TILE_SIZE * x, 0),
                (self.TILE_SIZE * x, self.TILE_SIZE*self.SIZE),
                width=2,
            )
        
        # Draw chess
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                
                if self.state["obs"][i, j] == 1:
                    pygame.draw.circle(
                        canvas,
                        (105, 100, 222),
                        ((j+0.5)*self.TILE_SIZE, (i+0.5)*self.TILE_SIZE),
                        self.TILE_SIZE / 3,
                    )
                
                elif self.state["obs"][i, j] == 2:
                    pygame.draw.circle(
                        canvas,
                        (222, 140, 212),
                        ((j+0.5)*self.TILE_SIZE, (i+0.5)*self.TILE_SIZE),
                        self.TILE_SIZE / 3,
                    )
        
        self.window.blit(canvas, canvas.get_rect())
        pygame.display.update()
        pygame.event.pump() # Handle the events, without this command the surface may be freezed 
        self.clock.tick(self.FPS)
    
    def close(self):
        if self.render_mode == "human":
            if self.window is not None:
                pygame.display.quit()
                pygame.quit()
    
    # Methods for MCTS
    def step_in_mcts(self, state, action):
        assert action in state["legal_actions"], "MCTS: step() input an illegal action."
        
        x, y, direction = self.std_action_to_raw(action)
        new_state = deepcopy(state)
        Gid = new_state["obs"][x, y]
        new_state["obs"][x, y] = 0
        
        if direction == self.UP:
            if new_state["obs"][x-1, y] == Gid:
                new_state["obs"][x-1, y] = 0
                new_state["obs"][x-2, y] = Gid
            else:
                new_state["obs"][x-2, y] = Gid
                
        elif direction == self.DOWN:
            if new_state["obs"][x+1, y] == Gid:
                new_state["obs"][x+1, y] = 0
                new_state["obs"][x+2, y] = Gid
            else:
                new_state["obs"][x+2, y] = Gid
                
        elif direction == self.LEFT:
            if new_state["obs"][x, y-1] == Gid:
                new_state["obs"][x, y-1] = 0
                new_state["obs"][x, y-2] = Gid
            else:
                new_state["obs"][x, y-2] = Gid
                
        elif direction == self.RIGHT:
            if new_state["obs"][x, y+1] == Gid:
                new_state["obs"][x, y+1] = 0
                new_state["obs"][x, y+2] = Gid
            else:
                new_state["obs"][x, y+2] = Gid
        
        new_state["legal_actions"] = self.get_legal_action_in_mcts(new_state)
        new_state["n_step"] += 1
                
        return new_state
    
    def get_legal_action_in_mcts(self, state):
        legal_actions = []
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if state['obs'][i, j] == 0: 
                    continue 
                if 0 <= i-2 < self.SIZE:
                    if state['obs'][i-1, j] != 0 and state['obs'][i-2, j] == 0: # Jump UP
                        # (x*self.COL + y)*4 + direc_value
                        legal_actions.append((i*self.SIZE+j)*4 + self.UP)           
                if 0 <= i+2 < self.SIZE:
                    if state['obs'][i+1, j] != 0 and state['obs'][i+2, j] == 0: # Jump DOWN
                        legal_actions.append((i*self.SIZE+j)*4 + self.DOWN)
                if 0 <= j-2 < self.SIZE:
                    if state['obs'][i, j-1] != 0 and state['obs'][i, j-2] == 0: # Jump LEFT
                        legal_actions.append((i*self.SIZE+j)*4 + self.LEFT)
                if 0 <= j+2 < self.SIZE:
                    if state['obs'][i, j+1] != 0 and state['obs'][i, j+2] == 0: # Jump RIGHT
                        legal_actions.append((i*self.SIZE+j)*4 + self.RIGHT)
        
        return legal_actions
    
    def get_reward_in_mcts(self, state):
        count = 0
        for i in range(self.SIZE):
            for j in range(self.SIZE):
                if state["obs"][i, j] > 0: 
                    count += 1
        if self.mode == 0:
            if count == 1:
                reward = 1
            else:
                reward = 0
        elif self.mode == 1:
            if count == 2:
                reward = 1
            else:
                reward = 0
                
        return reward 
    
    def get_action_mask_in_mcts(self, state):
        action_mask = np.zeros(self.action_space.n)
        action_mask[state["legal_actions"]] = 1
        
        return action_mask.astype(bool)
    
    def get_obs_in_mcts(self, state):
        if self.mode == 0:
            obs = state["obs"].flatten()
        elif self.mode == 1:
            channel_0 = np.zeros((self.SIZE, self.SIZE))
            channel_1 = np.zeros((self.SIZE, self.SIZE)) 
            
            for i in range(self.SIZE):
                for j in range(self.SIZE):
                    if state["obs"][i, j] == 1:
                        channel_0[i, j] = 1
                    elif state["obs"][i, j] == 2:
                        channel_1[i, j] = 1
            
            obs = np.concatenate((channel_0.flatten(), channel_1.flatten()))
        
        return obs
    
    
if __name__ == "__main__":
    env = RestMinEnv_v1(size=6, mode=1, render_mode=None, use_cluster_reward=True)
    obs = env.reset()
    rewards = []
    while True:
        print(env.state["obs"], "\n")
        obs, reward, done, info = env.random_step()
        print("Reward:", reward, "\n")
        rewards.append(reward)
        if done:
            break
    env.close()
    
    rewards = np.array(rewards)
    print("Return:", rewards.sum())
    









































