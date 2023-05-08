import numpy as np
import copy
from stable_baselines3.common.utils import obs_as_tensor

def softmax(x):
    # 输入value向量, 返回probability向量
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in enumerate(action_priors):
            if action not in self._children and prob > 0:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        # dictionary = {action_1: node_1, action_2, node_2}
        # act_node = dictionary[i] = [action_i, node_i] 
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None
    
    
class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, env, device, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._env = env
        self._device = device
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(True):
            if node.is_leaf():
                break
            # Greedily select next action.
            action, node = node.select(self._c_puct)
            state = self._env.step_in_mcts(state, action)

        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v in [-1, 1]
        # for the current player.
        obs = self._env.get_obs_in_mcts(state)
        obs = obs_as_tensor(obs.reshape(1, 1, -1), self._device)
        
        action_mask = self._env.get_action_mask_in_mcts(state)
        action_d = self._policy.get_distribution(obs, action_mask)
        logits = action_d.distribution.logits.reshape(-1).cpu().detach().numpy()
        action_probs = softmax(logits)
        # 由于计算误差, action_probs概率存在不合法动作的概率为正数 
        action_probs = action_probs * action_mask
        
        leaf_value = self._policy.predict_values(obs).reshape(-1).cpu().detach().numpy()
        
        done = state["legal_actions"] == [] 
        if not done:
            node.expand(action_probs)
        else:
            reward = self._env.get_reward_in_mcts(state)
            leaf_value = reward

        # Update value and visit count of nodes in this traversal.
        node.update_recursive(leaf_value)

    def get_action_probs(self, state, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # calc the move probabilities based on visit counts at the root node
        # 注意 zip(*data) 的用法, 这是个固定搭配 (其实就是zip的逆操作, 解压缩)
        # pairs = [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'd')]
        # numbers, letters = zip(*pairs)
        # -> numbers = (1, 2, 3, 4)
        # -> letters = ('a', 'b', 'c', 'd')
        act_visits = [(act, node._n_visits)
                      for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0/temp * np.log(np.array(visits) + 1e-10))
        
        # acts: tuple, act_probs: np.array
        return acts, act_probs

    def update_with_action(self, last_action):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_action in self._root._children:
            self._root = self._root._children[last_action]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"
    

class MCTSPlayer(object):
    """AI player based on MCTS"""

    def __init__(self, 
                 policy_value_function,
                 env, 
                 device,
                 c_puct=5, 
                 n_playout=500, 
                 is_train=False,
                 ):
        
        self._env = env
        self.mcts = MCTS(env,     
                         device,
                         policy_value_function,
                         c_puct, 
                         n_playout)
        self._is_train = is_train

    def reset_player(self):
        # Reset root node
        self.mcts.update_with_move(-1)

    def get_action(self, state, temp=1e-3, return_prob=False, add_noise=False):
        legal_action = state["legal_actions"]
        assert len(legal_action) > 0, "MCTS: No legal actions."
        
        action_probs = np.zeros(self._env.action_space.n)
        acts, probs = self.mcts.get_action_probs(state, temp)
        action_probs[list(acts)] = probs
        
        # _is_train=True 时保留已经建的树 (该功能有bug)
        # if self._is_train:
        #     # add Dirichlet Noise for exploration (needed for
        #     # self-play training)
        #     action = np.random.choice(
        #         acts,
        #         p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
        #     )
        #     # update the root node and reuse the search tree
        #     self.mcts.update_with_action(action)
        # else:
        #     # with the default temp=1e-3, it is almost equivalent
        #     # to choosing the move with the highest prob
        #     action = np.random.choice(acts, p=probs)
        #     # reset the root node
        #     self.mcts.update_with_action(-1)
        
        if add_noise:
            action = np.random.choice(
                    acts,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
            self.mcts.update_with_action(-1)
        else:
            action = np.random.choice(acts, p=probs)
            self.mcts.update_with_action(-1)
            
        if return_prob:
            return action, action_probs
        else:
            return action
            

    def __str__(self):
        return "MCTS {}".format(self.player)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    



