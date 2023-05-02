# RestMin_v1_solver
本工程使用 强化学习框架 [Stable-Baselines3](https://stable-baselines3.readthedocs.io/en/master/guide/install.html)。

## 环境介绍
该环境是一种中国古代棋的简化/魔改版本，分为两种模式。玩家经过一系列动作，减少棋盘中的棋子，终局时棋子剩得越少，得分越高。
<ul>
<li>模式0下，棋子皆为单色。棋子可以以相邻棋子作为跳板进行移动，移动后，跳板棋子移除。</li>
<li>模式1下，有双色棋子，棋子的移动规则同模式0，但只有跳板棋子与跳跃棋子同色时，才会被移除，反之会被保留。由于合法动作大量增加，探索难度显著提升。</li>
</ul>

| 模式 0 | 模式 1 |
| :---: | :---: |
| <img src="img/gameplay_mode0.gif" width="75%" height="75%"> | <img src="img/gameplay_mode1.gif" width="75%" height="75%"> |

### 终局条件
该棋的游玩版本和训练版本有着不同的终局条件。在[游玩版本](https://github.com/wwsyan/RestMin_v1_solver/blob/main/env/gameRaw.js#L296)中，
终局判断函数会按顺序进行以下的判定：
<ol>
<li>若不存在合法动作，则死局（<b>第一死局充分条件</b>）；</li>
<li>若存在同色棋子消除，则非死局（<b>非死局充分条件</b>）；</li>
<li>
  对于两个同色棋子 $(x_1,y_1)$ 和 $(x_2,y_2)$，若不满足（ $\bmod$ 指取余）：
  $|x_1-x_2|=0 \land |y_1-y_2|\bmod2=1$ 或
  $|y_1-y_2|=0 \land |x_1-x_2|\bmod2=1$ 或
  $|x_1-x_2|=1 \land |y_1-y_2|\bmod2=0$ 或
  $|y_1-y_2|=1 \land |x_1-x_2|\bmod2=0$，
  那么该同色的棋子对无法进行同色消除。
  遍历所有同色的棋子对，若全都无法进行同色消除，那么死局（<b>第二死局充分条件</b>）；
</li>
<li>启用递归算法，遍历未来一定步数内的所有状态，并进行非死局充分条件的判定，若存在非死局状态，则判定为非死局。</li>
</ol>

使用该组合判定的原因是：
<ol>
<li>未找到死局的充分必要条件，只找到一些充分条件；</li>
<li>可以仅使用一个足够深的递归判定来满足需要，但会带来较严重的卡顿，影响体验；</li>
</ol>

由于上述组合判定有着一定的计算负担，会较大地影响训练速度。因此，在[训练版本](https://github.com/wwsyan/RestMin_v1_solver/blob/main/env/env_pure.py#L181)中，
终局条件简化为：
<ol>
<li>若不存在合法动作，则死局（<b>第一死局充分条件</b>）；</li>
<li>若已行动步数达到50步，则死局。</li>
</ol>

### 奖励
在标准版本中，奖励只有在终局时获得，其余状态下都是0。如果最终达到了最优解，即最少的棋子数，那么将获得有区分度的大奖励
（[Detail](https://github.com/wwsyan/RestMin_v1_solver/blob/main/env/env_pure.py#L184)）。

### 观测空间与动作空间
对于 $6×6$ 的棋盘， 模式 0 和 模式 1 的 $observation$ 分别是 <code>MultiBinary(36)</code> 和 <code>MultiBinary(72)</code>；
$action$ 是 <code>Discrete(36*4)</code>，意为选中一个位置的棋子，并进行上下左右四个方向的移动，由于有大量不合法动作，所以训练时要使用动作掩码
（[Detail](https://github.com/wwsyan/RestMin_v1_solver/blob/main/env/env_pure.py#L153)）。

## 模式0：经典PPO












