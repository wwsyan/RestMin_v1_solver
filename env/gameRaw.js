class Game {
    constructor () {
        this.round = 0;
        this.state = null;
    }

    // 返回棋子XY的合法动作
    getLegalAction (X, Y, state) {
        const obs = deepcopy(state['obs']);

        if (obs[X][Y] == 0) {
            return []
        }

        const Gid = obs[X][Y];
        let legalAction = []; 
        // 上
        if (0 <= X-2 && X-2 < 6) {
            if (obs[X-1][Y] != 0 && obs[X-2][Y] == 0) { 
                if (obs[X-1][Y] == Gid) {
                    legalAction.push(['up', 'same']);
                } 
                else {
                    legalAction.push(['up', 'diverse']);
                }    
            }  
        }
        // 下
        if (0 <= X+2 && X+2 < 6) {
            if (obs[X+1][Y] != 0 && obs[X+2][Y] == 0) {
                if (obs[X+1][Y] == Gid) {
                    legalAction.push(['down', 'same']);
                } 
                else {
                    legalAction.push(['down', 'diverse']);
                }
                
            }  
        }
        // 左
        if (0 <= Y-2 && Y-2 < 6) {
            if (obs[X][Y-1] != 0 && obs[X][Y-2] == 0) { 
                if (obs[X][Y-1] == Gid) {
                    legalAction.push(['left', 'same']);
                } 
                else {
                    legalAction.push(['left', 'diverse']);
                }
            }  
        }
        // 右
        if (0 <= Y+2 && Y+2 < 6) {
            if (obs[X][Y+1] != 0 && obs[X][Y+2] == 0) { 
                if (obs[X][Y+1] == Gid) {
                    legalAction.push(['right', 'same']);
                } 
                else {
                    legalAction.push(['right', 'diverse']);
                }
            }  
        }
        return legalAction
    }

    // 根据棋子XY的合法动作，返回合法位置
    getLegalPos (X, Y, state) {
        let legalAction = this.getLegalAction(X, Y, state);
        let legalPos = [];
        for (let i = 0; i < legalAction.length; i++) {
            switch (legalAction[i][0]) {
                case 'up':
                    legalPos.push((X - 2)*6 + Y);
                    break;
                case 'down':
                    legalPos.push((X + 2)*6 + Y);
                    break;
                case 'left':
                    legalPos.push(X*6 + Y - 2);
                    break;
                case 'right':
                    legalPos.push(X*6 + Y + 2);
                    break;
            }
        }
        return legalPos
    }

    // 返回state下的所有合法动作
    getAllLegalAction (state) {
        const s = deepcopy(state);
        let allLegalAction = [];
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                let legalAction = this.getLegalAction(i, j, s); 
                if (legalAction.length > 0) {
                    allLegalAction.push([i, j, legalAction]);
                }
            }
        }
        return allLegalAction
    }

    // 进行状态更新的step函数，返回动作之后的新状态
    // state = {'obs': 6×6 array, 'legalAction': array}
    // legalAction = [x, y, [direction, type]]
    // action = [x, y, direction]
    step(state, action) {
        let s = deepcopy(state);
        let [x, y, direction] = action;
        let Gid = s['obs'][x][y];

        let flag = 'illegal';
        switch (direction) {
            case 'up':
                if (x - 2 < 0) {break}
                if (s['obs'][x - 1][y] != 0 && s['obs'][x - 2][y] == 0) {flag = 'up'}
                break
            case 'down':
                if (x + 2 > 6) {break}
                if (s['obs'][x + 1][y] != 0 && s['obs'][x + 2][y] == 0) {flag = 'down'}
                break
            case 'left':
                if (y - 2 < 0) {break}
                if (s['obs'][x][y - 1] != 0 && s['obs'][x][y - 2] == 0) {flag = 'left'}
                break
            case 'right':
                if (y + 2 < 0) {break}
                if (s['obs'][x][y + 1] != 0 && s['obs'][x][y + 2] == 0) {flag = 'right'}
                break
        }

        if (flag == 'illegal') {
            console.log('不合法的动作');
            return 'illegal action'
        }

        switch (flag) {
            case 'up':
                s['obs'][x][y] = 0;
                if (s['obs'][x - 1][y] == Gid) {
                    s['obs'][x - 1][y] = 0;
                    s['obs'][x - 2][y] = Gid;
                }
                else {
                    s['obs'][x - 2][y] = Gid;
                }
                break
            
            case 'down':
                s['obs'][x][y] = 0;
                if (s['obs'][x + 1][y] == Gid) {
                    s['obs'][x + 1][y] = 0;
                    s['obs'][x + 2][y] = Gid;
                }
                else {
                    s['obs'][x + 2][y] = Gid;
                }
                break

            case 'left':
                s['obs'][x][y] = 0;
                if (s['obs'][x][y - 1] == Gid) {
                    s['obs'][x][y - 1] = 0;
                    s['obs'][x][y - 2] = Gid;
                }
                else {
                    s['obs'][x][y - 2] = Gid;
                }
                break

            case 'right':
                s['obs'][x][y] = 0;
                if (s['obs'][x][y + 1] == Gid) {
                    s['obs'][x][y + 1] = 0;
                    s['obs'][x][y + 2] = Gid;
                }
                else {
                    s['obs'][x][y + 2] = Gid;
                }
                break
        }

        // 更新合法动作集
        s['legalAction'] = this.getAllLegalAction(s);

        // 判断是否终局
        let done = this.isEnd (s);

        // 若终局，则计算最终得分
        let reward;
        if (done) {
            reward = this.getScore(s);
        }
        else {reward = 0}

        return {
            nextState: s,
            done: done,
            reward: reward  
        }
    }

    // 该函数仅由isEnd函数调用
    step_0(state, action) {
        let s = deepcopy(state);
        let [x, y, direction] = action;
        let Gid = s['obs'][x][y];

        let flag = 'illegal';
        switch (direction) {
            case 'up':
                if (x - 2 < 0) {break}
                if (s['obs'][x - 1][y] != 0 && s['obs'][x - 2][y] == 0) {flag = 'up'}
                break
            case 'down':
                if (x + 2 > 6) {break}
                if (s['obs'][x + 1][y] != 0 && s['obs'][x + 2][y] == 0) {flag = 'down'}
                break
            case 'left':
                if (y - 2 < 0) {break}
                if (s['obs'][x][y - 1] != 0 && s['obs'][x][y - 2] == 0) {flag = 'left'}
                break
            case 'right':
                if (y + 2 < 0) {break}
                if (s['obs'][x][y + 1] != 0 && s['obs'][x][y + 2] == 0) {flag = 'right'}
                break
        }

        if (flag == 'illegal') {
            console.log('不合法的动作');
            return 'illegal action'
        }

        switch (flag) {
            case 'up':
                s['obs'][x][y] = 0;
                if (s['obs'][x - 1][y] == Gid) {
                    s['obs'][x - 1][y] = 0;
                    s['obs'][x - 2][y] = Gid;
                }
                else {
                    s['obs'][x - 2][y] = Gid;
                }
                break
            
            case 'down':
                s['obs'][x][y] = 0;
                if (s['obs'][x + 1][y] == Gid) {
                    s['obs'][x + 1][y] = 0;
                    s['obs'][x + 2][y] = Gid;
                }
                else {
                    s['obs'][x + 2][y] = Gid;
                }
                break

            case 'left':
                s['obs'][x][y] = 0;
                if (s['obs'][x][y - 1] == Gid) {
                    s['obs'][x][y - 1] = 0;
                    s['obs'][x][y - 2] = Gid;
                }
                else {
                    s['obs'][x][y - 2] = Gid;
                }
                break

            case 'right':
                s['obs'][x][y] = 0;
                if (s['obs'][x][y + 1] == Gid) {
                    s['obs'][x][y + 1] = 0;
                    s['obs'][x][y + 2] = Gid;
                }
                else {
                    s['obs'][x][y + 2] = Gid;
                }
                break
        }

        // 更新合法动作集
        s['legalAction'] = this.getAllLegalAction(s);

        return s
    }

    // 随机执行一个合法动作
    randomStep (state) {
        let actions = this.getAllLegalAction(state);
        let randomChess = actions[ Math.floor(Math.random() * actions.length) ];
        let [x, y, direction_and_type] = [randomChess[0], randomChess[1], randomChess[2]];
        let randomDirection = direction_and_type[ Math.floor(Math.random() * direction_and_type.length)][0];
        let action = [x, y, randomDirection];
        return this.step(state, action)
    } 

    // 终局判定函数
    isEnd (currentState) {
        const state = deepcopy(currentState);

        // 第一死局充分条件的判定
        console.log('第一死局充分条件判定中...');
        if (state['legalAction'].length == 0) {
            console.log('结果：终局')
            return true
        }


        // 同色消存在的判定
        // SE = Same Color Elimination
        const check_SCE = (currentState) => {
            for (let i = 0; i < currentState['legalAction'].length; i++) {
                for (let j = 0; j < currentState['legalAction'][i][2].length; j++) {
                    if (currentState['legalAction'][i][2][j][1] == 'same') {
                        return true
                    }    
                }
            }
            return false
        }
        console.log('同色消存在判定中...'); 
        if (check_SCE(state)) {
            console.log('结果：存在同色消，未终局');
            return
        }


        // 第二死局充分条件的判定
        console.log('第二死局充分条件判定中...');
        // let attentionList = [];
        let flag = false;
        for (let i = 0; i < 36; i++) {  
            for (let j = 0; j < 36; j++) {
                if (j <= i) {continue}

                let [x1, y1] = [Math.floor(i/6), i%6];
                let [x2, y2] = [Math.floor(j/6), j%6];
                
                // 同色消必要条件
                if (state['obs'][x1][y1] != 0 && state['obs'][x1][y1] == state['obs'][x2][y2]) {
                    let flag_1 = (Math.abs(x1 - x2) == 0) && (Math.abs(y1 - y2) % 2 == 1);
                    let flag_2 = (Math.abs(y1 - y2) == 0) && (Math.abs(x1 - x2) % 2 == 1);
                    let flag_3 = (Math.abs(x1 - x2) == 1) && (Math.abs(y1 - y2) % 2 == 0);
                    let flag_4 = (Math.abs(y1 - y2) == 1) && (Math.abs(x1 - x2) % 2 == 0);
                
                    if (flag_1 || flag_2 || flag_3 || flag_4) {
                        // attentionList.push([[x1, y1], [x2, y2]]);
                        flag = true;
                        break;
                    }
                }
            }
            if (flag) {break}
        }
        // console.log('可能发生同色消的组合数：', attentionList.length);
        // if (attentionList.length == 0) {
        //     console.log('结果：终局');
        //     return true
        // }
        if (!flag) {
            console.log('结果：终局');
            return true
        }


        // 递归函数：遍历 depthMax 次动作产生的状态，每次动作后都判断是否存在同色消
        // 在棋子较多时，直接进行一定量的推演应该会是更有效的
        const stepToCheckSCE = (currentState, depth, depthMax, historyState) => {
            // 若到达最大搜索深度，则直接返回
            console.log(`当前搜索深度：${depth}, 最大搜索深度：${depthMax}`);
            if (depth == depthMax) {return}

            const s = deepcopy(currentState);

            // 若死局，则直接返回
            if (s['legalAction'].length == 0) {
                return false
            }
            // 若存在同色消，则直接返回
            if (check_SCE(s)) {
                return true
            }

            // 遍历所有合法动作，执行递归
            // s['legalAction'][i] = [ x, y, [[direction1, type1], [direction2, type2]...] ]
            for (let i = 0; i < s['legalAction'].length; i++) {
                let chess = s['legalAction'][i];

                let [chessX, chessY] = [chess[0], chess[1]];
                for (let j = 0; j < chess[2].length; j++) {
                    let direction = chess[2][j][0];
                    let action = [chessX, chessY, direction];
                    let newState = this.step_0(s, action);

                    // 跳过重复访问的状态
                    let flag = false;
                    for (let k = 0; k < historyState.length; k++) {
                        if (JSON.stringify(newState['obs']) == JSON.stringify(historyState[k]['obs'])) {
                            flag = true;
                            break
                        }
                    }
                    if (flag) {continue}
                    else {
                        console.log('新状态：', newState);
                        historyState.push(deepcopy(newState));
                        console.log('状态数：', historyState.length);
                    }

                    let result = stepToCheckSCE(newState, depth+1, depthMax, historyState);
                    if (result) {return true}
                }
            }
            return false
        }

        console.log('递归搜索中...');
        let result = stepToCheckSCE(state, 0, 6, [state]);
        if (result) {
            console.log('结果：未终局');
        }
        else {
            console.log('结果：终局');
        }
    }

    // 计算输入状态的得分
    getScore (state) {
        let count = 0;
        for (let i = 0; i < 6; i++) {
            for (let j = 0; j < 6; j++) {
                if (state['obs'][i][j] > 0) {
                    count ++;
                }
            }
        }
        return 36 - count
    }
}