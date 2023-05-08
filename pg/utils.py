import numpy as np
from copy import deepcopy


def grid_trans(np_array, trans_type):
    if trans_type == "none":
        new_array = np_array
    
    elif trans_type == "rot90":
        new_array = np.rot90(np_array, 1)
    
    elif trans_type == "rot180":
        new_array = np.rot90(np_array, 2)
        
    elif trans_type == "rot270":
        new_array = np.rot90(np_array, 3)
    
    elif trans_type == "fliplr":
        new_array = np.fliplr(np_array)
    
    elif trans_type == "rot90 + fliplr":
        new_array = np.rot90(np_array, 1)
        new_array = np.fliplr(new_array)
        
    elif trans_type == "rot180 + fliplr":
        new_array = np.rot90(np_array, 2)
        new_array = np.fliplr(new_array)
    
    elif trans_type == "rot270 + fliplr":
        new_array = np.rot90(np_array, 3)
        new_array = np.fliplr(new_array)
        
    return deepcopy(new_array)


def get_cluster_num(state):
    # 返回区域的数量: 类扫雷的无雷区展开机制
    SIZE = state["obs"].shape[0]
    def get_cluster(x, y):
        cluster = []
        check_box = []
        check_box.append((x, y))
        while len(check_box):
            c_x, c_y = check_box.pop() # check index (x, y) from checkbox
            Gid = base[0, c_x, c_y]
            base[1, c_x, c_y] = 0
            cluster.append((c_x, c_y))
            if c_x - 1 >= 0:
                if base[0, c_x - 1, c_y] == Gid and base[1, c_x - 1, c_y]:
                    check_box.append((c_x - 1, c_y))
            if c_x + 1 < SIZE:
                if base[0, c_x + 1, c_y] == Gid and base[1, c_x + 1, c_y]:
                    check_box.append((c_x + 1, c_y))
            if c_y - 1 >= 0:
                if base[0, c_x, c_y - 1] == Gid and base[1, c_x, c_y - 1]:
                    check_box.append((c_x, c_y - 1))
            if c_y + 1 < SIZE:
                if base[0, c_x, c_y + 1] == Gid and base[1, c_x, c_y + 1]:
                    check_box.append((c_x, c_y + 1))
        return cluster
    
    base = np.ones((2, SIZE, SIZE))
    base[0] = state['obs']
    clusters = []
    for i in range(SIZE):
        for j in range(SIZE):
            base[1, i, j] = 0
            if base[0, i, j] == 0:
                continue
            if clusters == []:
                clusters.append(get_cluster(i, j))
            flag = True
            for cluster in clusters:
                if (i, j) in cluster:
                    flag = False
                    break
            if flag:
                clusters.append(get_cluster(i, j))
                
    return len(clusters)









