# maze using Q-learning algorithm.

import numpy as np
from maze_recursive_division import generateMaze


def init(t_nrow, t_ncol, t_nAction=4):
    r"""
            |   DOWN      Right       LEFT      UP
    --------|----------------------------------------
    (0, 0)  |
    (0, 1)  |
    (., .)  |
    """
    nState = t_nrow * t_ncol
    # initialize Q table
    Qtable = np.zeros((nState, t_nAction), dtype=np.float32)
    
    # an array for "state_idx --> pixel position"
    States = np.empty((nState,2), np.int16)
    for i in range(t_nrow):
        for j in range(t_ncol):
            index = i * t_ncol + j
            States[index] = i, j
    
    # an array for "choice --> position increment"
    movements = np.array([[+1,0], [0,+1], [0,-1], [-1,0]], np.int16)
    
    # an array for "maze pixel value --> reward"
    rewards = np.array([-100,-1000,+1000,+1], np.int16)

    return Qtable, movements, States, rewards

def move(QTable_row, t_pos, t_movements, eps=0.9):
    r"""
    modify t_pos and return choice
    """
    # if all 4 value in a row are all 0, then pick choice randomly 
    if (QTable_row[:] == 0.).all(): # in this case it seems that eps-greedy is unnecessary 
        choice = np.random.randint(low=0, high=QTable_row[:].size, size=1)[0]
    # else pick the one with largest QTable value
    else:
        choice = QTable_row[:].argmax()

    # modify position
    t_pos += t_movements[choice, :]

    return choice



def playAGame(t_Param, t_Qtable, t_Movements, t_States, t_Rewards, t_Maze, t_line=None, t_point=None):
    r"""
            |   y           x
    --------|-----------------
    start   |   1           0
    end     |   -2          -1


            |environment         reward
    --------|-----------------------------
    barrier |   0                   -100
    start   |   1                   -1000
    end     |   2                   +1000
    road    |   3                   +1
    """
    # start from the position next to the entrance of maze.
    pos = np.array([1,1], np.int16)
    
    # a list to memorize history step with maximum memory length of 2
    path = [0,0]
    
    # update plot
    if t_line is not None and t_point is not None:
        xdata = [pos[1],]; ydata = [pos[0],]
        t_line.set_xdata(xdata); t_line.set_ydata(ydata)
        t_point.set_xdata([pos[1],]); t_point.set_ydata(pos[0,])
        #t_line.figure.canvas.draw()
        plt.pause(0.01)

    for k in range(t_Param["nStep_Max"]):
        # calculate current state index
        state_idx = t_Param["ncol"] * pos[0] + pos[1]

        # modify history
        path.append( state_idx ); path.remove( path[0] )

        # update current position , and then return choice
        choice = move(t_Qtable[state_idx, :], pos, t_Movements)

        # update plot
        if t_line is not None and t_point is not None:
            xdata.append(pos[1]); ydata.append(pos[0])
            t_line.set_xdata(xdata); t_line.set_ydata(ydata)
            t_point.set_xdata([pos[1],]); t_point.set_ydata(pos[0,])
            #t_line.figure.canvas.draw()
            plt.pause(0.01)

        # calculate new state index
        state_idx_new = t_Param["ncol"] * pos[0] + pos[1]
        #print(f"[{pos[0]:>2d}, {pos[1]:2d}]", end="  ")
        # get environment; based on the new position, get reward
        env = t_Maze[pos[0], pos[1]]
        
        # if is turning back, punish
        if state_idx_new in path:
            R = -2
        # get reward from the Maze pixel value of the new state
        else:
            R = t_Rewards[ env ]

        # update Qtable
        try:
            t_Qtable[state_idx,choice] = (1-Param["alpha"]) * t_Qtable[state_idx,choice] + \
                                Param["alpha"] * (R + Param["gamma"] * t_Qtable[state_idx_new, :].max())
        except IndexError:
            print(pos[0],pos[1])
            break

        # whether game over
        if env != 3:
            break

    step = k+1
    
    # if reach maximum nStep, set env to 4
    if step == t_Param["nStep_Max"]:
        env = 4

    return env, step, tuple(pos)


if __name__ == "__main__":

    Param = {
        "nrow": 5*3,
        "ncol": 5*3,
        "nGame": 100,
        "nStep_Max": 100,
        "alpha": 0.8,
        "gamma": 0.2
    }

    # generate Maze
    Maze= generateMaze(Param["nrow"]//3, Param["ncol"]//3, seed=24)

    # Display the image
    from matplotlib import pyplot as plt
    plt.imshow(Maze, interpolation='none')
    line, = plt.plot([], [], "-", linewidth=1, color="red")
    point, = plt.plot([], [], "o", linewidth=1, color="red")
    plt.pause(0.1)

    #print(set(Maze.reshape(-1).tolist()))
    Qtable, Movements, States, Rewards = init(Param["nrow"], Param["ncol"], t_nAction=4)
    
    nstep_old = 0
    for g in range(Param["nGame"]):
        #break  # turn off learning
        env, step_collapse, final_pos = playAGame(Param, Qtable, Movements,
                            States, Rewards, Maze, t_line=line, t_point=point)
        
        # print out the final state, collapsed step and final position
        print(env, step_collapse, final_pos)
        
        # if reaching goal with a constant number of step, quit learning.
        if env == 2 and step_collapse == nstep_old:
            break
        else:
            nstep_old = step_collapse

    plt.show()
