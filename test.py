import game.plane_game as game
import numpy as np
import random
import json
GAME = 'bird'  
json_data = json.load(open("assets/data_random.json"))# the name of the game being played for log files
plane_game = game.GameState(json_data)
action0 = np.array([2*random.random()-1,2*random.random()-1])
action0 = np.concatenate((action0, np.ones(5)),axis=0)
action0 = np.concatenate((action0, np.zeros(95)),axis=0)
observation0, reward0, terminal = plane_game.frame_step(action0)



while True:
    # 动作只有两种，分别用【1，0】和【0，1】表示
    action0 = np.array([2*random.random()-1,2*random.random()-1])
    action0 = np.concatenate((action0, np.ones(5)),axis=0)
    action0 = np.concatenate((action0, np.zeros(95)),axis=0)
    observation0, reward0, terminal = plane_game.frame_step(action0)
    print(observation0)
    #  observation0是一张图像
    # 当失败时reward0为-1，还存活则reward为0.1
    # print(observation0.shape)
    # print(reward0)
    # print(terminal)