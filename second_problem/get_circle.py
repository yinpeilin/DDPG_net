import numpy as np
import torch
import sys
sys.path.append("./")
from brain.DDPG_net import DDPG
from game.plane_game import GameState
import json
import copy
import math
def OnceGame(json_data,circle_location_list):
    
    game = GameState(json_data,second_problem=True,circle_location_list=circle_location_list)
    print(game.devices_amount_of_information)
    print(game.devices_location)
    brain = DDPG(game,json_data["device"]["num_max"]*3+7,json_data["device"]["num_max"]+2,json_data["plane"]["a_max"])
    
    while True:
        game.showMap()
        action = brain.select_action(game.state)
        
        state = copy.deepcopy(game.state)
        next_state,reward,terminal = game.frame_step(action)
        # 处理nextObservation中的图像
        if terminal == True:
            if np.sum(next_state[-json_data["device"]["num_max"]:])< 0.1:
                return True
            else: 
                return False
def CircleChange(json_data,circle_location_list, n):
    if OnceGame(json_data,circle_location_list):
        return circle_location_list
    else:
        while True:
            for i in range(n):
                x_old = circle_location_list[i*2]
                y_old = circle_location_list[i*2+1]
                x_new = circle_location_list[(i+1)%n*2]
                y_new = circle_location_list[(i+1)%n*2+1]
                circle_location_list[i*2] = (x_old + x_new)/2
                circle_location_list[i*2+1] = (y_old + y_new)/2
                if OnceGame(json_data,circle_location_list):
                    x_new = circle_location_list[i*2]
                    y_new = circle_location_list[i*2+1]
                    circle_location_list[i*2] = (x_old + x_new)/2
                    circle_location_list[i*2+1] = (y_old + y_new)/2
                    while True:
                        if (x_new - x_old)**2 + (y_new - y_old)**2 <= 0.01:
                            return circle_location_list
                        if OnceGame(json_data,circle_location_list):
                            x_new = circle_location_list[i*2]
                            y_new = circle_location_list[i*2+1]
                            circle_location_list[i*2] = (x_old + x_new)/2
                            circle_location_list[i*2+1] = (y_old + y_new)/2
                            
                        else:
                            x_old = circle_location_list[i*2]
                            y_old = circle_location_list[i*2+1]
                            circle_location_list[i*2] = (x_old + x_new)/2
                            circle_location_list[i*2+1] = (y_old + y_new)/2
def CircleInit(json_data):
    n = json_data["second_problem"]["num"]
    circle_location_list = np.zeros(2*n)
    circle_r = json_data["second_problem"]["r"]
    for index in range(json_data["second_problem"]["num"]):
        circle_location_list[index*2] = circle_r* math.sin(2*math.pi*index/n) /math.sin(math.pi/n)
        circle_location_list[index*2+1] = circle_r* math.cos(2*math.pi*index/n) /math.sin(math.pi/n)
        
    return circle_location_list,circle_r

if __name__ == "__main__":
    json_data = json.load(open("assets\data.json"))
    
    circle_location_list, circle_r = CircleInit(json_data)
    
    circle_location_list = CircleChange(json_data, circle_location_list, circle_r)
    
    np.savetxt("circle_location_list.txt", circle_location_list)
