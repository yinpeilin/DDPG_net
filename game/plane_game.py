import json
import numpy as np
import sys
sys.path.append("./")
import random
import math
from assets.GetDataRandom import GetDeviceLocationRandom,GetDeviceInformationAmountRandom
import matplotlib.pyplot as plt


plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


class GameState:
    def __init__(self, json_data,second_problem = False,second_problem_circle_location = None):
        
        self.json_data = json_data
        
        json_data = GetDeviceLocationRandom(json_data)
        json_data = GetDeviceInformationAmountRandom(json_data)
        #能量
        self.plane_power = json_data['plane']['power']
        #通道数
        self.plane_k = json_data["plane"]['k']
        #运动
        self.plane_p_x = json_data['plane']['p_x']
        self.plane_p_y = json_data['plane']['p_y']
        self.plane_p_h = json_data['plane']['h']
        self.plane_v_x = json_data['plane']['v_x']
        self.plane_v_y = json_data['plane']['v_y']
        self.plane_a_x = 0.0
        self.plane_a_y = 0.0
        
        self.plane_v_min = json_data['plane']['v_min']
        self.plane_v_max = json_data['plane']['v_max']
        self.plane_a_limit = json_data['plane']['a_max']

        
        #信息相关
        self.plane_B_0 = json_data['plane']['B_0']
        self.plane_B_1 = json_data['plane']['B_1']
        #信噪比
        self.plane_rou_0 = json_data['rou_0']
        self.plane_rou_1 = json_data['rou_1']
        
        #信道选择参数
        self.channel_a = json_data['information']['a']
        self.channel_b = json_data['information']['b']
        
        #装置状态
        self.devices_count = np.array(json_data["device"]["num_max"])
        if second_problem:
            self.devices_location = second_problem_circle_location
            self.devices_amount_of_information = np.concatenate(self.devices_location,np.zeros((self.devices_count-json_data["second_problem"]["num"])*2))
        else:
            self.devices_location = np.array(json_data["device"]["location"]).flatten()
        if second_problem: 
            self.devices_amount_of_information = np.array(json_data["device"]["amount_of_information"],dtype=np.float32)[0:json_data["second_problem"]["num"]]
            self.devices_amount_of_information = np.concatenate(self.devices_amount_of_information,np.zeros(self.devices_count-json_data["second_problem"]["num"]))
            
        else:
            self.devices_amount_of_information = np.array(json_data["device"]["amount_of_information"],dtype=np.float32)
        #其他
        self.t_min = json_data["t_min"]
        self.plane_c1 = json_data['plane']['c1']
        self.plane_c2 = json_data['plane']['c2']
        #重力加速度
        self.g = 9.8
        
        self.state = np.array([self.plane_power,self.plane_p_x, self.plane_p_y, self.plane_v_x, self.plane_v_y, self.plane_a_x, self.plane_a_y])
        self.state =np.concatenate((self.state,self.devices_location,self.devices_amount_of_information),axis=0)

        
        # 作图
        self.plane_location_list = []
        
    def frame_step(self, input_actions):
        # input_actions
        # [delta_a_x , delta_a_y, choose_channel]
        reward = 0.0
        terminal = False
        
        
        
        #计算传输信息量，device信息改变
        data_transfer_all = 0.0
        choose_device_index = np.argsort(input_actions[2:])[self.devices_count-self.plane_k:] #choose the device which the power is maximum. 计算传输率和传输率改变率
        
        for device_index in choose_device_index:
            distance = math.sqrt((self.devices_location[device_index*2]-self.plane_p_x)**2+(self.devices_location[device_index*2+1]-self.plane_p_y)**2+self.plane_p_h**2)
            data_transfer = 0.0
            p_los = 1.0/(1+self.channel_a*math.exp(-self.channel_b*(math.log(self.plane_p_h/distance)-self.channel_a)))
            if random.uniform(0, 1) < p_los:
                data_transfer = self.plane_B_0*math.log2(1+self.plane_rou_0/(distance**2)) * self.t_min
            else:
                data_transfer = self.plane_B_1*math.log2(1+self.plane_rou_1/(distance**2)) * self.t_min
            if self.devices_amount_of_information[device_index] < data_transfer:
                data_transfer = self.devices_amount_of_information[device_index]
                self.devices_amount_of_information[device_index] = 0.0
            else:
                self.devices_amount_of_information[device_index] -= data_transfer
            data_transfer_all += data_transfer 
            
        
        
        
        #计算能量损耗
        
        v_all = math.sqrt(self.plane_v_x**2+self.plane_v_y**2)
        a_all = math.sqrt(self.plane_a_x**2+self.plane_a_y**2)
        r = (v_all **2 )/(a_all +0.000001)
        plane_power_loss = self.t_min*((self.plane_c1+self.plane_c2/(self.g*self.g*r*r))*math.pow(v_all,3)+self.plane_c2/v_all)
        
        self.plane_power -= plane_power_loss 
        
        # 获得reward
        reward -= data_transfer_all*100000/plane_power_loss
        
        #根据当前的值计算位置和速度变化
        
        self.plane_p_x += self.plane_v_x*self.t_min
        self.plane_p_y += self.plane_v_y*self.t_min
        
        if self.plane_p_x < self.json_data["device"]["location_x_limit"][0] or self.plane_p_x > self.json_data["device"]["location_x_limit"][1] \
            or self.plane_p_y < self.json_data["device"]["location_y_limit"][0] or self.plane_p_y > self.json_data["device"]["location_y_limit"][1]:
            # self.plane_v_x -= self.plane_a_x*self.t_min
            # self.plane_v_y -= self.plane_a_y*self.t_min
            reward -= 10
        
        self.plane_v_x += self.plane_a_x*self.t_min
        self.plane_v_y += self.plane_a_y*self.t_min
        if self.plane_v_x**2 + self.plane_v_y**2 > self.plane_v_max**2 or self.plane_v_x**2 + self.plane_v_y**2 < self.plane_v_min**2:
            # self.plane_v_x -= self.plane_a_x*self.t_min
            # self.plane_v_y -= self.plane_a_y*self.t_min
            reward -= 10
            
        
        self.plane_a_x += input_actions[0]*self.plane_a_limit*self.t_min
        self.plane_a_y += input_actions[1]*self.plane_a_limit*self.t_min
        if self.plane_a_x**2 + self.plane_a_y**2  > self.plane_a_limit**2:
            self.plane_a_x = self.plane_a_x/math.sqrt(self.plane_a_x**2 + self.plane_a_y**2)*self.plane_a_limit
            self.plane_a_y = self.plane_a_y/math.sqrt(self.plane_a_x**2 + self.plane_a_y**2)*self.plane_a_limit# limit acceleration to limit value
            reward -= 10
        # print(reward,self.plane_a_x,self.plane_a_y)
        
        #得到new_state
        
        
        if self.isover():
            #SOUNDS['hit'].play()
            #SOUNDS['die'].play()
            terminal = True
            # reward = np.sum(self.devices_amount_of_information != 0)*0.1
            self.__init__(json_data=self.json_data)
    
        self.state = np.array([self.plane_power,self.plane_p_x, self.plane_p_y, self.plane_v_x, self.plane_v_y, self.plane_a_x, self.plane_a_y])
        self.state =np.concatenate((self.state, self.devices_location, self.devices_amount_of_information),axis=0)
        return self.state, reward, terminal

    def isover(self):
        if self.plane_power <= 0.0:
            return True
        return False
    
    def showMap(self):
        
        self.plane_location_list.extend([self.plane_p_x,self.plane_p_y])
        
        
        plt.clf()
        
        
        # plt.scatter(self.devices_location[::2],self.devices_location[1::2])
        for i in range(self.devices_count):
            if self.devices_amount_of_information[i]>0:
                grey_color = self.devices_amount_of_information[i]/self.json_data["device"]["amount_of_information_limit"][1]*0.5
                p2 = plt.scatter([self.devices_location[i*2]], [self.devices_location[i * 2+1]],color = (grey_color,grey_color,grey_color))
            else:
                p3 = plt.scatter([self.devices_location[i*2]], [self.devices_location[i * 2+1]],color = 'blue')
            # self.ax.scatter(self.devices_location[i*2], self.devices_location[i * 2+1],color = ,marker='o')
            
        
        p1 = plt.scatter([self.plane_p_x],[self.plane_p_y],color='red',label = "无人机")
        plt.plot(self.plane_location_list[::2],self.plane_location_list[1::2],color='blue')
        try:
            plt.legend([p1,p2,p3],["无人机","物联网设备","完成采集的物联网设备"],loc = 'upper right')
        except:
            plt.legend([p1],["无人机"],loc = 'upper right')
        plt.grid(True)
        # plt.show()
        plt.pause(0.001)

if __name__ == "__main__":
    GAME = 'plane'  
    json_data = json.load(open("assets/data.json"))# the name of the game being played for log files
    plane_game = GameState(json_data)
    action0 = np.array([2*random.random()-1,2*random.random()-1])
    action0 = np.concatenate((action0, np.ones(5)),axis=0)
    action0 = np.concatenate((action0, np.zeros(95)),axis=0)
    observation0, reward0, terminal = plane_game.frame_step(action0)
    while True:
    # 动作只有两种，分别用【1，0】和【0，1】表示
    
        plane_game.showMap()
        action0 = np.array([2*random.random()-1,2*random.random()-1])
        action0 = np.concatenate((action0, np.ones(5)),axis=0)
        action0 = np.concatenate((action0, np.zeros(95)),axis=0)
        observation0, reward0, terminal = plane_game.frame_step(action0)
        # print(observation0)