#!/usr/bin/env python
import pandas as pd
import numpy as np
import pickle
from numpy import einsum 
from numpy import linalg as LA
from numpy import sin, cos, pi


class Swimmer():
     
    def __init__(self, swimmer_type_id = 1, action_interval = 5.0, reward_gain = 0.2):
       
        with open(f'../model/swimmer_type{swimmer_type_id}.pickle','rb') as f:
            swimmer_type = pickle.load(f)
        #print(swimmer_type)
        with open(f'../model/param.pickle','rb') as f2:
            param = pickle.load(f2)
        #print(param) 
        self.numParticle = swimmer_type['numParticle']
        self.position = swimmer_type['pos']
        self.pos_init = swimmer_type['init_pos']
        self.vector = np.zeros(((self.numParticle,self.numParticle, 3)))
        self.displacement = np.zeros(3)
        self.K_star = swimmer_type['K_star']
        self.frec = swimmer_type['frec']
        self.radius = swimmer_type['radius']
        self.arm_len_init = swimmer_type['arm_length']
        self.arm_len = np.zeros(self.numParticle-1)
        
        self.L = np.sum(self.arm_len_init)
        
        self.DT = param['DT']
        self.OUTPUT_ITER = param['OUTPUT_ITER']
        self.MAX_TIME = param['MAX_TIME']
        self.MAX_STEP = self.MAX_TIME / action_interval       
        print(self.L)
    

        self.reward_gain = reward_gain
        self.step_counter = 0
        self.test_count = 0

        print(self.reward_gain)

    def get_vector(self,position, numParticle):
        self.vector_norm = np.empty((numParticle, numParticle))
        self.unit_vector = np.empty(((numParticle,numParticle, 3)))
        
        for i in range(numParticle):
            for j in range(numParticle):
                self.vector[i][j] = position[j] - position[i]

                self.vector_norm[i][j] = LA.norm(self.vector[i][j])

                if (self.vector_norm[i][j] == 0):
                    self.unit_vector[i][j] = 0
                else:
                    self.unit_vector[i][j] = self.vector[i][j] / self.vector_norm[i][j]

        for i in range(numParticle-1):
            self.arm_len[i] = self.vector_norm[i][i+1]
        #self.data_arm.append(c.arm_len.copy())
        #print('\033[46m' + "arm length : " + '\033[0m' , c.arm_len)
        #print(self.vector)
        #print(f"vector_norm{self.vector_norm}")
        #print(f"vector_init{self.unit_vector}")
        
                        
    def get_angle(self, u: np.ndarray, v : np.ndarray):
        i = np.inner(u, v)
        n = LA.norm(u) * LA.norm(v)
        c = i /n 
        z = np.cross(u,v)
        if z[2] > 0:
            return np.arccos(np.clip(c, -1.0, 1.0))
        elif z[2] < 0:   
            return 2*pi - np.arccos(np.clip(c, -1.0, 1.0))
                   
        
    def get_argument(self, u: np.ndarray, v: np.ndarray):
        angle1 = math.atan2(u[1], u[0])
        angle2 = math.atan2(v[1], v[0])
        return  angle1 -angle2


    def get_force_active(self,numParticle , torque: np.ndarray):
        
        self.force_active = np.zeros((numParticle,3))
        memo_force = np.zeros((3,3))
        for i in range(1,numParticle-1):
            if torque[i] > 0:
                tmp1 =  abs(torque[i]) * self.unit_vector[i][i-1][0]/self.vector_norm[i][i-1] #vector x position
                tmp2 = -abs(torque[i]) * self.unit_vector[i][i-1][1]/self.vector_norm[i][i-1] #vector y position
               
                memo_force[0][0] = tmp2
                memo_force[0][1] = tmp1
                

                self.force_active[i-1][0] += tmp2 #force x
                self.force_active[i-1][1] += tmp1 #force y
                                
                tmp1 = -abs(torque[i]) * self.unit_vector[i][i+1][0]/self.vector_norm[i][i+1] #vector x position
                tmp2 =  abs(torque[i]) * self.unit_vector[i][i+1][1]/self.vector_norm[i][i+1] #vector y position
                self.force_active[i+1][0] += tmp2 #force x
                self.force_active[i+1][1] += tmp1 #force y
                
                memo_force[2][0] = tmp2
                memo_force[2][1] = tmp1
                
                self.force_active[i] += - memo_force[0] - memo_force[2]
            
            elif torque[i] < 0:
                tmp1 = -abs(torque[i]) * self.unit_vector[i][i-1][0]/self.vector_norm[i][i-1] #vector x position
                tmp2 =  abs(torque[i]) * self.unit_vector[i][i-1][1]/self.vector_norm[i][i-1] #vector y position
                self.force_active[i-1][0] += tmp2 #force x
                self.force_active[i-1][1] += tmp1 #force y
                        
                memo_force[0][0] = tmp2
                memo_force[0][1] = tmp1
                
                tmp1 =  abs(torque[i]) * self.unit_vector[i][i+1][0]/self.vector_norm[i][i+1] #vector x position
                tmp2 = -abs(torque[i]) * self.unit_vector[i][i+1][1]/self.vector_norm[i][i+1] #vector y position
                self.force_active[i+1][0] += tmp2 #force x
                self.force_active[i+1][1] += tmp1 #force y

                
                memo_force[2][0] = tmp2
                memo_force[2][1] = tmp1

            
                self.force_active[i] += -memo_force[0] - memo_force[2]
            else:
                self.force_active[i-1] += 0
                self.force_active[i+1] += 0
                self.force_active[i]   += 0


        #print("check force_active free :", np.sum(self.force_active,axis = 0))
        #print(f"force active: {self.force_active}")

    def get_force_passive(self,numParticle):
        self.force_passive = np.zeros((numParticle,3))

        for i in range(numParticle-1):
            self.force_passive[i]     += self.K_star*(self.vector_norm[i][i+1]/self.L - self.arm_len_init[i]/self.L)*self.unit_vector[i][i+1]
            self.force_passive[i+1]   += self.K_star*(self.vector_norm[i][i+1]/self.L - self.arm_len_init[i]/self.L)*self.unit_vector[i+1][i]

        #print(f"force passive : {self.force_passive}")

    def get_total_force(self, numParticle, torque):
        self.get_force_active(numParticle,torque)      
        self.get_force_passive(numParticle)
        self.force_total = np.zeros((numParticle,3))
        self.force_total = self.force_active + self.force_passive
        #print('\033[46m' + "force total :" + '\033[0m' + '\n', self.force_total)
        #print('\033[46m' + "check force free :" + '\033[0m',np.sum(self.force_total, axis = 0))


    def calculate_tensor(self, numParticle):
        
        self.ossen_tensor = np.zeros((((numParticle,numParticle,3,3))))
        
        for i in range(numParticle):
            for j in range(numParticle):
                if(self.vector_norm[i][j] == 0):
                    continue
                reshape_r = self.vector[i][j].reshape(3,1)

                tmp = np.eye(3) + np.einsum('i,j -> ij', self.vector[i][j],self.vector[i][j])/(self.vector_norm[i][j]**2)
        
                self.ossen_tensor[i][j] = tmp / (8*pi*self.vector_norm[i][j])


    def update(self, numParticle, force):
        self.v = np.zeros((numParticle,3))
        box = np.zeros((numParticle, 3))

        for i in range(numParticle):
            for j in range(numParticle):
                if i == j:
                    box[j] = force[j] /(6 * pi *self.radius)
                else:
                    tmp = np.dot(self.ossen_tensor[i][j], force[j].T)
                    box[j] = tmp.T
                self.v[i] = np.sum(box, axis= 0)
        
        #print(f"prev position :\n {self.position}")
        self.position += self.v * self.DT / self.frec
        #print(f"update positon :\n {self.position}")

    def check_displacement(self, position):
        centroid = np.sum(position,axis = 0)
        self.displacement = np.concatenate([self.displacement, centroid], axis = 0)
        

    def angle_data(self,angle1,angle2):
        
        self.dataset.append(angle1)
        self.dataset.append(angle2)
 
    def getObservation(self):
        observation = self.position - self.center_position
        return observation
    """
    def reset(self):
        self.position = self.position_init
        self.center_position = Get_center_position()
        self.prev_center_position = self.center_position
        return self.getObservation()
    """
    def reset(self):
        self.step_counter = 0
        return self.pos_init
    

    def Get_center_position(self):
        return np.sum(self.position ,axis = 0)
    
    
    def step(self, action : np.ndarray):
        done = False
        info = {}
    
        self.input_action = action
        self.center_position = self.Get_center_position()
        for i in range(200):
            #self.one_step(self.input_action)
            self.one_step(self.position,action)
            #self.ministep()
        self.prev_center_position = self.Get_center_position()
        reward = self.reward_gain*LA.norm(self.center_position - self.prev_center_position)

        """
        if self.step_counter % 100 == 0: 
            print(f'reward {self.step_counter}  count {self.test_count} : {reward} ') 
        """
        self.step_counter += 1
       
        if self.step_counter >= 200:
            done = True
            self.test_count += 1
        return self.getObservation(), reward, done, info 
    
    def one_step(self,position,action):
        self.get_vector(position,self.numParticle)
        self.get_total_force(self.numParticle, action)
        self.calculate_tensor(self.numParticle)
        self.update(self.numParticle, self.force_total)


    def output(self,position : np.ndarray, numParticle, path):
        data = position.reshape([1,numParticle*3])
        path = './test.csv'
        dir_exist_flag = os.path.isfile(path)
        print(dir_exist_flag)
        if dir_exist_flag:
            with open(path, 'a') as f:
                np.savetxt(f,data) 
        else:
            np.savetxt(path, data, delimiter=',', fmt='%.6f' ) 

 
if __name__ == '__main__':
    swimmer = Swimmer()
    action = [1,1 ,1,1]
 
