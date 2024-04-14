# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:29:31 2024

@author: zahra
"""

import numpy as np
import random as rnd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from collections import deque

class KKW:
    def __init__(self,density):
        self.D0 = 15
        self.D1 = 2.55
        self.a = self.b = 1
        self.vp = 4
        self.pa1 = 0.7
        self.pa2 = 0.052
        self.p0 = 0.425
        self.pd = 0.04
        self.vmax = 5 # cells/timestep
        self.l = 1 #cells
        self.dt = 1 #s
        self.dx = 0.5 #m
        
        self.ncells = 300
        self.ntimesteps = 600
        self.density = density    #0.03 free flow, #0.1 jam
        
        self.nvehicles = int(self.ncells * self.density) 
        
        self.D = 0
        self.dacc = 0
        self.sgap = np.zeros((self.nvehicles,self.ntimesteps))
        self.vdes = 0

        self.pa = 0
        self.pb = 0
        
        self.global_flow_data = []
        self.local_flow_data = [] 
        self.local_density_data = [] 
        
        self.v = []
        for i in range(self.nvehicles):
            sublist = []
            for _ in range(self.ntimesteps):
                sublist.append([i, 0])
            self.v.append(sublist)       
    
        # random_numbers = [rnd.randint(0, self.vmax) for _ in range(self.nvehicles)]
        for j in range(self.nvehicles):
            self.v[j][0][1] = self.vmax
        
        self.x = []
        for i in range(self.nvehicles):
            sublist = []
            for _ in range(self.ntimesteps):
                sublist.append([i, 0])
            self.x.append(sublist)
     
        random_numbers = rnd.sample(range(1, self.ncells), self.nvehicles)
        random_numbers = sorted(random_numbers)
        for j in range(self.nvehicles):
           self.x[j][0][1] = random_numbers[j]
           
        # Initialize queue for vehicles waiting to enter the road
        self.queue = deque()
        
    def run(self):

        rnd.seed(42)
        total_speed = 0
        checkpoints = [50, 299] 
        for t in range(0, self.ntimesteps-1):
            
            if t % 45 == 0:#self.ncells/3/self.vmax
                total_vehicles1 = 0
                total_speed1 = 0
                total_vehicles2 = 0
                total_speed2 = 0
            xi = rnd.random()
            for i in range(self.nvehicles):
                order = self.x[i][t][0]
                if order == (self.nvehicles+1):
                    self.v[i][t+1][0] = self.nvehicles+1
                    self.v[i][t+1][1] = self.vmax
                    self.x[i][t+1][0] = self.nvehicles+1
                    self.x[i][t+1][1] = self.ncells
                else: 
                    leading_order = int(self.x[i][t][0]) + 1
                    leading_idx = -2
                    for idx in range(self.nvehicles):
                        if int(self.x[idx][t][0]) == leading_order:
                            leading_idx = idx
                            
                    if leading_idx == -2:
                        vleading = int(self.vmax)
                        self.sgap[i,t] = int(self.ncells)
                    else:
                        vleading = int(self.v[leading_idx][t][1])
                        self.sgap[i,t] = int(self.x[leading_idx][t][1]) - int(self.x[i][t][1]) - 1
                     
                    gap = int(self.sgap[i,t])
                            
                    if self.v[i][t][1] < vleading:
                        self.dacc = int(self.a)
                    if self.v[i][t][1] == vleading:
                        self.dacc = 0
                    if self.v[i][t][1] > vleading:
                        self.dacc = -1 * int(self.b)

                    
                    if gap < 0 and t<10:
                        print("gap",gap,
                            "time:",t,"id:",i,"leading",leading_idx,
                              "position",self.x[i][t][1] ,self.x[leading_idx][t][1],
                              "order:",self.x[i][t][0],self.x[leading_idx][t][0],
                              "pposition:",self.x[i][t-1][1],self.x[leading_idx][t-1][1],
                              "pspeed",self.v[i][t-1][1],self.v[leading_idx][t-1][1],
                              "pgap:",self.sgap[i,t-1],
                              "porder:",self.x[i][t-1][0],self.x[leading_idx][t-1][0],
                              )
                    self.D = self.D0 + self.D1 * self.v[i][t][1]
                    if gap > (self.D - self.l):
                        self.vdes = self.v[i][t][1] + self.a
                    if gap <= (self.D - self.l):
                        self.vdes = self.v[i][t][1] + self.dacc
                   
                    vfuture = max(0, min(self.vmax,gap,self.vdes))
                    
                    if vfuture < self.vp:
                        self.pa = self.pa1
                    if vfuture >= self.vp:
                        self.pa = self.pa2
                        
                    if vfuture == 0:
                        self.pb = self.p0
                    if vfuture > 0:
                        self.pb = self.pd
                    
    
                    if xi < self.pa:
                        eta = int(self.a)
                    elif self.pa <= xi and xi < self.pa + self.pb:
                       eta = -1 * int(self.b) 
                    elif xi >= (self.pa + self.pb):
                       eta = 0
                       
                    self.v[i][t+1][1] = max(0, min(self.vmax,self.v[i][t][1] + eta, self.v[i][t][1] + self.a, gap))
                    self.x[i][t+1][1] = int(self.x[i][t][1]) + int(self.v[i][t+1][1])
                    self.v[i][t+1][0] = int(self.v[i][t][0])
                    self.x[i][t+1][0] = int(self.x[i][t][0])
                    
                    xfuture = int(self.x[i][t+1][1])
                    
                    total_speed += self.v[i][t+1][1]
                    if xfuture > self.ncells:
                        # Add the vehicle that moved beyond the last position of the road to the queue
                        self.queue.append(i)
                        
                        self.v[i][t+1][0] = self.nvehicles+1
                        self.v[i][t+1][1] = self.vmax
                        self.x[i][t+1][0] = self.nvehicles+1
                        self.x[i][t+1][1] = self.ncells
                        
                  ####### WAY 1 ########## Just count moving vehicles
                # if self.x[i][t+1][1] >= checkpoints[1] and self.x[i][t][1] < checkpoints[1] :       
                #     total_vehicles1 += 1
                    #total_speed1 += (1/self.v[i][t+1][1])
                    
                    ####### WAY 2 ###### 
                if self.x[i][t+1][1] >= checkpoints[1]  and self.x[i][t+1][1] < (checkpoints[1] + 5) and self.x[i][t+1][1] > (self.x[i][t][1] + 0.1) :  
                    total_speed1 += self.v[i][t+1][1]
                    total_vehicles1 += 1
      
                    ####### WAY 1 ########## Just count moving vehicles
                # if self.x[i][t+1][1] >= checkpoints[0] and self.x[i][t][1] < checkpoints[0] :       
                #     total_vehicles2 += 1
                    #total_speed2 += (1/self.v[i][t+1][1])
                    
                    ####### WAY 2 ###### counts stopped vehicles too
                if self.x[i][t+1][1] >= checkpoints[0]  and self.x[i][t+1][1] < (checkpoints[0] + 5) :  
                    total_speed2 += self.v[i][t+1][1]
                    total_vehicles2 += 1
                                           
            

            if (t + 1)%45 == 0 and total_speed1> 0 and total_vehicles1>0 :
                ####### WAY 1 ##########
                # flow = total_vehicles1/45
                # meanspeed = total_vehicles1/total_speed1
                # density = flow / meanspeed

                ####### WAY 2 ######
                density = total_vehicles1 / 225
                flow = total_speed1 / 225

                self.local_flow_data.append(flow)
                self.local_density_data.append(density)

            if (t + 1)%45 == 0 and total_speed2> 0 and total_vehicles2>0 :
                ####### WAY 1 ##########
                # flow = total_vehicles2/45
                # meanspeed = total_vehicles2/total_speed2
                # density = flow / meanspeed
                
                ####### WAY 2 ######
                density = total_vehicles2/ 225
                flow = total_speed2 / 225
                
                self.local_flow_data.append(flow)
                self.local_density_data.append(density)
            
            if not any(row[1] == 1 for row in (row[t] for row in self.x)):  # Check if position 1 is free
                if self.queue:  # Check if there are vehicles in the queue
                    
                    # Dequeue the first vehicle in the queue
                    waiting_vehicle = self.queue.popleft()
                    #print("waiting_vehicle",waiting_vehicle)
                    for j in range(self.nvehicles):
                        if j not in self.queue :
                            self.x[j][t+1][0] = int(self.x[j][t+1][0]) + 1
                            self.v[j][t+1][0] = int(self.v[j][t+1][0]) + 1
                    self.x[waiting_vehicle][t+1][1] = 1
                    self.v[waiting_vehicle][t+1][1] = int(self.vmax)
                    self.x[waiting_vehicle][t+1][0] = 0
                    self.v[waiting_vehicle][t+1][0] = 0

        global_mean_speed = total_speed / (self.nvehicles * self.ntimesteps)
        global_flow = global_mean_speed * self.density
        print("flow",global_flow,"density",self.density)

                
        return
    
    def global_density(self):
        return
    
    def plot_all(self):
        time = np.arange(self.ntimesteps)
    
        # Extract positions from the nested list
        flattened_positions = [x[1] for sublist in self.x for x in sublist]
    
        # Flatten the speeds array
        flattened_speeds = [v[1] for sublist in self.v for v in sublist]
    
        # Create an array of time values corresponding to each position
        time_repeated = np.tile(time, len(self.x))
        
        # Normalize speeds to range [0, 1], considering vmax
        normalized_speeds =  [speed / self.vmax for speed in flattened_speeds]
    
        # Create a colormap ranging from black to light gray
        cmap = cm.get_cmap('Greys')
        reversed_cmap = cmap.reversed()
        colors_adjusted = [cmap((self.vmax - i) / self.vmax) for i in range(int(self.vmax))]
        reversed_cmap = cm.colors.ListedColormap(colors_adjusted[::1])
        # Map normalized speeds to colors from the colormap
        colors = reversed_cmap(normalized_speeds)
            
        # Plotting
        plt.scatter(time_repeated, flattened_positions, marker='.', color=colors, s=4)
       
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Position vs Time')
        plt.legend()
        plt.grid(True)
        plt.savefig("PositionVSTime.pdf", bbox_inches="tight")
        plt.show()

        
    def plot_one_by_one(self):
        
       
        # Assuming time values are evenly spaced, otherwise provide time values separately
        time = np.arange(self.ntimesteps)
        
        cmap = cm.get_cmap('Greys')
        colors_adjusted = [cmap((self.vmax - i) / self.vmax) for i in range(int(self.vmax))]
        reversed_cmap = cm.colors.ListedColormap(colors_adjusted[::1])

        
        # Plotting
        for i in range(self.nvehicles):
            normalized_speeds =  [self.v[i][t][1] / self.vmax for t in range(self.ntimesteps) ]
            #normalized_speeds = self.v[i] / self.vmax
            colors = reversed_cmap(normalized_speeds)
            
            plt.scatter(time, [x[1] for x in self.x[i]], marker='.', color=colors, s=4)

            plt.xlabel('Time')
            plt.ylabel('Position')
            plt.title('Position vs Time')
            plt.legend()
            plt.grid(True)
            plt.show()


            
    def plot_spaceGap(self):
                
         # Assuming time values are evenly spaced, otherwise provide time values separately
         time = np.arange(self.sgap.shape[1])
         
         # Plotting
         for i in range(self.sgap.shape[0]):

             plt.scatter(time, self.sgap[i], marker='.', s=5)

             plt.xlabel('Time')
             plt.ylabel('Position')
             plt.title('Position vs Time')
             plt.legend()
             plt.grid(True)
             plt.show()
             
    def plot_flow_vs_density(self, densities):
        flow_counts = [[] for _ in range(len(densities))]
        density_counts = [[] for _ in range(len(densities))]
        
        for idx, density in enumerate(densities):
            self.__init__(density)
            self.run()
            flow_counts[idx] = self.local_flow_data  # Assign flow data to the corresponding density index
            density_counts[idx] = self.local_density_data
        # self.__init__(0.3)
        # self.run()    
        # flow_counts = self.local_flow_data  # Assign flow data to the corresponding density index
        # density_counts = self.local_density_data
            
        # Plotting
        for idx, density in enumerate(densities):
            #plt.scatter([density] * len(flow_counts[idx]), flow_counts[idx], marker='.', s=4)
            plt.scatter(density_counts[idx], flow_counts[idx], marker='.', s=4)
            
        plt.xlabel('Density')
        plt.ylabel('Flow')
        plt.title('Flow vs Density')
        plt.grid(True)
        plt.show()

if __name__ == "__main__":
    densities = np.arange(0.01, 1, 0.1)
    kkw_instance = KKW(0.3)
    kkw_instance.run()
    kkw_instance.plot_all()
    kkw_instance.plot_flow_vs_density(densities)

