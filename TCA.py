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
    def __init__(self):
        self.D0 = 15
        self.D1 = 2.55
        self.a = self.b = 1
        self.vp = 7
        self.pa1 = 0.2
        self.pa2 = 0.052
        self.p0 = 0.425
        self.pd = 0.04
        self.vmax = 5 # cells/timestep
        self.l = 1 #cells
        self.dt = 1 #s
        self.dx = 0.5 #m
        
        self.ncells = 300
        self.ntimesteps = 580
        self.density = 0.4
        
        self.nvehicles = int(self.ncells * self.density) 
        
        self.D = 0
        self.dacc = 0
        self.sgap = np.zeros((self.nvehicles,self.ntimesteps))
        self.vdes = 0

        self.pa = 0
        self.pb = 0
        
        self.v = []
        for i in range(self.nvehicles):
            sublist = []
            for _ in range(self.ntimesteps):
                sublist.append([i, 0])
            self.v.append(sublist)       
    
        # random_numbers = [rnd.randint(0, self.vmax) for _ in range(self.nvehicles)]
        for j in range(self.nvehicles):
            self.v[j][0][1] = self.vmax
        
        #self.x = np.zeros((self.nvehicles,self.ntimesteps))
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

        for t in range(1, self.ntimesteps):
            xi = rnd.random()
            for i in range(self.nvehicles):
                b = self.x[i][t][0]
                if self.x[i][t-1][0] == (self.nvehicles+1):
                    self.v[i][t][0] = self.nvehicles+1
                    self.v[i][t][1] = self.vmax
                    self.x[i][t][0] = self.nvehicles+1
                    self.x[i][t][1] = self.ncells
                else: 
                    j = self.v[i][t-1][0] + 1
                    leading_idx = -2
                    for idx in range(self.nvehicles):
                        if self.v[idx][t-1][0] == j:
                            leading_idx = idx
                    if leading_idx == -2:
                        vleading = self.vmax
                    else:
                        vleading = self.v[leading_idx][t-1][1]
                     
                            
                    if self.v[i][t-1][1] < vleading:
                        self.dacc = int(self.a)
                    if self.v[i][t-1][1] == vleading:
                        self.dacc = 0
                    if self.v[i][t-1][1] > vleading:
                        self.dacc = -1 * int(self.b)
                        
                    j = self.x[i][t-1][0] + 1
                    leading_idx = -2
                    for idx in range(self.nvehicles):
                        if self.x[idx][t-1][0] == j:
                            leading_idx = idx
                    if leading_idx == -2:
                        self.sgap[i,t-1] = self.vmax
                        print("i",i)
                        print("j",j)
                    else:
                        self.sgap[i,t-1] = self.x[leading_idx][t-1][1] - self.x[i][t-1][1] - 1
                        
                    gap = self.sgap[i,t-1]
                    if gap < 0 and t<10:
                        print("gap",gap,
                            "time:",t,"id:",i,"leading",leading_idx,
                              "position",self.x[i][t-1][1] ,self.x[leading_idx][t-1][1],
                              "order:",self.x[i][t-1][0],self.x[leading_idx][t-1][0],
                              "position:",self.x[i][t-2][1],self.x[leading_idx][t-2][1],
                              "speed",self.v[i][t-2][1],self.v[leading_idx][t-2][1],
                              "pgap:",self.sgap[i,t-2],
                              "porder:",self.x[i][t-2][0],self.x[leading_idx][t-2][0],
                              )
                    self.D = self.D0 + self.D1 * self.v[i][t-1][1]
                    if self.sgap[i,t-1] > (self.D - self.l):
                        self.vdes = self.v[i][t-1][1] + self.a
                    if self.sgap[i,t-1] <= (self.D - self.l):
                        self.vdes = self.v[i][t-1][1] + self.dacc
                   
                    self.v[i][t][1] = max(0, min(self.vmax,self.sgap[i,t-1],self.vdes))
                    
                    if self.v[i][t][1] < self.vp:
                        self.pa = self.pa1
                    if self.v[i][t][1] >= self.vp:
                        self.pa = self.pa2
                        
                    if self.v[i][t][1] == 0:
                        self.pb = self.p0
                    if self.v[i][t][1] > 0:
                        self.pb = self.pd
                    
    
                    if xi < self.pa:
                        eta = self.a
                    elif self.pa <= xi and xi < self.pa + self.pb:
                       eta = -self.b 
                    elif xi >= (self.pa + self.pb):
                       eta = 0
                       
                    self.v[i][t][1] = max(0, min(self.vmax,self.v[i][t-1][1] + eta, self.v[i][t-1][1] + self.a, int(self.sgap[i,t-1])))
                    self.x[i][t][1] = self.x[i][t-1][1] + self.v[i][t][1]
                    
                    if self.x[i][t][1] > self.ncells:
                        # Add the vehicle that moved beyond the last position of the road to the queue
                        self.queue.append(i)
                        print("time",t,"i",i)
                        self.v[i][t][0] = self.nvehicles+1
                        self.v[i][t][1] = self.vmax
                        self.x[i][t][0] = self.nvehicles+1
                        self.x[i][t][1] = self.ncells
                    # Remove the vehicle 
                    # self.x = np.delete(self.x, i, axis=0)
                    # self.v = np.delete(self.v, i, axis=0)
                    # self.sgap = np.delete(self.sgap, i, axis=0)
                        
                    
                
            if not any(row[1] == 1 for row in (row[t] for row in self.x)):  # Check if position 1 is free
                if self.queue:  # Check if there are vehicles in the queue
                    
                    # Dequeue the first vehicle in the queue
                    waiting_vehicle = self.queue.popleft()
                    print("waiting_vehicle",waiting_vehicle)
                    # Insert the waiting vehicle at the beginning of x
                    # self.x = np.insert(self.x, 0, np.zeros(self.ntimesteps), axis=0)
                    # self.v = np.insert(self.v, 0, np.zeros(self.ntimesteps), axis=0)
                    # self.sgap = np.insert(self.sgap, 0, np.zeros(self.ntimesteps), axis=0)
                    # Place the vehicle at position 1 with random speed between 0 and 
                    #idx_list = [row[0] for row in (row[t] for row in self.x)]
                    for j in range(self.nvehicles):
                        if j not in self.queue :
                            self.x[j][t+1][0] = int(self.x[j][t][0]) + 1
                            self.v[j][t+1][0] = int(self.v[j][t][0]) + 1
                    self.x[waiting_vehicle][t][1] = 1
                    self.v[waiting_vehicle][t][1] = int(self.vmax)
                    self.x[waiting_vehicle][t+1][0] = 0
                    self.v[waiting_vehicle][t+1][0] = 0



                
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
            print(self.x[i][100][1])
            print(self.x[i][101][1])
            print(self.v[i][100][1])
            print(self.v[i][101][1])
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

if __name__ == "__main__":
    kkw_instance = KKW()
    kkw_instance.run()
    kkw_instance.plot_all()
