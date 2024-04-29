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
from collections import Counter
import math
import TrafficGraph as TG

class KKW:
    def __init__(self,density,init_mode):
        self.D0 = 2
        self.D1 = 2
        self.a = self.b = 1
        self.vp = 4
        self.pa1 = 0.8
        self.pa2 = 0.15
        self.p0 = 0.425
        self.pd = 0.04
        self.vmax = 5 # cells/timestep
        self.l = 1 #cells
        self.dt = 1 #s
        self.dx = 0.5 #m
        
        self.ncells = 300
        self.ntimesteps = 401
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
        self.local_phase_data = []
        
        self.vehicles_phase = np.ones((self.nvehicles,self.ntimesteps))
                   
        # Initialize queue for vehicles waiting to enter the road
        self.queue = deque()
        
        self.v = []
        for i in range(self.nvehicles):
            sublist = []
            for _ in range(self.ntimesteps):
                sublist.append([i, 0])
            self.v.append(sublist)       
            
        self.x = []
        for i in range(self.nvehicles):
            sublist = []
            for _ in range(self.ntimesteps):
                sublist.append([i, 0])
            self.x.append(sublist)
    
        # random_numbers = [rnd.randint(0, self.vmax) for _ in range(self.nvehicles)]
        for j in range(self.nvehicles):
            self.v[j][0][1] = self.vmax 
            
        if init_mode == 'random':     
            if self.ncells == self.nvehicles:
                for j in range(self.nvehicles):
                   self.x[j][0][1] = j
            else:
                random_numbers = rnd.sample(range(1, self.ncells), self.nvehicles)
                random_numbers = sorted(random_numbers)
                for j in range(self.nvehicles):
                   self.x[j][0][1] = random_numbers[j]
        elif init_mode == 'periodic':
            self.x[0][0][1] = 1
            for j in range(1,self.nvehicles):
                self.x[j][0][1] = self.x[j-1][0][1] + self.vmax + 1
                if self.x[j][0][1] > self.ncells:
                    self.queue.append(j)
                    self.v[j][0][0] = self.nvehicles+1
                    self.v[j][0][1] = self.vmax
                    self.x[j][0][0] = self.nvehicles+1
                    self.x[j][0][1] = self.ncells

    
    def most_common_element(self,lst):
        counts = Counter(lst)
        max_count = max(counts.values())
        return [num for num, count in counts.items() if count == max_count]
        
    
    def run(self,LMeasureFormula):

        rnd.seed(42)
        total_speed = 0
        checkpoints = [50, 200, 255]
        phase = [1,2,3] #1: free, 2: synchronized 3:moving jam
        Tmp = 20 #measurement period
        K1d = 5  #segment length
        for t in range(0, self.ntimesteps-1):
            
            if t % Tmp == 0:
                total_vehicles1 = 0
                total_vehicles2 = 0
                total_vehicles3 = 0
                total_speed1 = 0
                total_speed2 = 0
                total_speed3 = 0
                mean_phase1 = []
                mean_phase2 = []
                mean_phase3 = []
            xi = rnd.random()
            for i in range(self.nvehicles):
                order = self.x[i][t][0]
                if order == (self.nvehicles+1):
                    self.v[i][t+1][0] = self.nvehicles+1
                    self.v[i][t+1][1] = self.vmax
                    self.x[i][t+1][0] = self.nvehicles+1
                    self.x[i][t+1][1] = self.ncells
                    self.vehicles_phase[i,t+1] = phase[0]
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

                    
                    if gap < 0 and t < 10:   
                        print("Error: Negative Gap!")
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
                    min_value = min(self.vmax,gap,self.vdes)
                    if min_value == self.vmax:
                        self.vehicles_phase[i,t+1] = phase[0]
                    elif min_value == gap:
                        self.vehicles_phase[i,t+1] = phase[2]
                    else:
                        self.vehicles_phase[i,t+1] = phase[1]
                    
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
                    if t < 10:  # For computing global flow
                        total_speed += self.v[i][t+1][1]
                    if xfuture > self.ncells:
                        # Add the vehicle that moved beyond the last position of the road to the queue
                        self.queue.append(i)
                        
                        self.v[i][t+1][0] = self.nvehicles+1
                        self.v[i][t+1][1] = self.vmax
                        self.x[i][t+1][0] = self.nvehicles+1
                        self.x[i][t+1][1] = self.ncells
                        
                ######### Local Measurements ############
                #detector1
                if LMeasureFormula == 1: # Segment detector counts stopped vehicles too
                    if self.x[i][t+1][1] >= checkpoints[0]  and self.x[i][t+1][1] < (checkpoints[0] + 5):
                        total_speed1 += self.v[i][t+1][1]
                        total_vehicles1 += 1
                        mean_phase1.append(self.vehicles_phase[i,t+1]) 
                        
                if LMeasureFormula == 2: # unit length detector just counts moving vehicles       
                    if self.x[i][t+1][1] >= checkpoints[0] and self.x[i][t][1] < checkpoints[0] :       
                        total_vehicles1 += 1
                        total_speed1 += (1/self.v[i][t+1][1])
                        mean_phase1.append(self.vehicles_phase[i,t+1]) 
                        
                #detector2
                if LMeasureFormula == 1: # Segment detector counts stopped vehicles too
                    if self.x[i][t+1][1] >= checkpoints[1]  and self.x[i][t+1][1] < (checkpoints[1] + 5):  
                        total_speed2 += self.v[i][t+1][1]
                        total_vehicles2 += 1
                        mean_phase2.append(self.vehicles_phase[i,t+1])
                        
                if LMeasureFormula == 2: # unit length detector just counts moving vehicles
                    if self.x[i][t+1][1] >= checkpoints[1] and self.x[i][t][1] < checkpoints[1] :       
                        total_vehicles2 += 1
                        total_speed2 += (1/self.v[i][t+1][1])
                        mean_phase2.append(self.vehicles_phase[i,t+1])

                #detector3
                if LMeasureFormula == 1: # Segment detector counts stopped vehicles too   
                    if self.x[i][t+1][1] >= checkpoints[2]  and self.x[i][t+1][1] < (checkpoints[2] + 5) :   #and self.x[i][t+1][1] > (self.x[i][t][1] + 0.1) :
                        total_speed3 += self.v[i][t+1][1]
                        total_vehicles3 += 1
                        mean_phase3.append(self.vehicles_phase[i,t+1])
                        
                if LMeasureFormula == 2: # unit length detector just counts moving vehicles
                    if self.x[i][t+1][1] >= checkpoints[2] and self.x[i][t][1] < checkpoints[2] :       
                        total_vehicles3 += 1
                        total_speed3 += (1/self.v[i][t+1][1])
                        mean_phase3.append(self.vehicles_phase[i,t+1])
                                         
            
            ## Flow and Density Computation for different detectors after Tmp
            if (t + 1) % Tmp == 0 and total_speed1> 0 and total_vehicles1>0 :                
                if LMeasureFormula == 1:
                    local_density = total_vehicles1 / (Tmp*K1d)
                    flow = total_speed1 / (Tmp*K1d)
                    
                if LMeasureFormula == 2:
                    flow = total_vehicles1/Tmp
                    meanspeed = total_vehicles1/total_speed1
                    local_density = flow / meanspeed
                
                #mean_phase =  round(mean_phase1 / total_vehicles1)
                mean_phase = self.most_common_element(mean_phase1)

                self.local_flow_data.append(flow)
                self.local_density_data.append(local_density)
                self.local_phase_data.append(mean_phase[0])

            if (t + 1) % Tmp == 0 and total_speed2> 0 and total_vehicles2>0 :              
                if LMeasureFormula == 1:
                    local_density = total_vehicles2/ (Tmp*K1d)
                    flow = total_speed2 / (Tmp*K1d)
                    
                if LMeasureFormula == 2:
                    flow = total_vehicles2/Tmp
                    meanspeed = total_vehicles2/total_speed2
                    local_density = flow / meanspeed
                
                #mean_phase =  round(mean_phase2 / total_vehicles2)
                mean_phase = self.most_common_element(mean_phase2)
              
                self.local_flow_data.append(flow)
                self.local_density_data.append(local_density)
                self.local_phase_data.append(mean_phase[0])
                
            if (t + 1) % Tmp == 0 and total_speed3> 0 and total_vehicles3>0 :               
                if LMeasureFormula == 1:
                    local_density = total_vehicles3/ (Tmp*K1d)
                    flow = total_speed3 / (Tmp*K1d)
                    
                if LMeasureFormula == 2:
                    flow = total_vehicles3/Tmp
                    meanspeed = total_vehicles3/total_speed3
                    local_density = flow / meanspeed
                
                #mean_phase =  round(mean_phase3 / total_vehicles3)
                mean_phase = self.most_common_element(mean_phase3)
                
                self.local_flow_data.append(flow)
                self.local_density_data.append(local_density)
                self.local_phase_data.append(mean_phase[0])
            #############################################
            
            #Entering vehicle from queue to road
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

        #global_mean_speed = total_speed / (self.nvehicles * self.ntimesteps)
        #global_flow = global_mean_speed * self.density
        global_flow = total_speed/(self.ncells*10)
        self.global_flow_data.append(global_flow)
        print("flow",global_flow,"density",self.density)

                
        return
    
    def global_density(self):
        return

    def plot_position_vs_time(self):
         
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
        
        
    def plot_position_vs_time_colored(self):
         
        time = np.arange(self.ntimesteps)
        colors = np.ndarray.flatten(self.vehicles_phase)
    
        # Extract positions from the nested list
        flattened_positions = [x[1] for sublist in self.x for x in sublist]
      
        # Create an array of time values corresponding to each position
        time_repeated = np.tile(time, len(self.x))
        
        # Map phase integers to colors
        color_mapping = {1: 'blue', 2: 'green', 3: 'red'}
        phase_colors = [color_mapping[phase] for phase in colors]
        
        # Use the mapped colors in the scatter plot
        plt.scatter(time_repeated, flattened_positions, marker='.', c=phase_colors, s=4)
        
       
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title('Position vs Time (Colored by Phase)')
        plt.grid(True)
        plt.savefig("PositionVSTime_colored.pdf", bbox_inches="tight")
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
             
    def plot_flow_vs_density(self, densities, LMeasureFormula, init_mode):
        flow_counts = [[] for _ in range(len(densities))]
        density_counts = [[] for _ in range(len(densities))]
        phase_colors = {1: 'blue', 2: 'green', 3: 'red'}  # Define colors for each phase
        phase_markers = {1: 'o', 2: 's', 3: '^'}  # Define marker styles for each phase
        global_flow = []
        
        for idx, density in enumerate(densities):
            self.__init__(density, init_mode)
            self.run(LMeasureFormula)
            flow_counts[idx] = self.local_flow_data
            density_counts[idx] = self.local_density_data
            phases = self.local_phase_data  # Fetch phase data
            global_flow.append(self.global_flow_data)
            
            for i in range(len(phases)):
                # Plot each point with corresponding shape and color based on phase
                plt.scatter(density_counts[idx][i], flow_counts[idx][i], marker=phase_markers[int(phases[i])], color=phase_colors[int(phases[i])], s=5)
        
        plt.plot(densities, global_flow, color='black', label='Global Flow')
                      
        plt.xlabel('Density')
        plt.ylabel('Flow')
        plt.title('Flow vs Density')
        plt.grid(True)
        plt.show()
        
    def create_graph(self):
        TG.generate_graph(self.x,self.v,self.vehicles_phase,self.ncells)
        

if __name__ == "__main__":
    densities = np.arange(0.01, 0.4, 0.01)
    LMeasureFormula = 1 #1: segment detectors, 2:unit length
    init_mode = 'random'
    # kkw_instance = KKW(0.18,init_mode)
    # kkw_instance.plot_flow_vs_density(densities, LMeasureFormula, init_mode)
    # kkw_instance.run(LMeasureFormula)
    # kkw_instance.plot_position_vs_time()
    # kkw_instance.plot_position_vs_time_colored()
    for itr in densities:
        kkw_instance = KKW(itr,init_mode)
        kkw_instance.run(LMeasureFormula)
    #     kkw_instance.plot_position_vs_time()
    #     kkw_instance.plot_position_vs_time_colored()
        kkw_instance.create_graph()

