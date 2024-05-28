
import os, sys, csv

if 'SUMO_HOME' in os.environ:
     tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
     sys.path.append(tools)
else:
     sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import numpy as np
np.random.seed(1234)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from scipy import signal

scenario = "highway"

traci.start(["sumo", "-c", scenario+"/"+scenario+".sumocfg"])

deltaX = 0.010 # in km, more than a vehicle
L = 2.5 # in km
deltaT = traci.simulation.getDeltaT()/60 # in min
Tmax = 15 # in min
Tstart = 8 # in min
sigma = 0.01 # in km
tau = 0.06 # in min
dt = 1 #s
dx = 0.5 #m

Nt = int(np.ceil(Tmax/deltaT))
NtStart = int(np.floor(Tstart/deltaT))
Nx = int(np.ceil(L/deltaX))

numberOfVehicles = np.zeros((Nx, Nt-NtStart))
trafficLightPhase = 0


def generate_graph(x,v,ncells):
    features = []
    edges = []
    nvehicles = len(x[0])
    ntimesteps = len(x)
    l = 0
    print("nvehicles",nvehicles)
    print("ntimesteps",ntimesteps)
    for t in range(ntimesteps):
        if t % 20 == 0:
            edges.append([])
            features.append([])
        for i in range(nvehicles):
            features[l].append([v[t][i], x[t][i], t])
            if x[t][i] != int(ncells):
                for j in range(i, nvehicles): # graph with self-loop
                    distance = x[t][j] - x[t][i]
                    if abs(distance) <= 6 and x[t][j] != int(ncells) :
                        edges[l].append(i+len(features[l]))
                        edges[l].append(j+len(features[l]))
                        if t % 20 != 0:
                            edges[l].append(i+len(features[l]))
                            edges[l].append(i+len(features[l])-nvehicles)
        if (t+1) % 20 == 0:
            l += 1

    save_data(features, edges, 'graph_dataset.txt')
    
    
def save_data(features, edges, file_path):
    with open(file_path, 'a') as file:
        for i in range(len(features)):
            # Convert features, edges, and label of each graph to strings
            feature_str = '|'.join([','.join(map(str, feat)) for feat in features[i]])
            edge_str = ' '.join(map(str, edges[i]))
            # Write the graph data to the file
            file.write(f"{feature_str}   {edge_str}\n")

def plot_position_vs_time(x,v):
     
    # Extract positions from the nested list
    flattened_positions = [p for sublist in x for p in sublist]
    time = list(range(len(flattened_positions)))
    nvehicles = len(x[0])
    ntimesteps = len(x)
    print("nvehicles",nvehicles)
    print("ntimesteps",ntimesteps)
    print("time", len(time))
    # Flatten the speeds array
    flattened_speeds = [v for sublist in v for v in sublist]

    if len(flattened_positions) != len(flattened_speeds):
        raise ValueError("Lengths of position array and velocity array must match.")

    
    # Normalize speeds to range [0, 1], considering vmax
    normalized_speeds =  [speed / 5 for speed in flattened_speeds]

    # Create a colormap ranging from black to light gray
    cmap = cm.get_cmap('Greys')
    reversed_cmap = cmap.reversed()
    colors_adjusted = [cmap((5 - i) / 5) for i in range(5)]
    reversed_cmap = cm.colors.ListedColormap(colors_adjusted[::1])
    # Map normalized speeds to colors from the colormap
    colors = reversed_cmap(normalized_speeds)
        
    # Plotting
    plt.scatter(time, flattened_positions, marker='.', color=colors, s=4)
   
    plt.xlabel('Time')
    plt.ylabel('Position')
    plt.title('Position vs Time')
    plt.legend()
    plt.grid(True)
    plt.savefig("PositionVSTime.jpg", bbox_inches="tight")
    plt.show()

p = []
v = []
ncells = int((L * 1000) / dx)
print("\n NtStart",NtStart)
print("Nt",Nt)
for n in range(Nt):
   if n%100 == 0:
      print("step", n)
   if n >= NtStart:
       p.append([])
       v.append([])
       for vehID in traci.vehicle.getIDList():
           if traci.vehicle.getRouteID(vehID) == 'route_0' or traci.vehicle.getRouteID(vehID) == 'route_1':
               vehPos = traci.vehicle.getPosition(vehID)[0]
               vehSpeed = traci.vehicle.getSpeed(vehID)

               if 0 <= vehPos < L*1000:    
                   p[n-NtStart].append(int(vehPos/dx))
                   v[n-NtStart].append(int((vehSpeed*dt)/dx))
               else:
                   p[n-NtStart].append(ncells)
                   v[n-NtStart].append(0)                      
            
               if vehPos >= L*1000:
                   continue
     
               i = int(np.floor(vehPos/(1000*deltaX)))
               if 0 <= i < Nx:
                   numberOfVehicles[i,n-NtStart] += 1

   traci.simulationStep()
print("step", n)
traci.close()

#min-max normalization
max_val =  max(max(s) for s in v)
min_val =  min(min(s) for s in v)
diff = max_val - min_val
for r in range(len(v)):
    for c in range(len(v[r])):
        v[r][c] = int(((v[r][c] - min_val) / diff) * 5)

t = np.linspace(Tstart, Tmax, Nt-NtStart)
x = np.linspace(0, L, Nx)
X, Y = np.meshgrid(t, x)

fig = plt.figure(figsize=(7.5, 5))
plt.pcolor(X, Y, numberOfVehicles, shading='auto', cmap='rainbow')
plt.xlabel('Time [min]')
plt.ylabel('Position [km]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()
plt.show()

maxI = int(np.ceil(5*sigma/deltaX))
maxJ = int(np.ceil(5*tau/deltaT))
kernel = np.zeros((2*maxI+1, 2*maxJ+1))
for i in range(2*maxI+1):
    for j in range(2*maxJ+1):
        newI = i-maxI-1
        newJ = j-maxJ-1
        kernel[i,j] = np.exp(-abs(newI)*deltaX/sigma - abs(newJ)*deltaT/tau)
N = kernel.sum()
density = signal.convolve2d(numberOfVehicles, kernel, boundary='symm', mode='same')/N
densityMax = np.amax(density)
density = density/densityMax

fig = plt.figure(figsize=(7.5, 5))
plt.pcolor(X, Y, density, shading='auto', vmin=0, vmax=1, cmap='rainbow')
plt.xlabel('Time [min]')
plt.ylabel('Position [km]')
plt.xlim(min(t), max(t))
plt.ylim(min(x), max(x))
plt.colorbar()
plt.tight_layout()
plt.show()


generate_graph(p,v,ncells)
plot_position_vs_time(p,v)

# with open(scenario+'/spaciotemporal.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows([[L, (Tmax-Tstart)]])
#     writer.writerows(density)
    
# with open(scenario+'/pv.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(np.array([xVar, tVar, rhoPV, vPV]).T)
    