# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:58:02 2024

@author: zahra
"""

from collections import Counter

def most_common_element(lst):
    counts = Counter(lst)
    max_count = max(counts.values())
    return [num for num, count in counts.items() if count == max_count]
    

def generate_graph(x,v,vehicles_phase,ncells):
    features = []
    edges = []
    labels = []
    nvehicles = len(x)
    ntimesteps = len(x[0])
    l = 0

    for t in range(ntimesteps-1):
        edges.append([])
        features.append([])
        for i in range(nvehicles):
            features[l].append(v[i][t+1][1])
            for j in range(i, nvehicles): # graph with self-loop
                distance = x[j][t+1][1] - x[i][t+1][1]  
                if abs(distance) <= 6 and x[j][t+1][1] != int(ncells) and x[i][t+1][1] != int(ncells) :
                    edges[l].append(i)
                    edges[l].append(j)
        #features[l].append(v[nvehicles-1][t+1][1])
        labelt = most_common_element(vehicles_phase[:,t+1])
        labels.append(int(labelt[0]))
        l += 1
        if int(labelt[0]) == 1:
            for _ in range(3):
                edges.append(edges[l-1])
                features.append(features[l-1])
                labels.append(int(labelt[0]))
                l += 1
            
    save_data(features, edges, labels, 'graph_dataset.txt')
    
    
def save_data(features, edges, labels, file_path):
    with open(file_path, 'a') as file:
        for i in range(len(features)):
            # Convert features, edges, and label of each graph to strings
            feature_str = ','.join(map(str, features[i]))
            edge_str = ' '.join(map(str, edges[i]))
            label_str = str(labels[i])
            # Write the graph data to the file
            file.write(f"{feature_str}   {edge_str}   {label_str}\n")

# Example usage:
# features = [[1.0, 2.0, 3.0], [0.5, 1.5, 2.5], [2.0, 3.0, 4.0]]
# edges = [[0, 1, 1, 2, 2, 3, 3, 0], [0, 1, 1, 2, 2, 3, 3, 0], [0, 1, 1, 2, 2, 3, 3, 0]]
# labels = [0, 1, 0]    