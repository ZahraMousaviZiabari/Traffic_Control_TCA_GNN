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
        if t % 20 == 0:
            edges.append([])
            features.append([])
            counting_label = []
        for i in range(nvehicles):
            features[l].append([v[i][t+1][1], x[i][t+1][1], t])
            if x[i][t+1][1] != int(ncells):
                counting_label.append(vehicles_phase[i,t+1])
                for j in range(i, nvehicles): # graph with self-loop
                    distance = x[j][t+1][1] - x[i][t+1][1]
                    if abs(distance) <= 6 and x[j][t+1][1] != int(ncells) :
                        edges[l].append(i+len(features[l]))
                        edges[l].append(j+len(features[l]))
                        if t % 20 != 0:
                            edges[l].append(i+len(features[l]))
                            edges[l].append(i+len(features[l])-nvehicles)
        #features[l].append(v[nvehicles-1][t+1][1])
        if (t+1) % 20 == 0:
            labelt = most_common_element(counting_label)
            labels.append(int(labelt[0]))
            l += 1
            if int(labelt[0]) == 2:
                for _ in range(2):
                    edges.append(edges[l-1])
                    features.append(features[l-1])
                    labels.append(int(labelt[0]))
                    l += 1
            
    save_data(features, edges, labels, 'graph_dataset.txt')
    
    
def save_data(features, edges, labels, file_path):
    with open(file_path, 'a') as file:
        for i in range(len(features)):
            # Convert features, edges, and label of each graph to strings
            feature_str = '|'.join([','.join(map(str, feat)) for feat in features[i]])
            edge_str = ' '.join(map(str, edges[i]))
            label_str = str(labels[i])
            # Write the graph data to the file
            file.write(f"{feature_str}   {edge_str}   {label_str}\n")


# Example usage:
# features = [[[1.0, 10], [2.0, 20], [3.0, 30]], [[0.5, 15], [1.5, 25], [2.5, 35]], [[2.0, 10], [3.0, 20], [4.0, 30]]]
# edges = [[0, 1, 1, 2, 2, 3, 3, 0], [0, 1, 1, 2, 2, 3, 3, 0], [0, 1, 1, 2, 2, 3, 3, 0]]
# labels = [0, 1, 0]   