# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 14:58:02 2024

@author: zahra
"""

from collections import Counter

def unique_elements_by_frequency(lst):
    # Count the frequency of each element in the list
    freq_count = Counter(lst)
    
    # Sort the elements by frequency in descending order
    sorted_elements = sorted(freq_count, key=lambda x: freq_count[x], reverse=True)
    
    return sorted_elements, freq_count
    

def generate_graph(x,v,vehicles_phase,ncells):
    features = []
    edges = []
    labels = []
    nvehicles = len(x)
    ntimesteps = len(x[0])
    l = 0
    th = int(ncells/100)

    for t in range(ntimesteps-1):
        step = t % 20
        if t % 20 == 0:
            edges.append([])
            features.append([])
            counting_label = []
            q = 0
        for i in range(nvehicles):
            if x[i][t][1] != int(ncells):
                features[l].append([v[i][t+1][1], x[i][t][1], t])
                counting_label.append(vehicles_phase[i,t])
                for j in range(i, nvehicles): # graph with self-loop
                    distance = x[j][t][1] - x[i][t][1]
                    if abs(distance) <= 6 and x[j][t][1] != int(ncells):
                        edges[l].append(i+1+(step*nvehicles)-q)
                        edges[l].append(j+1+(step*nvehicles)-q)
                if step != 0 and x[i][t-1][1] != int(ncells):
                    edges[l].append(i+1+(step*nvehicles)-q)
                    edges[l].append(i+1+(step*nvehicles)-nvehicles)
            else:
                q += 1

        #features[l].append(v[nvehicles-1][t+1][1])
        if (t+1) % 20 == 0:
            labelt, freq_count = unique_elements_by_frequency(counting_label)
            
            # if len(labelt) >= 2:
            #     if int(labelt[0]) == 1 and int(labelt[1]) == 3:
            #       labels.append(int(labelt[1]))
            #     elif int(labelt[0]) == 1 and int(labelt[1]) == 2 and (int(freq_count[1])-int(freq_count[2]) < th):
            #       labels.append(int(labelt[1]))
            #     elif int(labelt[0]) == 2 and int(labelt[1]) == 3 and (int(freq_count[2])-int(freq_count[3]) < th):
            #       labels.append(int(labelt[1]))
            #     else:
            #       labels.append(int(labelt[0]))
            # else:
            labels.append(int(labelt[0]))
            l += 1
            if int(labelt[0]) == 2  and int(labelt[1]) == 1:
                for _ in range(1):
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