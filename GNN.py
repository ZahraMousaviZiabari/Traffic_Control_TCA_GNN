import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch.nn import Linear
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from mpl_toolkits.mplot3d import Axes3D



class GraphDataset(Dataset):
    def __init__(self, data_file):
        self.graph_data = self.load_data(data_file)
        
    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        graph = self.graph_data[idx]
        x = torch.tensor(graph['node_features'], dtype=torch.float)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).t().contiguous()
        y = torch.tensor(graph['target']-1, dtype=torch.long)
        num_nodes = len(graph['node_features'])
        data = Data(x=x, edge_index=edge_index, y=y)
        return data
    
    def load_data(self, data_file):
        graph_data = []
        with open(data_file, 'r') as file:
            for line in file:
                line = line.strip().split()
                # Parse node features
                node_features = [[int(x) for x in feat.split(',')] for feat in line[0].split('|')]
                # Parse edge connections
                edge_indices = [int(x) for x in line[1:]]
                num_nodes = len(node_features)
                edge_indices = [(edge_indices[i], edge_indices[i+1]) for i in range(0, len(edge_indices)-1, 2) if edge_indices[i] < num_nodes and edge_indices[i+1] < num_nodes]
                # Parse target label
                target = int(line[-1])
                # Create graph dictionary
                graph = {'node_features': node_features, 'edge_index': edge_indices, 'target': target}
                graph_data.append(graph)
        return graph_data


class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x = self.lin1(data.x)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x
    
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim[0])
        self.conv2 = GCNConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GCNConv(hidden_dim[1], output_dim)
        self.pooling = global_mean_pool
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        #x = F.relu(x)
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.pooling(x, batch)  # Perform global pooling to obtain graph embeddings

        return x

def compute_class_proportions(dataset):
    # Initialize counters for each class
    class_counts = torch.zeros(3, dtype=torch.float)
    
    # Loop over the dataset to count occurrences of each class
    for data in dataset:
        class_counts[data.y] += 1
    
    # Compute class proportions
    class_proportions = class_counts / len(dataset)
    
    return class_proportions

def train(model, train_loader, optimizer, device):
   # Training loop
   model.train()
   # Add Kullback-Leibler (KL) Divergence Loss
   kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
   criterion_ce = torch.nn.CrossEntropyLoss()
   total_loss = 0.0
   for data in train_loader:
       data = data.to(device)
       optimizer.zero_grad()
       output = model(data)
       output_flat = output.view(-1, output.size(-1))
       target_flat = data.y.view(-1)
       
       ce_loss = criterion_ce(output_flat, target_flat)
      
        # Compute KL Divergence Loss using class proportions
       kl_loss_value = kl_loss(F.log_softmax(output, dim=-1), class_proportions)
       
       lambda_kl = 0.1
       # Combine both losses
       loss = ce_loss + kl_loss_value * lambda_kl  

       loss.backward()
       optimizer.step()
       total_loss += loss.item() * data.num_graphs
   return total_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    # Evaluation on testing data
    model.eval()
    y_true = []
    y_pred = []
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.num_graphs
            correct += (predicted == data.y).sum().item()
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            
    accuracy = correct / total
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    cm = confusion_matrix(y_true, y_pred)
    
    return accuracy, precision, recall, f1, cm

def visualize(graph_embeddings, labels, ptype):
    if ptype == ('2d'):
        # Apply t-SNE for dimensionality reduction
        embeddings_2d = TSNE(n_components=2).fit_transform(graph_embeddings)
        
        unique_labels = set(labels)
        phase_colors = {0: 'blue', 1: 'green', 2: 'red'}
        
        # Visualize the embeddings
        plt.figure(figsize=(10, 10))
   
        for i, label in enumerate(unique_labels):
            indices = labels == label
            plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], s=40, c=[phase_colors[i]], label=f'Class {label+1}')
        
        #plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=70, c=labels, cmap="Set2")
        plt.title('t-SNE Visualization of Graph Embeddings')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.show() 

    if ptype == ('3d'):
        # Apply t-SNE for dimensionality reduction
        embeddings_3d = TSNE(n_components=3).fit_transform(graph_embeddings)
        
        unique_labels = set(labels)
        phase_colors = {0: 'blue', 1: 'green', 2: 'red'}
        
        # Visualize the embeddings
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for i, label in enumerate(unique_labels):
            indices = labels == label
            ax.scatter(embeddings_3d[indices, 0], embeddings_3d[indices, 1], embeddings_3d[indices, 2], s=40, c=[phase_colors[i]], label=f'Class {label+1}')

        ax.set_title('t-SNE Visualization of Graph Embeddings')
        ax.set_xlabel('t-SNE Dimension 1')
        ax.set_ylabel('t-SNE Dimension 2')
        ax.set_zlabel('t-SNE Dimension 3')
        ax.legend()
        plt.show() 
        
def plot_loss(train_losses):
    plt.plot(train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

        
if __name__ == "__main__":
    # Create dataset and data loaders
    dataset = GraphDataset('graph_dataset.txt')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Compute class proportions
    class_proportions = compute_class_proportions(train_dataset).to(device)

    
    # Initialize model, optimizer, and loss function
    input_dim = dataset[0].num_node_features   # Get the number of features per node from the first graph
    model = GCN(input_dim=input_dim, hidden_dim=[64,32], output_dim=3).to(device)
    #model = MLP(input_dim=dataset[0].num_features, hidden_dim=16, output_dim=3).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=4e-4)
    
    # Training loop
    train_losses = []
    for epoch in range(1, 40):
        train_loss = train(model, train_loader, optimizer, device)
        train_losses.append(train_loss)
        print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}')
        
    # Plot the training loss
    plot_loss(train_losses)
    
  
    # Testing and Evaluation
    test_acc, precision, recall, f1, cm = test(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print('Confusion Matrix:')
    print(cm)
    
    graph_embeddings = []
    graph_labels = []
    model.eval()
    
    # Iterate over the test dataset and collect node embeddings and labels
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            graph_embeddings.append(output.cpu().numpy())
            graph_labels.append(data.y.cpu().numpy())
    
    # Concatenate node embeddings and labels
    graph_embeddings = np.concatenate(graph_embeddings, axis=0)
    graph_labels = np.concatenate(graph_labels, axis=0)
    visualize(graph_embeddings, graph_labels, '2d')
      
    
# Example Dataset 
# features               edges             labels   
# 1,2|5,43|5,138|5,1    0 1 1 2 2 3 3 0   0      #graph1
# 5,48|5,143|5,6|5,43   0 1 1 2 2 3 3 0   1


