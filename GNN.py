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



class GraphDataset(Dataset):
    def __init__(self, data_file):
        self.graph_data = self.load_data(data_file)
        
    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        graph = self.graph_data[idx]
        x = torch.tensor(graph['node_features'], dtype=torch.float).view(-1, 1)
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
                node_features = [int(x) for x in line[0].split(',')]
                # Parse edge connections
                edge_indices = [int(x) for x in line[1:]]
                edge_indices = [(edge_indices[i], edge_indices[i+1]) for i in range(0, len(edge_indices)-1, 2)]
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
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv3(x, edge_index)
        x = self.pooling(x, batch)  # Perform global pooling to obtain graph embeddings

        return x


def train(model, train_loader, optimizer, device):
   # Training loop
   model.train()
   criterion = torch.nn.CrossEntropyLoss()
   total_loss = 0.0
   for data in train_loader:
       data = data.to(device)
       optimizer.zero_grad()
       output = model(data)
       output_flat = output.view(-1, output.size(-1))
       target_flat = data.y.view(-1)
       
       loss = criterion(output_flat, target_flat)
       loss.backward()
       optimizer.step()
       total_loss += loss.item() * data.num_graphs
   return total_loss / len(train_loader.dataset)

def test(model, test_loader, device):
    # Evaluation on testing data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.num_graphs
            correct += (predicted == data.y).sum().item()
    accuracy = correct / total
    return accuracy

def visualize(graph_embeddings, labels):

    # Apply t-SNE for dimensionality reduction
    embeddings_2d = TSNE(n_components=2).fit_transform(graph_embeddings)
    
    # Visualize the embeddings
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])
    
    plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], s=70, c=labels, cmap="Set2")
    plt.title('t-SNE Visualization of Node Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.show() 

if __name__ == "__main__":
    # Create dataset and data loaders
    dataset = GraphDataset('graph_dataset.txt')
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize model, optimizer, and loss function
    model = GCN(input_dim=dataset[0].num_features, hidden_dim=[128,64], output_dim=3).to(device)
    #model = MLP(input_dim=dataset[0].num_features, hidden_dim=16, output_dim=3).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
   
    for epoch in range(1, 45):
        train_loss = train(model, train_loader, optimizer, device)
        print(f'Epoch: {epoch:03d}, Loss: {train_loss:.4f}')
  
    # Testing
    test_acc = test(model, test_loader, device)
    print(f'Test Accuracy: {test_acc:.4f}')
    
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
    visualize(graph_embeddings, graph_labels)
      
    
# Example Dataset 
# features             edges             labels   
# 1.0, 2.0, 3.0, 4.0   0 1 1 2 2 3 3 0   0      #graph1
# 0.5, 1.5, 2.5, 3.5   0 1 1 2 2 3 3 0   1
# 2.0, 3.0, 4.0, 5.0   0 1 1 2 2 3 3 0   0

