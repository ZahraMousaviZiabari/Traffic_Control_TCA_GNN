import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE



class GraphDataset(Dataset):
    def __init__(self, data_file):
        self.graph_data = self.load_data(data_file)
        
    def __len__(self):
        return len(self.graph_data)
    
    def __getitem__(self, idx):
        graph = self.graph_data[idx]
        x = torch.tensor(graph['node_features'], dtype=torch.float).view(-1, 1)
        edge_index = torch.tensor(graph['edge_index'], dtype=torch.long).t().contiguous()
        y = torch.tensor([graph['target'] - 1] * len(graph['node_features']), dtype=torch.long)
        num_nodes = len(graph['node_features'])  # Get the number of features
        num_features = 1
        # Create a batch vector where each element corresponds to the graph index
        batch = torch.zeros(num_nodes, dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y, num_nodes=num_nodes, batch=batch)
    
        data.num_features = num_features  # Update the num_features attribute
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


class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim[0])
        self.conv2 = GCNConv(hidden_dim[0], hidden_dim[1])
        self.conv3 = GCNConv(hidden_dim[1], output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        x = self.conv3(x, edge_index)
        return F.log_softmax(x, dim=1)


def train():
   # Training loop
   model.train()
   for data in train_loader:
       data = data.to(device)
       optimizer.zero_grad()
       output = model(data)
       # Flatten the output and target tensors
       output_flat = output.view(-1, output.size(-1))
       target_flat = data.y.view(-1)

       # Calculate the loss
       loss = F.cross_entropy(output_flat, target_flat)
       
       optimizer.zero_grad()
       loss.backward()
       optimizer.step()

   return loss

def test():
    # Evaluation on testing data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += data.y.size(0)
            correct += (predicted == data.y).sum().item()
    accuracy = correct / total
    return accuracy

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
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
    # Initialize model, optimizer, and loss function
    model = GCN(input_dim=dataset[0].num_features, hidden_dim=[32,16], output_dim=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-3)
    
   
    for epoch in range(1, 31):
        loss = train()
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
    test_acc = test()
    print(f'Test Accuracy: {test_acc:.4f}')

        
    
# Example Dataset 
# features             edges             labels   
# 1.0, 2.0, 3.0, 4.0   0 1 1 2 2 3 3 0   0      #graph1
# 0.5, 1.5, 2.5, 3.5   0 1 1 2 2 3 3 0   1
# 2.0, 3.0, 4.0, 5.0   0 1 1 2 2 3 3 0   0

