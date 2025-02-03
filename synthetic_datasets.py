import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm
from random import random
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.utils import to_networkx, erdos_renyi_graph
import networkx as nx
import copy

from noisenoise import add_noise_to_dataset

def pyg_to_networkx_connected(pyg_data):
    """
    Converts a PyTorch Geometric graph into a NetworkX graph, 
    dropping unconnected nodes.

    Parameters:
        pyg_data (torch_geometric.data.Data): The PyG graph data.

    Returns:
        networkx.Graph: The converted NetworkX graph without isolated nodes.
    """
    # Convert PyG graph to NetworkX
    nx_graph = to_networkx(pyg_data, to_undirected=True)

    # Remove isolated (unconnected) nodes
    isolated_nodes = list(nx.isolates(nx_graph))
    nx_graph.remove_nodes_from(isolated_nodes)

    return nx_graph

def get_degree_list(data):
    edge_index = data.edge_index
    num_nodes = data.num_nodes
    degrees = torch.zeros(num_nodes, dtype=torch.long)
    for i in range(num_nodes):
        degrees[i] = torch.sum(edge_index == i)/2
    return degrees

def noise_and_visualise(dataset):
    n_noise_levels = 8
    noise_levels = np.linspace(0, 1, n_noise_levels).tolist()
    fig, axes = plt.subplots(4, n_noise_levels, figsize=(n_noise_levels*2, 8), squeeze=True)

    index_label_0 = 0
    index_label_1 = 0

    for di, d in enumerate(dataset):
        if d.y == 0:
            index_label_0 = di
            break

    for di, d in enumerate(dataset):
        if d.y == 1:
            index_label_1 = di
            break

    d0 = dataset[index_label_0]
    d1 = dataset[index_label_1]
    
    g0 = to_networkx(d0)
    g1 = to_networkx(d1)
    pos0 = nx.kamada_kawai_layout(g0)
    pos1 = nx.kamada_kawai_layout(g1)

    for i, noise in enumerate(noise_levels):
        noisy_dataset = add_noise_to_dataset(copy.deepcopy(dataset), noise, noise)
        d0_noisy = noisy_dataset[index_label_0]
        d1_noisy = noisy_dataset[index_label_1]

        axes[0, i].set_title(f"t= {noise:.2f}")

        g0 = to_networkx(d0_noisy, to_undirected=True)
        g1 = to_networkx(d1_noisy, to_undirected=True)
        
        axes[0,i].set_title(f"Nodes: {d0_noisy.num_nodes}, edges: {d0_noisy.num_edges}")
        axes[1,i].set_title(f"Nodes: {d1_noisy.num_nodes}, edges: {d1_noisy.num_edges}")

        nx.draw_networkx_edges(g0, pos = pos0, ax=axes[0, i], node_size=0, edge_color="gray")
        nx.draw_networkx_edges(g1, pos = pos1, ax=axes[1, i], node_size=0, edge_color="gray")
        # nx.draw(pyg_to_networkx_connected(d1_noisy), ax=axes[1, i], node_size=0, edge_color="gray")

        d0_feats = d0_noisy.x.numpy().flatten()
        d1_feats = d1_noisy.x.numpy().flatten()

        axes[2, i].hist(d0_feats, bins=20, alpha=0.5, label='Class 0')
        axes[2, i].hist(d1_feats, bins=20, alpha=0.5, label='Class 1')

        axes[3, i].hist(get_degree_list(d0_noisy).numpy(), alpha=0.5, label='Class 0', bins = np.linspace(0,4,12))
        axes[3, i].hist(get_degree_list(d1_noisy).numpy(), alpha=0.5, label='Class 1', bins = np.linspace(0,4,12))

        for ax in axes[:-1, i]:
            ax.set_xticks([])  # Remove x-axis ticks
            ax.set_yticks([])  # Remove y-axis ticks
            ax.set_frame_on(False)  # Remove the border/frame

        axes[3, i].set_yticks([])  # Remove y-axis ticks
        # axes[3, i].set_frame_on(False)

        # axes[2,i].set_axis_off()

    axes[0,0].set_ylabel("Ladder Ring", rotation = "vertical")
    axes[1,0].set_ylabel("Hex Grid", rotation = "vertical")
    axes[2,0].set_ylabel("Node Features", rotation = "vertical")
    axes[3,0].set_ylabel("Degrees", rotation = "vertical")

    plt.tight_layout()
    plt.savefig("synthetic_noise.png", dpi=300)
    # plt.show()



def generate_hexagonal_grid_graph(width, height) -> Data:
    """
    Generates a hexagonal grid graph with approximately `num_edges` edges.
    Ensures that nodes with only 2 edges are connected to another node with 2 edges.

    Parameters:
        num_edges (int): The desired number of edges in the graph.

    Returns:
        Data: A PyTorch Geometric Data object representing the hexagonal grid.
    """
    # if num_edges < 6:
    #     raise ValueError("A hexagonal grid must have at least 6 edges.")

    # # Compute the best grid dimensions (width, height)
    # estimated_hexes = num_edges // 2  # Each hex contributes about 3 edges
    # grid_width = int(math.sqrt(estimated_hexes))  # Empirical scaling factor
    # grid_height = max(1, estimated_hexes // grid_width)  # Ensure non-zero height

    # Generate a hexagonal lattice using NetworkX
    G = nx.hexagonal_lattice_graph(width, height, create_using=nx.Graph)
    num_edges = 3*width*height - (width+height+3)/2

    # Convert to an undirected graph
    G = nx.Graph(G)

    # # If the number of edges exceeds the desired count, trim excess edges
    # while G.number_of_edges() > num_edges:
    #     edge_to_remove = list(G.edges)[-1]
    #     G.remove_edge(*edge_to_remove)

    # Identify nodes with exactly 2 edges and connect them
    two_degree_nodes = [node for node, degree in dict(G.degree()).items() if degree == 2]

    while len(two_degree_nodes) > 1:
        node1 = two_degree_nodes.pop()
        node2 = two_degree_nodes.pop()
        G.add_edge(node1, node2)
    # print(G.edges)
    G = nx.convert_node_labels_to_integers(G)
    # Convert NetworkX graph to PyTorch Geometric format
    edge_index = torch.tensor(list(G.edges), dtype=torch.long).T
    data = Data(edge_index=edge_index)

    return data

def generate_circular_ladder_graph(num_edges: int) -> Data:
    """
    Generates a circular ladder graph with `num_rungs` rungs.
    The ladder wraps back on itself, forming a ring.

    Parameters:
        num_rungs (int): The number of rungs (steps) in the ladder.

    Returns:
        Data: A PyTorch Geometric Data object representing the circular ladder graph.
    """
    if num_edges < 4:
        raise ValueError("A circular ladder graph requires at least 2 rungs.")
    num_rungs = int(num_edges/2)
    edges = []

    # Create 2 * num_rungs nodes (each rung has two nodes)
    for i in range(num_rungs):
        # Define top and bottom nodes for this rung
        top = i
        bottom = i + num_rungs

        # Connect vertical rungs
        edges.append((top, bottom))

        # Connect horizontal rails
        edges.append((top, (i + 1) % num_rungs))  # Top ring
        edges.append((bottom, ((i + 1) % num_rungs) + num_rungs))  # Bottom ring

    # Convert edge list to tensor
    edge_index = torch.tensor(edges, dtype=torch.long).T

    # Create PyTorch Geometric Data object
    data = Data(edge_index=edge_index, num_nodes = torch.max(torch.unique(edge_index)))

    return data

def generate_bimodal_nodes(data, mean=1, dev=1):
    """Attaches a specified normal distribution to the nodes of the input data."""
    n_nodes = data.num_nodes
    n_features = 5

    # Generate normal node features
    data.x = torch.randn(n_nodes, n_features) * dev + mean
    return data

def generate_bimodal_edges(data, mean=1, dev=1):
    """Attaches a specified normal distribution to the nodes of the input data."""
    n_edges = data.num_edges
    n_features = 5

    # Generate normal node features
    data.edge_attr = torch.randn(n_edges, n_features) * dev + mean
    return data





def generate_triangular_grid(resolution=3):
    """Generates a square triangular grid with 'resolution' triangles per side."""
    
    num_nodes_per_side = resolution + 1  # Grid is (resolution+1) x (resolution+1)
    indices = [(i, j) for i in range(num_nodes_per_side) for j in range(num_nodes_per_side)]
    
    # Mapping 2D indices to a single node index
    index_map = {pos: idx for idx, pos in enumerate(indices)}
    
    edges = set()
    
    for i in range(resolution):
        for j in range(resolution):
            # Node indices
            v0 = index_map[(i, j)]
            v1 = index_map[(i + 1, j)]
            v2 = index_map[(i, j + 1)]
            v3 = index_map[(i + 1, j + 1)]
            
            # Split the square into two triangles
            edges.add((v0, v1))
            edges.add((v1, v3))
            edges.add((v3, v2))
            edges.add((v2, v0))
            
            if (i + j) % 2 == 0:
                edges.add((v0, v3))  # Diagonal from bottom-left to top-right
            else:
                edges.add((v1, v2))  # Diagonal from top-left to bottom-right

    edge_index = torch.tensor(list(edges), dtype=torch.long).T
    return Data(edge_index=edge_index, num_nodes=len(indices))



def erdos_renyi_from_data(data):
    """Generates a random graph of the same density as the input"""
    n_nodes = data.num_nodes
    n_edges = data.num_edges

    potential_connections = (n_nodes**2) 

    density = n_edges / potential_connections
    data.edge_index = erdos_renyi_graph(num_nodes=n_nodes, edge_prob=density, directed=False)
    return data



class SyntheticDataset(InMemoryDataset):
    def __init__(self, root, label_type, num_samples=8000, transform=None, pre_transform=None):
        self.label_type = label_type
        self.num_samples = num_samples
        super(SyntheticDataset, self).__init__(root, transform, pre_transform)

        # Ensure processed directory exists before loading
        os.makedirs(self.processed_dir, exist_ok=True)

        # Load or generate data
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.task_type = "classification"
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self.label_type}.pt']

    def process(self):
        """Processes and saves the dataset."""
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generates synthetic graph data and stores it."""
        data_list = []
        is_feature = self.label_type.endswith("feature")

        for _ in tqdm(range(self.num_samples)):
            # resolution = np.random.randint(2, 4)
            # sphere_edges = 3 * 2 ** (2 * resolution) * 10  # Approximate edge count for sphere
            # triangle_resolution = int(np.round((np.sqrt(2 * sphere_edges / 3) - 1), decimals=0))  # Adjust for edges
            # width = np.random.randint(2, 4)
            # height = np.random.randint(2, 4)

            # num_edges = 3*width*height - (width+height+3)/2 + 2*(width + height)
            num_edges = int(2*np.random.randint(24,256))

            # num_edges = 

            is_sphere = random() > 0.5
            structure_label = 1 if is_sphere else 0    
            feature_label = 1 if random() > 0.5 else 0

            # if is_sphere:
            #     data = generate_hexagonal_grid_graph(width = width, height = height)
            # else:
            data = generate_circular_ladder_graph(num_edges=num_edges)
            if is_sphere:
                data = erdos_renyi_from_data(data)

            # mean = 1 if random() > 0.5 else -1
            data = generate_bimodal_nodes(data, mean=2*(feature_label-0.5))
            data = generate_bimodal_edges(data, mean=2*(feature_label-0.5))

            data.y = torch.tensor([feature_label if is_feature else structure_label], dtype=torch.long)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

        # Ensure processed directory exists before saving
        os.makedirs(self.processed_dir, exist_ok=True)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return f"SyntheticDataset({self.label_type})"
    

class SyntheticDouble(InMemoryDataset):
    def __init__(self, root, label_type, num_samples=8000, transform=None, pre_transform=None):
        self.label_type = label_type
        self.num_samples = num_samples
        super(SyntheticDouble, self).__init__(root, transform, pre_transform)

        # Ensure processed directory exists before loading
        os.makedirs(self.processed_dir, exist_ok=True)

        # Load or generate data
        if not os.path.exists(self.processed_paths[0]):
            self.process()
        self.task_type = "classification"
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [f'data_{self.label_type}.pt']

    def process(self):
        """Processes and saves the dataset."""
        self._generate_synthetic_data()

    def _generate_synthetic_data(self):
        """Generates synthetic graph data and stores it."""
        data_list = []
        is_coupled = self.label_type.endswith("coupled")

        for _ in tqdm(range(self.num_samples)):
            # resolution = np.random.randint(2, 4)
            # sphere_edges = 3 * 2 ** (2 * resolution) * 10  # Approximate edge count for sphere
            # triangle_resolution = int(np.round((np.sqrt(2 * sphere_edges / 3) - 1), decimals=0))  # Adjust for edges
            # width = np.random.randint(2, 8)
            # height = np.random.randint(2, 8)

            # num_edges = 3*width*height - (width+height+3)/2 + 2*(width + height)
            num_edges = int(2*np.random.randint(24,256))


            if is_coupled:
                is_ladder = random() > 0.5
                is_neg_mean = random() > 0.5

                if is_ladder == is_neg_mean:
                    label = 1
                else:
                    label = 0

            else:
                is_ladder = random() > 0.5
                is_neg_mean = is_ladder

                label = int(is_ladder)

            # is_sphere = random() > 0.5
            # structure_label = 1 if is_sphere else 0    
            # feature_label = 1 if random() > 0.5 else 0

            # if is_ladder:
            #     data = generate_hexagonal_grid_graph(width = width, height = height)
            # else:
            #     data = generate_circular_ladder_graph(num_edges=num_edges)

            data = generate_circular_ladder_graph(num_edges=num_edges)
            if not is_ladder:
                data = erdos_renyi_from_data(data)

            # mean = 1 if random() > 0.5 else -1
            data = generate_bimodal_nodes(data, mean= -1 if is_neg_mean else 1)
            data = generate_bimodal_edges(data, mean= -1 if is_neg_mean else 1)

            data.y = torch.tensor(label, dtype=torch.long)
            data_list.append(data)

        self.data, self.slices = self.collate(data_list)

        # Ensure processed directory exists before saving
        os.makedirs(self.processed_dir, exist_ok=True)

        torch.save((self.data, self.slices), self.processed_paths[0])

    def __repr__(self):
        return f"CoupledDataset({self.label_type})"


def visualize_graph(data, title="Graph Visualization"):
    """Visualizes a given PyG Data object."""
    graph = to_networkx(data, to_undirected=True)
    plt.figure(figsize=(6, 6))
    nx.draw(graph, node_size=20, node_color = data.x[:,0], edge_color="gray")
    plt.annotate(title, (0,0))
    plt.show()


if __name__ == "__main__":
    # Generate and load dataset
    dataset = SyntheticDataset(root='data/synthetic', label_type="structure", num_samples=500)
    noise_and_visualise(dataset)
    # # Print dataset details
    # print(dataset)
    # for i in range(len(dataset)):
    #     if dataset[i].y == 0:
    #     # Visualize a couple of graphs from the dataset
    #         print(torch.mean(dataset[i].x))
    #         visualize_graph(dataset[i], title=f"Sample Graph 1 {dataset[i].y}")
    #         break

    # for i in range(len(dataset)):
    #     if dataset[i].y == 1:
    #     # Visualize a couple of graphs from the dataset
    #         print(torch.mean(dataset[i].x))
    #         visualize_graph(dataset[i], title=f"Sample Graph 1 {dataset[i].y}")
    #         break
    # edges_c1 = [d.num_edges for d in dataset if d.y == 0]
    # edges_c2 = [d.num_edges for d in dataset if d.y == 1]
    # plt.hist(edges_c1, bins=20, alpha=0.5, label='Class 0')
    # plt.hist(edges_c2, bins=20, alpha=0.5, label='Class 1')
    # plt.legend()
    # plt.show()
