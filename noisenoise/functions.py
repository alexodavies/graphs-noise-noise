import torch
from tqdm import tqdm
from random import random
from .feature_noise import add_continuous_feature_noise, add_discrete_feature_noise, shuffle_categorical_feature_noise, weighted_categorical_feature_noise, dataset_shuffle_feature_noise, dataset_shuffle_node_feature_noise, dataset_feature_ranges
from .structure_noise import add_structure_noise, add_structure_noise_degree_preserving, erdos_renyi_from_data

def add_noise_to_graph(data, t_structure, t_feature, min_x = 0, max_x = 0, min_attr = 0, max_attr = 0):
    if random() < t_structure:
        data = erdos_renyi_from_data(data)

    if random() < t_feature:
        if data.x is not None:
            data.x = torch.rand_like(data.x) * (max_x-min_x) + min_x
        if data.edge_attr is not None:
            data.edge_attr = torch.rand_like(data.edge_attr) * (max_attr-min_attr) + min_attr
    # data = add_structure_noise(data, t_structure)
    # data = add_discrete_feature_noise(data, t_feature)
    # data = shuffle_categorical_feature_noise(data, t_feature)
    # data = add_continuous_feature_noise(data, t_feature)
    return data

def add_noise_to_dataset(dataset, t_structure, t_feature):
    noisy_dataset = []
    min_x, max_x, min_edge_attr, max_edge_attr = dataset_feature_ranges(dataset)
    for data in tqdm(dataset, desc='Adding noise', colour='green', leave=False):
        noisy_data = add_noise_to_graph(data, t_structure, t_feature, min_x = min_x, max_x = max_x, min_attr=min_edge_attr, max_attr = max_edge_attr)
        noisy_dataset.append(noisy_data)

    # noisy_dataset = dataset_shuffle_node_feature_noise(noisy_dataset, t_feature)
    
    return noisy_dataset

def add_weighted_noise_to_graph(data, t_structure, t_feature, weights_nodes, weights_edges):
    data = add_structure_noise(data, t_structure)
    # data = add_discrete_feature_noise(data, t_feature)
    data = weighted_categorical_feature_noise(data, t_feature, weights_nodes, weights_edges)
    return data

def add_weighted_noise_to_dataset(dataset, t_structure, t_feature, weights_nodes, weights_edges):
    noisy_dataset = []
    
    for data in tqdm(dataset, desc='Adding noise', colour='green', leave=False):
        noisy_data = add_noise_to_graph(data, t_structure, t_feature)
        noisy_dataset.append(noisy_data)
    
    return noisy_dataset

def compute_onehot_probabilities(dataloader):
    n_feats = None
    total = None
    n_points = 0

    for data in dataloader:
        if n_feats is None:
            n_feats = data.x.shape[1]
            total = torch.zeros(n_feats)
        
        n_points += data.x.shape[0]
        total += data.x.sum(dim=0)

    return total / n_points

def compute_onehot_probabilities_edge(dataloader):
    n_feats = None
    total = None
    n_points = 0

    for data in dataloader:
        if n_feats is None:
            n_feats = data.edge_attr.shape[1]
            total = torch.zeros(n_feats)
        
        n_points += data.edge_attr.shape[0]
        total += data.edge_attr.sum(dim=0)

    return total / n_points

