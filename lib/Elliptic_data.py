import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data

def load_elliptic_data(config):
    """
    Load the edge list and features of the Elliptic dataset, with an option to use augmented data.
    Args:
        config :.
            - DATASET.AUGMENT_DIR (str): Path to the augmented dataset directory.
            - DATASET.USE_AUGMENTSET (bool): Flag indicating whether to use augmented data. Default is False.
    Returns:
        tuple: A tuple containing the feature data and edge list.
    """
    # Determine the dataset directory based on the augmentation flag
    print(config)
    DATASET_DIR = config["DATASET"]["AUGMENT_DIR"] if config["DATASET"]["USE_AUGMENTSET"] else config["DATASET"]["BASE_DIR"]

    # Construct file paths
    edges_file = os.path.join(DATASET_DIR, 'elliptic_txs_edgelist.csv')
    classes_file = os.path.join(DATASET_DIR, 'elliptic_txs_classes.csv')
    features_file = os.path.join(DATASET_DIR, 'elliptic_txs_features.csv')

    # Read data
    df_edges = pd.read_csv(edges_file).replace('-', '')
    df_classes = pd.read_csv(classes_file).replace({'-':'','suspicious':'1'})
    df_features = pd.read_csv(features_file, header=None)

    # Map class labels to 0, 1, 2
    df_classes['class'] = df_classes['class'].map({'unknown': 2, '1': 1, '2': 0})

    # Merge feature data and class information
    df_merge = df_features.merge(df_classes, how='outer', left_on=0, right_on="txId").reset_index(drop=True)
    features_tensor = torch.tensor(df_merge.drop('txId', axis=1).values, dtype=torch.float32)
    
    return features_tensor, df_edges

def get_edge_index(config, data: torch.FloatTensor, df_edges: pd.DataFrame):
    """
    Generate the edge index of a graph based on feature data and an edge list.

    Args:
        config: Configuration settings.
        data (torch.FloatTensor): Feature data.
        df_edges (pd.DataFrame): Edge list.
        
    Returns:
        torch.LongTensor: The edge index of the graph.
    """
    edges = df_edges.copy()

    # Create a node ID mapping
    id_mapping = {node_id: index for index, node_id in enumerate(data[:, 0].numpy().astype(int))}

    # Map the node IDs in the edge list
    edges['txId1'], edges['txId2'] = edges['txId1'].map(id_mapping), edges['txId2'].map(id_mapping)
    edges.dropna(inplace=True)
    edges = edges.astype(int)

    # Build the edge index
    edge_index = edges[['txId1', 'txId2']].values.T

    # If it is an undirected graph, add reverse edges
    if config["DATASET"]["UNDIRECTED"]:
        reversed_edge_index = np.flip(edge_index, axis=0)
        edge_index = np.concatenate((edge_index, reversed_edge_index), axis=1)
    return torch.tensor(edge_index, dtype=torch.long)

def load_dataset(config):
    """
    Retrieve data and relevant indices.

    Args:
        config (Namespace): Configuration parameters.

    Returns:
        torch.Tensor: The data.
        torch.Tensor: Edge indices.
        torch.Tensor: Indices of sup data.
        torch.Tensor: Indices of unsup data.
        torch.Tensor: Edge indices of unsup data (if un=True).
    """
    data, df_edges = load_elliptic_data(config)

    if config["TRAIN"]["STEP"] is not None:
        print(f"STEP IS NOT NONE!! \n SETTING STEP to {config['TRAIN']['STEP']}")
        step_idx = torch.nonzero(data[:, 1] == config["TRAIN"]["STEP"]).squeeze()
        data = data[step_idx, :]

    edge_index = get_edge_index(config, data, df_edges)
    sup_idx = torch.nonzero(data[:, -1] != 2).squeeze()
    unsup_idx = torch.nonzero(data[:, -1] == 2).squeeze()

    if config["DATASET"]["UNDIRECTED"]:
        un_edge = get_edge_index(config, data[unsup_idx], df_edges)
        return data, edge_index, sup_idx, unsup_idx, un_edge
    else:
        return data, edge_index, sup_idx, unsup_idx

def split_idx(config, data, sup_idx):
    """
    Splits the index into training and validation sets.
    """
    y_train = data.y[sup_idx]
    _, _, _, _, train_idx, valid_idx = train_test_split(data.x[sup_idx], y_train,
                                                        sup_idx, test_size=config["DATASET"]["TEST_RATIO"],
                                                        random_state=config["DATASET"]["SEED"], stratify=y_train)
    print('Get Dataset Ready: \n For Train shape {} & Valid shape {}'.format(len(train_idx), len(valid_idx)))
    return train_idx, valid_idx

def get_dataset(data, edge_index):
    """
    Constructs a graph data object.

    Args:
        data (torch.Tensor): The data.
        edge_index (torch.Tensor): The edge index.

    Returns:
        torch_geometric.data.Data: The graph data object.
    """
    X, Y = data[:, 2:], data[:, -1]
    input_data = Data(x=X, edge_index=edge_index, y=Y)
    return input_data

if __name__ == '__main__':
    from util import get_configs
    config = get_configs(config_dir="./configs/config.yaml")
    data, edge_index, sup_idx, unsup_idx = load_dataset(config)
    input_Data = get_dataset(data, edge_index)
    train_idx, valid_idx = split_idx(config, input_Data, sup_idx)
