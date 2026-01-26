import torch
import pandas as pd
import numpy as np

def to_pyg_data(nodes_df: pd.DataFrame, edges_df: pd.DataFrame):
    """
    Convert NGraph DataFrames to a PyTorch Geometric Data object.
    
    Currently assumes 'id' in nodes_df and 'source', 'target' in edges_df.
    """
    try:
        from torch_geometric.data import Data
    except ImportError:
        print("torch_geometric not found. Returning a simple dictionary of tensors.")
        Data = dict

    # 1. Edge Index
    # We need to map global Node IDs to local 0..N indices for PyG
    id_map = {node_id: i for i, node_id in enumerate(nodes_df['id'])}
    
    src = edges_df['source'].map(id_map).values
    dst = edges_df['target'].map(id_map).values
    
    edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)

    # 2. Node Features
    # Placeholder: currently we just have node IDs and labels.
    # In a real scenario, we'd parse embeddings from properties.
    x = torch.ones((len(nodes_df), 1)) # Dummy features

    return Data(x=x, edge_index=edge_index)
