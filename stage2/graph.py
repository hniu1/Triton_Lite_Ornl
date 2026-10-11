"""Optional tile message passing; messages are not conserved hydraulic fluxes."""

import torch
from torch import nn


class TileGraphCommunication(nn.Module):
    """Directed, mean-aggregated messages on synchronized tile neighborhoods.

    ``edge_index[0]`` is source and ``edge_index[1]`` is destination. Callers
    must assign a distinct group to each (event, origin, lead) combination.
    No self edges are needed: each node has a residual update.
    """

    def __init__(self, channels, edge_features=0, layers=2):
        super().__init__()
        if channels < 1 or edge_features < 0 or layers < 1:
            raise ValueError("Invalid graph dimensions")
        self.edge_features = edge_features
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * channels + edge_features, channels),
                          nn.GELU(), nn.Linear(channels, channels))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * channels, channels), nn.GELU(),
                          nn.Linear(channels, channels))
            for _ in range(layers)
        ])

    def forward(self, nodes, edge_index, group_ids, edge_attributes=None):
        if nodes.ndim != 2:
            raise ValueError("nodes must be [tiles, channels]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2 or edge_index.dtype != torch.long:
            raise ValueError("edge_index must be int64 [2, edges]")
        if group_ids.shape != (nodes.shape[0],) or group_ids.dtype != torch.long:
            raise ValueError("group_ids must be int64 [tiles]")
        if edge_index.device != nodes.device or group_ids.device != nodes.device:
            raise ValueError("Graph tensors must share the node device")
        if edge_index.numel() and (edge_index.min() < 0 or edge_index.max() >= nodes.shape[0]):
            raise ValueError("Edge references a missing tile")
        source, destination = edge_index
        if torch.any(group_ids[source] != group_ids[destination]):
            raise ValueError("Edges cannot cross event/origin/lead groups")
        if self.edge_features:
            if edge_attributes is None or edge_attributes.shape != (source.numel(), self.edge_features):
                raise ValueError("Missing or incorrectly shaped edge attributes")
            if edge_attributes.device != nodes.device or edge_attributes.dtype != nodes.dtype:
                raise ValueError("Edge attributes must share node device and dtype")
        elif edge_attributes is not None:
            raise ValueError("This graph is configured without edge attributes")
        if source.numel() == 0:
            return nodes
        degree = nodes.new_zeros((nodes.shape[0], 1))
        degree.index_add_(0, destination, nodes.new_ones((source.numel(), 1)))
        for message, update in zip(self.messages, self.updates):
            inputs = [nodes[source], nodes[destination]]
            if self.edge_features:
                inputs.append(edge_attributes)
            values = message(torch.cat(inputs, dim=1))
            aggregated = torch.zeros_like(nodes)
            aggregated.index_add_(0, destination, values)
            aggregated = aggregated / degree.clamp_min(1)
            change = update(torch.cat([nodes, aggregated], dim=1))
            nodes = nodes + change * (degree > 0).to(nodes.dtype)
        return nodes
