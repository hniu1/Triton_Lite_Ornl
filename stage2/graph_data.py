"""Raster-verified tile adjacency and synchronized neighborhood batches."""

import hashlib

import numpy as np
import torch
from torch.utils.data import Dataset, default_collate


class TileTopology:
    """Node IDs are dataset block positions, not raw raster block labels."""

    def __init__(self, block_labels, contacts):
        self.block_labels = tuple(int(label) for label in block_labels)
        if len(set(self.block_labels)) != len(self.block_labels):
            raise ValueError("Duplicate block labels")
        lookup = {label: index for index, label in enumerate(self.block_labels)}
        neighbors = [set() for _ in self.block_labels]
        for left, right in contacts:
            if left not in lookup or right not in lookup:
                raise ValueError("Contact references an unknown block label")
            source, destination = lookup[left], lookup[right]
            if source == destination:
                continue
            neighbors[source].add(destination)
            neighbors[destination].add(source)
        self.neighbors = tuple(tuple(sorted(values)) for values in neighbors)
        edges = [(source, destination) for source, values in enumerate(self.neighbors)
                 for destination in values]
        self.edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        digest = hashlib.sha256()
        digest.update(np.asarray(self.block_labels, dtype="<i8").tobytes())
        digest.update(self.edge_index.numpy().astype("<i8").tobytes())
        self.digest = digest.hexdigest()

    @classmethod
    def from_raster(cls, grid, block_labels, chunk_rows=256):
        if grid.ndim != 2 or chunk_rows < 1:
            raise ValueError("Expected a 2D block grid and positive chunk size")
        labels = set(int(label) for label in block_labels)
        if not labels or min(labels) < 0:
            raise ValueError("Block labels must be nonnegative")
        contacts, observed = set(), set()
        for start in range(0, grid.shape[0], chunk_rows):
            stop = min(start + chunk_rows, grid.shape[0])
            patch = np.asarray(grid[start:stop])
            observed.update(int(value) for value in np.unique(patch) if value >= 0)
            pairs = [(patch[:, :-1], patch[:, 1:])]
            if stop < grid.shape[0]:
                vertical = np.asarray(grid[start:stop + 1])
            else:
                vertical = patch
            pairs.append((vertical[:-1], vertical[1:]))
            for left, right in pairs:
                valid = (left != right) & (left >= 0) & (right >= 0)
                if np.any(valid):
                    unique = np.unique(np.stack([left[valid], right[valid]], axis=1), axis=0)
                    contacts.update(tuple(sorted((int(a), int(b)))) for a, b in unique)
        if observed != labels:
            raise ValueError(f"Raster/lookup mismatch: missing={sorted(labels-observed)[:8]}, "
                             f"unknown={sorted(observed-labels)[:8]}")
        return cls(block_labels, contacts)

    def neighborhood(self, centers, hops):
        if hops < 0 or not centers:
            raise ValueError("Need center tiles and nonnegative hops")
        if any(center < 0 or center >= len(self.neighbors) for center in centers):
            raise ValueError("Unknown center tile")
        selected = set(centers)
        frontier = set(centers)
        for _ in range(hops):
            frontier = {neighbor for node in frontier for neighbor in self.neighbors[node]} - selected
            selected.update(frontier)
        nodes = sorted(selected)
        mapping = {node: index for index, node in enumerate(nodes)}
        edges = [(mapping[source], mapping[destination]) for source in nodes
                 for destination in self.neighbors[source] if destination in selected]
        edge_index = torch.tensor(edges, dtype=torch.long).reshape(-1, 2).T.contiguous()
        truncated = torch.tensor([any(n not in selected for n in self.neighbors[node])
                                  for node in nodes], dtype=torch.bool)
        return nodes, edge_index, truncated

    def metadata(self):
        return dict(nodes=len(self.neighbors), directed_edges=self.edge_index.shape[1],
                    isolated_nodes=sum(not neighbors for neighbors in self.neighbors),
                    topology_sha256=self.digest, adjacency="four-neighbor raster label contact",
                    node_index="dataset block position", block_labels=list(self.block_labels))


class GraphNeighborhoodDataset(Dataset):
    """A sample is an entire connected batch around a list of center keys.

    Use DataLoader(batch_size=None, sampler=RandomLeadTimeBatchSampler(...)).
    All center keys must share an event, target time, and lead. Context target
    labels are loaded by the existing adapter but never supplied to the model
    or included in loss/metrics.
    """

    def __init__(self, direct_dataset, topology, hops):
        self.direct_dataset = direct_dataset
        self.topology = topology
        self.hops = int(hops)
        labels = tuple(int(value) for value in direct_dataset.block_rows["block_index"])
        if labels != topology.block_labels or hops < 1:
            raise ValueError("Topology does not match dataset or hops is not positive")

    def __getattr__(self, name):
        if name == "direct_dataset":
            raise AttributeError(name)
        return getattr(self.direct_dataset, name)

    def __len__(self):
        return len(self.direct_dataset)

    def __getitem__(self, keys):
        return self.make_batch(keys)

    def make_batch(self, keys):
        if not keys or any(len(key) != 4 for key in keys):
            raise ValueError("Expected a nonempty list of direct forecast keys")
        groups = {(int(event), int(target), int(lead)) for event, target, _, lead in keys}
        if len(groups) != 1:
            raise ValueError("Graph batch must share event, target and lead")
        event, target, lead = next(iter(groups))
        centers = {int(key[2]) for key in keys}
        nodes, edges, truncated = self.topology.neighborhood(centers, self.hops)
        batch = default_collate([self.direct_dataset[(event, target, node, lead)] for node in nodes])
        batch["graph_edge_index"] = edges
        batch["graph_group_ids"] = torch.zeros(len(nodes), dtype=torch.long)
        batch["graph_center_mask"] = torch.tensor([node in centers for node in nodes])
        batch["graph_block_positions"] = torch.tensor(nodes, dtype=torch.long)
        batch["graph_context_truncated"] = truncated
        return batch


def select_center_forecasts(output, batch):
    """Keep only designated centers for both loss and numerical metrics."""
    selection = batch.get("graph_center_mask")
    if selection is None:
        return output, batch
    count = selection.numel()
    selected_batch = {
        name: value[selection] if isinstance(value, torch.Tensor) and value.ndim
        and value.shape[0] == count else value
        for name, value in batch.items() if not name.startswith("graph_")
    }
    return tuple(value[selection] if value is not None else None for value in output), selected_batch
