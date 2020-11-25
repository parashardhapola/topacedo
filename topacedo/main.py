import numpy as np
from numba import jit, float64, int64
from scipy.sparse import csr_matrix
import pandas as pd
from typing import List, Tuple, Union
from scipy.sparse.csgraph import connected_components
import pcst_fast


__all__ = ['TopacedoSampler', 'calc_neighbourhood_density', 'calc_indegree']


@jit(float64[:](int64[:], int64[:], float64[:], float64[:]), nopython=True)
def _indegree_calculator(ai: np.ndarray, aj: np.ndarray, v: np.ndarray, a: np.ndarray) -> np.ndarray:
    n_cells = a.shape[0]
    for i in range(n_cells):
        for p in range(ai[i], ai[i + 1]):
            j = aj[p]
            a[j] += v[p]
    return a


def calc_indegree(g: csr_matrix) -> np.ndarray:
    """
    Calculate indegree of each node in a graph

    Args:
        g: csr_matrix representing a graph

    Returns: Indegree of each node of a graph as a numpy vector
    """
    return _indegree_calculator(
        g.indptr.astype(np.int64), g.indices.astype(np.int64),
        g.data.astype(np.float64), np.zeros(g.shape[0]))


@jit(float64[:](int64[:], int64[:], float64[:]), nopython=True)
def _density_calculator(ai: np.ndarray, aj: np.ndarray, v: np.ndarray) -> np.ndarray:
    n_cells = v.shape[0]
    if n_cells != ai.shape[0] - 1:
        raise ValueError("ERROR: Vector size mismatch in `_iter_graph`. This is a bug. Please report.")
    a = np.zeros(n_cells)
    for i in range(n_cells):
        for p in range(ai[i], ai[i + 1]):
            j = aj[p]
            a[i] += v[j]
    return a + v


def calc_neighbourhood_density(g: csr_matrix, search_depth: int) -> np.ndarray:
    """
    Calculates the density of each node in the cell-cell neighbourhood graph.

    Args:
        g: csr_matrix representing a non-directed graph
        search_depth: If the value of `search_depth` is 0, then node density is simply the indegree of each node
                      in the graph. For values greater than 1 node density represents neighbourhood degree. For example,
                      when `search_depth` = 1, node density of a node is the sum of indegree of all its adjacent nodes
                      i.e. nodes that are at distance of 1. With increasing value of `neighbourhood_degree`, neighbours
                      further away from each node are also included included.

    Returns:

    """

    d = calc_indegree(g)
    ai, aj = g.indptr.astype(np.int64), g.indices.astype(np.int64)
    for n in range(search_depth):
        d = _density_calculator(ai, aj, d)
    return d


def _min_max_norm(x: Union[pd.Series, np.ndarray]) -> Union[pd.Series, np.ndarray]:
    """
    Min-max normalization of a vector
    Args:
        x: Values

    Returns:

    """
    minx = x.min()
    maxx = x.max()
    return (x - minx) / (maxx - minx)


def calc_dsra(gi: np.ndarray, d: np.ndarray, min_v: float) -> pd.Series:
    """
    Calculates Density based sampling rate adjuster (DSRA) by computing
     median neighbourhood densities of cells within each group

    Args:
        gi: Group values. For example, cluster identities of a cell
        d: Cell density as calculated with  `calc_neighbourhood_density` function in topacedo
        min_v: minimum value for sampling rate.
    Returns: Values between 0 and 1

    """
    # Calculate the median density with in each group/cluster
    dm = pd.DataFrame(
        {'groups': gi, 'density': d},
    ).groupby('groups').median()['density']
    # Calculate sampling rate.
    # Please note that atleast one group will have a value of zero
    sr = _min_max_norm(dm)
    sr[sr < min_v] = min_v
    sr.index = sr.index.astype(gi.dtype)
    return sr


def assign_seed_status(gi: np.ndarray, sra: pd.Series,
                       sr: float, min_c: float, rand_s: int) -> list:
    """

    Args:
        gi: Group values. For example, cluster identities of a cell
        sra: Sampling rate adjuster as obtained from `calc_dsra`
        sr: Sampling rate i.e. fraction of cells to sample from each group as seeds.
            This value is multiplied with `sra` to get the effective sampling rate for each group
        min_c: Minimum number of cells to sample from each group
        rand_s: Random seed for reproducibility.

    Returns: A list containing indices of cells that were assigned seed node status

    """
    sn = []
    gi = pd.Series(gi)
    # Iterate over individual groups
    for i in gi.unique():
        # Get cells from each group. Here we are mostly interested in the index of c.
        # But we keep in pd.Series form here so as to use its sample method later
        c = gi[gi == i]
        if len(c) > min_c:
            # Set number of cells to be sampled the group. Should be atleast min_c
            n = max(min_c, int(len(c)*sr*sra[i]))
            s = c.sample(n=n, random_state=rand_s).index
        else:
            s = c.index
        sn.extend(s)
    return sn


class TopacedoSampler:

    def __init__(self, graph: csr_matrix, groups: np.ndarray,
                 density_depth: int, sampling_rate: float, min_cells: int, min_sr: float,
                 seed_reward: float, non_seed_reward: float, edge_cost_factor: float,
                 rand_seed: int):
        """

        Args:
            graph: CSR matrix representing a cell-cell neighbourhood graph
            groups: Group identity of each cell. For example, cluster identity
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`
            sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                           than this value depending on the neighbourhood density of the cells.
            min_cells: Minimum number of cells to sample from each group
            min_sr: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this value.
            seed_reward: Reward/prize value for seed nodes
            non_seed_reward: Reward/prize for non-seed nodes
            edge_cost_factor:
            rand_seed: Seed for setting random state

        Returns:

        """

        self.graph = graph
        self.groups = groups
        self.seedPrize, self.nonSeedPrize = seed_reward, non_seed_reward
        self.edgeCostFactor = edge_cost_factor
        self.densities = calc_neighbourhood_density(graph, density_depth)
        self.dsra = calc_dsra(groups, self.densities, min_sr)
        self.seeds = assign_seed_status(groups, self.dsra, sampling_rate, min_cells, rand_seed)
        # Identify the connected component of which each cell is a part
        _, self.components = connected_components(graph)

    def _run_pcst_on_component(self, seeds, component):
        idx = np.where(self.components == component)[0]
        subgraph = self.graph[idx].T[idx].tocoo()  # subgraph of the component
        costs = self.edgeCostFactor * ((1 + subgraph.data.min()) - subgraph.data)  # Edge penalties/cost
        prizes = [self.seedPrize if x in seeds else self.nonSeedPrize for x in idx]
        edges = np.vstack([subgraph.row, subgraph.col]).T
        root, num_clusters, pruning, verbosity = -1, 1, 'strong', 0
        x, y = pcst_fast.pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity)
        # 'root' was hard set to -1 because there is not root node in the graph.
        # 'num_clusters' was set to 1 because we want a fully connected sampling of the component
        return idx[x], [[idx[x[0]], idx[x[1]]] for x in edges[y]]

    def run(self) -> Tuple[List, List]:
        seeds = {x: None for x in self.seeds}
        sampled_cells, sampled_edges = [], []
        for component in set(self.components):
            sc, se = self._run_pcst_on_component(seeds, component)
            sampled_cells.extend(sc)
            sampled_edges.extend(se)
        cover = set(sampled_cells).intersection(self.seeds)
        seed_diff = len(seeds) - len(cover)
        if seed_diff != 0:
            print(f"WARNING: {seed_diff} seeds not present in downsampled set. Try increasing the reward for seeds",
                  flush=True)
        down_ratio = 100 * len(sampled_cells) / self.graph.shape[0]
        down_ratio = "%.2f" % down_ratio

        seed_ratio = len(sampled_cells) / len(seeds)
        if seed_ratio > 2 and self.nonSeedPrize > 0:
            print(f"WARNING: High subsample to seed ratio detected. Try decreasing the non-seed reward", flush=True)
        seed_ratio = "%.3f" % seed_ratio

        print(f"INFO: {len(sampled_cells)} cells ({down_ratio}%) sub-sampled. "
              f"Subsample to Seed ({len(seeds)} cells) ratio: {seed_ratio}", flush=True)
        return sampled_cells, sampled_edges
