import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd
from typing import List, Tuple, Union
from scipy.sparse.csgraph import connected_components
from .density import calc_neighbourhood_density
from .recluster import make_digraph, recluster
try:
    import pcst_fast
except ImportError:
    raise ImportError('ERROR: PCST is not installed. Install using the following command:\n\t'
                      ' pip install https://github.com/fraenkel-lab/pcst_fast/tarball/master#egg=pcst_fast-1.0.7')

__all__ = ['TopacedoSampler']


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


def calc_group_median(gi: np.ndarray, v: np.ndarray, bw: float, sign: int):
    gm = pd.DataFrame(
        {'g': gi, 'v': v},
    ).groupby('g').median()['v']
    res = bw ** (sign * _min_max_norm(gm))
    res.index = res.index.astype(gi.dtype)
    return res


def calc_mean_snn(graph: csr_matrix) -> np.ndarray:
    n = len(graph[0].indices) - 1
    snn = []
    indices = [set(graph[x].indices) for x in range(graph.shape[0])]
    for i in range(graph.shape[0]):
        snn.extend([len(indices[i].intersection(indices[x])) / n for x in indices[i]])
    snn = np.array(snn)
    snn_g = graph.copy()
    snn_g.data = snn
    return np.array(snn_g.sum(axis=1).reshape(1, -1))[0]


def assign_seed_status(gi: np.ndarray, dsra: pd.Series, ssra: np.array,
                       sr: float, min_c: float, min_sr: float, dend_graph, rand_s: int) -> list:
    """

    Args:
        gi: Group values. For example, cluster identities of a cell
        dsra: Sampling rate adjuster
        ssra:
        sr: Sampling rate i.e. fraction of cells to sample from each group as seeds.
            This value is multiplied with `sra` to get the effective sampling rate for each group
        min_c: Minimum number of cells to sample from each group
        min_sr:
        dend_graph:
        rand_s: Random seed for reproducibility.

    Returns: A list containing indices of cells that were assigned seed node status

    """
    sn = []
    gi = pd.Series(gi)
    # Iterate over individual groups
    rng = np.random.default_rng(rand_s)
    for i in sorted(gi.unique()):
        # Get cells from each group. Here we are mostly interested in the index of c.
        # But we keep in pd.Series form here so as to use its sample method later
        c = gi[gi == i]
        if len(c) > min_c:
            # Effective sampling rate
            esr = max(min_sr, sr*dsra[i]*ssra[i])
            if esr > sr:
                esr = sr
            # Set number of cells to be sampled the group. Should be atleast min_c
            n = max(min_c, int(len(c)*esr))
            subclusts = recluster(dend_graph, np.where(gi.values == i)[0], n)
            s = [rng.choice(x, size=1)[0] for x in subclusts]
            # s = c.sample(n=n, random_state=rand_s).index
        else:
            s = c.index
        sn.extend(s)
    return sorted(sn)


class TopacedoSampler:

    def __init__(self, graph: csr_matrix, groups: np.ndarray, groups_dendrogram,
                 density_depth: int, density_bandwidth: float,
                 max_sampling_rate: float, min_sampling_rate: float, min_cells: int,
                 snn_bandwidth: float, seed_reward: float, non_seed_reward: float,
                 edge_cost_multiplier: float, edge_cost_bandwidth: float, rand_seed: int):
        """

        Args:
            graph: CSR matrix representing a cell-cell neighbourhood graph
            groups: Group identity of each cell. For example, cluster identity
            groups_dendrogram:
            density_depth: Same as 'search_depth' parameter in `calc_neighbourhood_density`
            max_sampling_rate: Maximum fraction of cells to sample from each group. The effective sampling rate is lower
                           than this value depending on the neighbourhood density of the cells.
            min_cells: Minimum number of cells to sample from each group
            min_sampling_rate: Minimum sampling rate. Effective sampling rate is not allowed to be lower than this
                               value.
            seed_reward: Reward/prize value for seed nodes
            non_seed_reward: Reward/prize for non-seed nodes
            edge_cost_multiplier:
            edge_cost_bandwidth:
            rand_seed: Seed for setting random state

        Returns:

        """

        self.graph = graph
        self.groups = groups
        self.seedPrize, self.nonSeedPrize = seed_reward, non_seed_reward
        self.edgeCostMultiplier = edge_cost_multiplier
        self.edgeCostBandwidth = edge_cost_bandwidth
        self.densities = calc_neighbourhood_density(graph, density_depth)
        self.dsra = calc_group_median(groups, self.densities, density_bandwidth, -1)
        self.meanSnn = calc_mean_snn(self.graph)
        self.ssra = calc_group_median(groups, self.meanSnn, snn_bandwidth, 1)
        self.seeds = assign_seed_status(groups, self.dsra, self.ssra, max_sampling_rate, min_cells,
                                        min_sampling_rate, make_digraph(groups_dendrogram), rand_seed)
        # Identify the connected component of which each cell is a part
        _, self.components = connected_components(graph)

    def _run_pcst_on_component(self, seeds, component):
        idx = np.where(self.components == component)[0]
        subgraph = self.graph[idx].T[idx].tocoo()  # subgraph of the component
        costs = self.edgeCostMultiplier * (self.edgeCostBandwidth ** (-subgraph.data))  # Edge penalties/cost
        prizes = [self.seedPrize if x in seeds else self.nonSeedPrize for x in idx]
        edges = np.vstack([subgraph.row, subgraph.col]).T
        root, num_clusters, pruning, verbosity = -1, 1, 'strong', 0
        x, y = pcst_fast.pcst_fast(edges, prizes, costs, root, num_clusters, pruning, verbosity)
        # 'root' was hard set to -1 because there is no root node in the graph.
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
