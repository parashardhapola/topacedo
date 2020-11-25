import numpy as np
from numba import jit, float64, int64
from scipy.sparse import csr_matrix
import pandas as pd
from typing import List, Tuple, Dict
from scipy.sparse.csgraph import connected_components
import pcst_fast
from tqdm import tqdm


__all__ = ['topacedo_sampler', 'calc_neighbourhood_density', 'calc_indegree']


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


def calc_neighbourhood_density(g: csr_matrix, neighborhood_depth: int) -> np.ndarray:
    """
    Calculates the density of each node in the cell-cell neighbourhood graph. If the value of `neighbourhood_degree`
    is 0, then node density is simply the indegree of each node in the cell-cell neighbourhood graph (KNN graph).
    For values greater than 1 node density represents neighbourhood degree. For example, when
    `neighbourhood_degree` = 1, node density of a node is the sum of indegree of all its adjacent nodes i.e. nodes that
    are at distance of 1. With increasing value of `neighbourhood_degree`, neighbours further away from each node
    are also included included.
    Args:
        g: csr_matrix representing a non-directed graph
        neighborhood_depth:

    Returns:

    """

    d = calc_indegree(g)
    ai, aj = g.indptr.astype(np.int64), g.indices.astype(np.int64)
    for n in range(neighborhood_depth):
        d = _density_calculator(ai, aj, d)
    return d


def get_seed_nodes(clusts: pd.Series, frac: float, cff: pd.Series,
                   min_nodes: int, rand_num: int) -> Dict[int, None]:
    seeds = []
    for i in clusts.unique():
        c = clusts[clusts == i]
        if len(c) > min_nodes:
            s = c.sample(frac=min(1, frac+frac*cff[i]), random_state=rand_num).index
            if len(s) < min_nodes:
                s = c.sample(n=min_nodes, random_state=rand_num).index
        else:
            s = c.index
        seeds.extend(s)
    return {x: None for x in seeds}


def topacedo_sampler(graph: csr_matrix, clusters: pd.Series, seed_frac: float,
                 cluster_factor: pd.Series, min_nodes: int, rewards: Tuple[float, float],
                 pruning_method: str, rand_state: int) -> Tuple[List, List]:

    ss, se = [], []
    _, l = connected_components(graph)
    seeds = get_seed_nodes(clusters, seed_frac, cluster_factor, min_nodes, rand_state)
    print(f"INFO: {len(seeds)} seed cells selected", flush=True)
    for i in set(l):
        idx = np.where(l == i)[0]
        g = graph[idx].T[idx].tocoo()
        c = (1 + g.data.min()) - g.data
        r = [rewards[0] if x in seeds else rewards[1] for x in idx]
        e = np.vstack([g.row, g.col]).T
        x, y = pcst_fast.pcst_fast(e, r, c, -1, 1, pruning_method, 0)
        ss.extend(idx[x])
        se.extend([[idx[x[0]], idx[x[1]]] for x in e[y]])
    cover = set(ss).intersection(list(seeds.keys()))
    if len(cover) != len(seeds):
        print(f"WARNING: Not all seed cells in downsampled data. Try increasing the reward for seeds", flush=True)
    seed_ratio = len(ss) / len(seeds)
    if seed_ratio > 2 and rewards[1] > 0:
        print(f"WARNING: High seed ratio detected. Try decreasing the non-seed reward", flush=True)
    print(f"INFO: {len(ss)} cells sub-sampled. {len(cover)} of which are present in seed list.", flush=True)
    down_ratio = 100 * len(ss)/graph.shape[0]
    down_ratio = "%.2f" % down_ratio
    print(f"INFO: Downsampling percentage: {down_ratio}%", flush=True)
    seed_ratio = "%.3f" % seed_ratio
    print(f"INFO: Seed ratio: {seed_ratio}", flush=True)
    return ss, se
