import numpy as np
from numba import jit, float64, int64
from scipy.sparse import csr_matrix

__all__ = ['calc_neighbourhood_density']


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
