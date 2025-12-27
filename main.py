from utils import (
    init_base_colors,
    init_hop_sets,
    update_colors_once,
    expand_hop_sets_once,
)
import torch

from collections import Counter
from typing import List


class GeometricGraph:
    def __init__(self, X: torch.Tensor, edges: List[tuple[int, int]]):
        """
        X contains the nodes of the graph... connected by edges.
        """
        self.X = X.detach().clone().float()
        self.n, self.d = self.X.shape
        self.N = [[] for _ in range(self.n)]
        for u, v in edges:
            self.N[u].append(v)
            self.N[v].append(u)

    def neighbors(self, i):
        return self.N[i]


def gwl(
    G,
    num_iters: int = 2,
    igwl: bool = False,
) -> List[str]:
    """
    Does 2 itirations of the gwl test.
    TODO: gonna have to add a better stop criterion than max_iter. This will work for my simple example though.
    """

    colors = init_base_colors(G)
    hop_sets, T = init_hop_sets(G, num_iters, igwl)

    for _ in range(T):
        if not igwl:
            hop_sets = expand_hop_sets_once(G, hop_sets)
        colors = update_colors_once(
            G=G,
            hop_sets=hop_sets,
            prev_colors=colors,
        )
    return colors


if __name__ == "__main__":
    edges = [(0, 1), (1, 2), (2, 3)]

    XA = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [2.0, 1.0]], dtype=torch.float32
    )

    XB = torch.tensor(
        [[-1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [2.0, -1.0]], dtype=torch.float32
    )

    GA = GeometricGraph(XA, edges)
    GB = GeometricGraph(XB, edges)

    colsA_igwl = gwl(GA, igwl=True)
    colsB_igwl = gwl(GB, igwl=True)

    igwl_is_isometric = Counter(colsA_igwl) == Counter(colsB_igwl)
    print("IGWL: Isometric Graphs?", igwl_is_isometric)
    print("IGWL test failed") if igwl_is_isometric else print("IGWL test passed")
    print("__" * 10)
    colsA_gwl2 = gwl(GA, num_iters=2, igwl=False)
    colsB_gwl2 = gwl(GB, num_iters=2, igwl=False)
    gwl_is_isometric = Counter(colsA_gwl2) == Counter(colsB_gwl2)
    print("GWL: Isometric Graphs?", gwl_is_isometric)
    print("GWL test failed") if gwl_is_isometric else print("GWL test passed")
