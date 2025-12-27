import torch
import hashlib
import json
from typing import List, Set, Tuple


def node_coloring(obj) -> str:
    """
    a hashing for unique node coloring.... similar nodes get the same color
    Inspired by https://reinvantveer.github.io/2017/06/02/stable-dict-hashes.html
    """
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def normalize(v: torch.Tensor, eps: float = 1e-12):
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def get_invariants(rel_vecs: torch.Tensor):
    """
    Summarizes geometric information into distances and angles.
    """
    D = rel_vecs.norm(dim=-1)
    D = torch.sort(D).values
    D = [round(float(x), 6) for x in D.tolist()]

    # TODO: I should probably just add an if statement here for efficiency?
    #       This works but unnecessary computation for when there is only one neighbor.
    U = normalize(rel_vecs)
    C = (U @ U.t()).clamp(-1.0, 1.0)
    idx = torch.triu_indices(C.shape[0], C.shape[1], offset=1)
    A = C[idx[0], idx[1]]
    A = torch.sort(A).values
    A = [round(float(x), 6) for x in A.tolist()]

    return {"D": D, "A": A}


def init_base_colors(G) -> List[str]:
    """
    First iteration, initialization of the hashes
    """
    return [node_coloring({"deg": len(G.neighbors(i))}) for i in range(G.n)]


def _color_node(
    G,
    i: int,
    hop_sets: List[Set[int]],
    prev_colors: List[str],
) -> str:
    """
    does a node coloring by taking inti account neighbors as well.
    """
    rel = _gather_rel_vectors(G, i, hop_sets)
    signature = get_invariants(rel)
    signature["neigh_color_multiset"] = sorted(prev_colors[j] for j in G.neighbors(i))
    return node_coloring(signature)


def _gather_rel_vectors(G, i: int, hop_sets: List[Set[int]]) -> torch.Tensor:
    """
    Gets relative vectors to current node i.
    """
    idx = torch.tensor(sorted(hop_sets[i]), dtype=torch.long, device=G.X.device)
    return G.X.index_select(0, idx) - G.X[i].unsqueeze(0)


def init_hop_sets(G, num_iters: int, igwl: bool) -> Tuple[List[Set[int]], int]:
    """
    If igwl: immediate neighbirs, else: empty sets that will be gradually expanded
    """
    if igwl:
        hop_sets = [set(G.neighbors(i)) - {i} for i in range(G.n)]
        T = 1
    else:
        hop_sets = [set() for _ in range(G.n)]
        T = num_iters
    return hop_sets, T


def update_colors_once(
    G,
    hop_sets: List[Set[int]],
    prev_colors: List[str],
) -> List[str]:
    """
    one refinement of GWL
    """
    new_colors: List[str] = []
    for i in range(G.n):
        color_i = _color_node(
            G=G,
            i=i,
            hop_sets=hop_sets,
            prev_colors=prev_colors,
        )
        new_colors.append(color_i)
    return new_colors


def expand_hop_sets_once(G, hop_sets: List[Set[int]]) -> List[Set[int]]:
    """
    only called in GWL not IGWL.
    it grows its hop by adding its neighbors and its neighbors' neighbors.
    """
    new_hop_sets: List[Set[int]] = []
    for i in range(G.n):
        S = set(hop_sets[i])
        S.update(G.neighbors(i))
        for j in G.neighbors(i):
            S.update(hop_sets[j])
        S.discard(i)
        new_hop_sets.append(S)
    return new_hop_sets


if __name__ == "__main__":
    some_tensor = torch.randn(2, 3)
    normalized_tensor = normalize(some_tensor)
    print(normalized_tensor.shape)
    print(normalized_tensor.norm(dim=-1))
