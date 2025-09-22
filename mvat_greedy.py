
#!/usr/bin/env python3
"""
Greedy MVAT (Most Vulnerable Attack Trace) augmentation on top of SAT.

Idea:
1) Build an initial SAT trace T (DAG) using weights w (AND=max, OR=min).
2) Under independence, compute P(g) on T using node/edge probabilities p.
3) Greedily add extra incoming edges to OR nodes (including goal) to increase P(g),
   while keeping the trace acyclic. If a candidate parent u is not yet in T,
   we attach its own minimal SAT-subtrace (to make u "provable") before linking u->d.
4) Repeat until no positive gain.

Notes:
- This is a heuristic (Phase 2). It doesn't guarantee global optimality, but is simple and effective.
- Cycle check: we forbid adding (u->d) if d can already reach u in the current trace.
- We cache SAT subtraces to arbitrary targets u to avoid recomputation.

Usage:
  python3 mvat_greedy.py graph_prob_mv.json --pretty
  python3 mvat_greedy.py graph_prob_mv.json --iters 5 --eps 1e-9
"""

import json, math, sys, heapq, copy
from collections import defaultdict, deque

WHITE, GRAY, BLACK = 0, 1, 2

class Node:
    __slots__ = ("id","typ","w","p","in_deg","done","height","color","pred")
    def __init__(self, id, typ, w, p):
        self.id = id
        self.typ = typ  # 'p', 'r', 'd', 'g'
        self.w = float(w)
        self.p = float(p)  # success prob for node
        self.in_deg = 0
        self.done = 0
        # OR/g starts at +inf; AND starts at -inf; p-node starts at w(v)
        if typ in ("d", "g"):
            self.height = math.inf
        elif typ == "r":
            self.height = -math.inf
        elif typ == "p":
            self.height = self.w
        else:
            raise ValueError(f"Unknown node type {typ}")
        self.color = WHITE
        self.pred = None  # best predecessor id for OR/g (for SAT backtrack)

def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = {n["id"]: Node(n["id"], n["type"], n.get("w", 0.0), n.get("p", 1.0)) for n in data["nodes"]}
    out_edges = defaultdict(list)  # u -> list of (v, w_uv, p_uv)
    in_edges = defaultdict(list)   # v -> list of (u, w_uv, p_uv)
    edge_prob = {}                 # (u,v) -> p(u,v)
    for e in data["edges"]:
        if len(e) == 3:
            u, v, w = e
            p_uv = 1.0
        else:
            u, v, w, p_uv = e
        out_edges[u].append((v, float(w), float(p_uv)))
        in_edges[v].append((u, float(w), float(p_uv)))
        nodes[v].in_deg += 1
        edge_prob[(u,v)] = float(p_uv)
    goal = data["goal"]
    if goal not in nodes:
        raise ValueError("Goal node not found in nodes")
    return nodes, out_edges, in_edges, edge_prob, goal

def sat_to_goal(nodes_in, out_edges, in_edges, goal):
    """Run SAT to a given goal node (d/g/r is allowed; goal must be reachable).
       Returns (ok, trace_dict) where trace_dict contains 'nodes','edges','height'.
    """
    # deepcopy nodes because SAT mutates fields (height/color/pred/done)
    nodes = {k: copy.copy(v) for k, v in nodes_in.items()}
    pq = []; counter = 0
    for v in nodes.values():
        # reset SAT fields
        v.in_deg = len(in_edges.get(v.id, []))
        v.done = 0
        v.color = WHITE
        v.pred = None
        if v.typ == "p":
            v.height = v.w
            v.color = GRAY
            heapq.heappush(pq, (v.height, counter, v.id)); counter += 1
        elif v.typ == "r":
            v.height = -math.inf
        elif v.typ in ("d","g"):
            v.height = math.inf

    while pq:
        h, _, uid = heapq.heappop(pq)
        u = nodes[uid]
        if u.color == BLACK:
            continue
        u.color = BLACK
        if uid == goal:
            return True, reconstruct(nodes, in_edges, goal)

        for vid, w_uv, p_uv in out_edges.get(uid, ()):
            v = nodes[vid]
            temp = u.height + w_uv + v.w
            if v.typ in ("d","g"):  # OR
                if temp < v.height:
                    v.height = temp
                    v.pred = uid
                if v.color == WHITE:
                    v.color = GRAY
                    heapq.heappush(pq, (v.height, counter, vid)); counter += 1
                elif v.color == GRAY:
                    heapq.heappush(pq, (v.height, counter, vid)); counter += 1
            elif v.typ == "r":     # AND
                if temp > v.height:
                    v.height = temp
                v.done += 1
                if v.done == v.in_deg and v.color == WHITE:
                    v.color = GRAY
                    heapq.heappush(pq, (v.height, counter, vid)); counter += 1
            else:
                raise ValueError(f"Unknown node type: {v.typ}")

    return False, {"height": None, "nodes": [], "edges": []}

def reconstruct(nodes, in_edges, goal):
    T_nodes = set()
    T_edges = []
    q = deque([goal])
    while q:
        v = q.popleft()
        if v in T_nodes: 
            continue
        T_nodes.add(v)
        nv = nodes[v]
        if nv.typ in ("d","g"):  # OR
            if nv.pred is not None:
                T_edges.append((nv.pred, v))
                q.append(nv.pred)
        elif nv.typ == "r":      # AND
            for pu, _, _ in in_edges[v]:
                T_edges.append((pu, v))
                q.append(pu)
        elif nv.typ == "p":
            pass
    return {"height": nodes[goal].height, "nodes": sorted(T_nodes), "edges": T_edges}

def compute_probability_of_trace(trace, nodes, edge_prob):
    """Topo DP on the given DAG trace to compute P(g)."""
    preds = defaultdict(list); children = defaultdict(list); indeg = {}
    for u, v in trace["edges"]:
        preds[v].append(u); children[u].append(v)
    for v in trace["nodes"]:
        indeg[v] = len(preds[v])
    q = deque([v for v in trace["nodes"] if indeg[v] == 0])
    P = {}
    while q:
        v = q.popleft()
        nv = nodes[v]
        if nv.typ == "p":
            P[v] = nv.p
        elif nv.typ == "r":
            if len(preds[v]) == 0:
                P[v] = 0.0
            else:
                prod = 1.0
                for u in preds[v]:
                    prod *= P.get(u, 0.0) * edge_prob.get((u,v), 1.0)
                P[v] = nv.p * prod
        elif nv.typ in ("d","g"):
            if len(preds[v]) == 0:
                P[v] = 0.0
            else:
                fail = 1.0
                for u in preds[v]:
                    fail *= (1.0 - P.get(u, 0.0) * edge_prob.get((u,v), 1.0))
                P[v] = nv.p * (1.0 - fail)
        else:
            P[v] = 0.0
        for w in children[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)
    # find goal
    g = None
    for v in trace["nodes"]:
        if nodes[v].typ == "g":
            g = v; break
    return P.get(g, 0.0), P

def has_path(children, src, dst):
    """Is dst reachable from src in the current DAG (children map)?"""
    if src == dst:
        return True
    seen = set([src])
    dq = deque([src])
    while dq:
        x = dq.popleft()
        for y in children.get(x, []):
            if y == dst:
                return True
            if y not in seen:
                seen.add(y); dq.append(y)
    return False

def greedy_augment(nodes, out_edges, in_edges, edge_prob, base_trace, max_iters=100, eps=1e-12):
    """Greedily add OR parents to increase P(g). Returns augmented trace and history."""
    # Current trace state
    trace_nodes = set(base_trace["nodes"])
    trace_edges = set(base_trace["edges"])
    history = []

    # Cache SAT subtraces to intermediate targets
    subtrace_cache = {}

    # Precompute incoming from full graph
    in_all = in_edges

    # Main loop
    it = 0
    while it < max_iters:
        it += 1
        # Compute current prob and adjacency
        current_trace = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
        Pg, Pnode = compute_probability_of_trace(current_trace, nodes, edge_prob)

        # Build children map for cycle checks
        children = defaultdict(list)
        for u, v in trace_edges:
            children[u].append(v)

        best_gain = 0.0
        best_candidate = None  # (u, d)
        best_new_edges = None  # set of edges to add (including subtrace)
        best_new_nodes = None

        # Consider every OR/g node currently in the trace
        OR_nodes = [v for v in trace_nodes if nodes[v].typ in ("d","g")]
        for d in OR_nodes:
            # All incoming edges from original graph
            for (u, w_uv, p_uv) in in_all.get(d, []):
                if (u, d) in trace_edges:
                    continue  # already included
                # Build tentative union: maybe need subtrace to u if u not in trace
                add_nodes = set()
                add_edges = set()
                if u not in trace_nodes:
                    # fetch or compute minimal subtrace to reach u
                    if u not in subtrace_cache:
                        ok, subT = sat_to_goal(nodes, out_edges, in_edges, u)
                        if not ok:
                            continue  # unreachable
                        subtrace_cache[u] = (set(subT["nodes"]), set(subT["edges"]))
                    su_nodes, su_edges = subtrace_cache[u]
                    add_nodes |= su_nodes
                    add_edges |= su_edges
                # Now consider adding (u->d)
                add_edges.add((u, d))

                # Cycle check: in the tentative graph, d must not reach u
                tmp_nodes = trace_nodes | add_nodes
                tmp_edges = trace_edges | add_edges
                tmp_children = defaultdict(list)
                for x, y in tmp_edges:
                    tmp_children[x].append(y)
                if has_path(tmp_children, d, u):
                    continue  # would create a cycle, skip

                # Evaluate gain: recompute P(g) for the tentative trace
                tentative_trace = {"nodes": sorted(tmp_nodes), "edges": sorted(tmp_edges)}
                Pg_new, _ = compute_probability_of_trace(tentative_trace, nodes, edge_prob)
                gain = Pg_new - Pg
                if gain > best_gain + 1e-18:
                    best_gain = gain
                    best_candidate = (u, d)
                    best_new_edges = add_edges
                    best_new_nodes = add_nodes

        if best_gain > eps and best_candidate is not None:
            # Apply the best augmentation
            trace_nodes |= best_new_nodes
            trace_edges |= best_new_edges
            current_trace = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
            Pg_after, _ = compute_probability_of_trace(current_trace, nodes, edge_prob)
            history.append({
                "iter": it,
                "added_edge": list(best_candidate),
                "delta_Pg": best_gain,
                "Pg_after": Pg_after
            })
        else:
            # No positive gain found
            break

    final_trace = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
    final_Pg, Pnode = compute_probability_of_trace(final_trace, nodes, edge_prob)
    return final_trace, final_Pg, Pnode, history

def pretty_print(base_trace, base_Pg, aug_trace, aug_Pg, history):
    print("=== Base (SAT) trace ===")
    print("Nodes:", ", ".join(base_trace["nodes"]))
    print("Edges:")
    for u, v in base_trace["edges"]:
        print(f"  {u} -> {v}")
    print(f"P(g) base = {base_Pg:.6f}\n")

    if history:
        print("=== Greedy augmentation steps ===")
        for h in history:
            u, d = h["added_edge"]
            print(f"iter {h['iter']}: add {u} -> {d}, ΔP(g) = {h['delta_Pg']:.6f}, P(g) -> {h['Pg_after']:.6f}")
        print()

    print("=== Final augmented trace ===")
    print("Nodes:", ", ".join(aug_trace["nodes"]))
    print("Edges:")
    for u, v in aug_trace["edges"]:
        print(f"  {u} -> {v}")
    print(f"P(g) final = {aug_Pg:.6f}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json", help="Path to probability-annotated graph JSON")
    ap.add_argument("--iters", type=int, default=100, help="Max greedy iterations")
    ap.add_argument("--eps", type=float, default=1e-12, help="Minimal positive gain to add an edge")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    nodes, out_edges, in_edges, edge_prob, goal = load_graph(args.graph_json)

    # 1) Base SAT trace to original goal
    ok, base_trace = sat_to_goal(nodes, out_edges, in_edges, goal)
    if not ok:
        print("NO ATTACK TRACE"); sys.exit(1)

    base_Pg, _ = compute_probability_of_trace(base_trace, nodes, edge_prob)

    # 2) Greedy augment
    aug_trace, aug_Pg, Pnode, history = greedy_augment(
        nodes, out_edges, in_edges, edge_prob, base_trace,
        max_iters=args.iters, eps=args.eps
    )

    if args.pretty:
        pretty_print(base_trace, base_Pg, aug_trace, aug_Pg, history)
    else:
        out = {
            "base": {"trace": base_trace, "P_g": base_Pg},
            "final": {"trace": aug_trace, "P_g": aug_Pg},
            "history": history
        }
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
