"""
Greedy MVAT (multi-parent OR) with SUM-cost and HARD budget only.

- Builds a baseline SAT trace using weights w (AND=max, OR=min).
- Computes its success probability P(g) under independence (node/edge probs p).
- Greedily augments OR/goal nodes by adding extra parent edges to maximize ΔP(g),
  subject to:
    * Acyclicity (no cycles introduced).
    * HARD budget on SUM-cost: C_sum(T) <= B, where
        C_sum(T) = sum_{v in T} w(v) + sum_{(u,v) in T} w(u,v)
- If a candidate parent u is not in the current trace, the algorithm attaches the
  minimal-cost subtrace to u (obtained by running SAT with u as the goal).

USAGE
-----
python3 mvat_sum_budget_hard.py graph_sum_budget_demo.json --pretty --budget 2.0
python3 mvat_sum_budget_hard.py graph_sum_budget_demo.json --pretty --budget 10.5 --iters 50 --eps 1e-9

INPUT JSON
----------
{
  "nodes": [
    {"id":"p1","type":"p","w":0.0,"p":0.9},
    {"id":"r1","type":"r","w":0.2,"p":0.95},
    {"id":"d1","type":"d","w":0.1,"p":0.9},
    {"id":"g", "type":"g","w":0.0,"p":1.0}
  ],
  "edges": [
    ["p1","r1", 0.3, 0.9],   // edge format: [u, v, w(u,v), p(u,v)] ; p optional (defaults 1.0)
    ["r1","d1", 0.3, 0.8],
    ["d1","g",  0.2, 0.9]
  ],
  "goal": "g"
}
"""

import json, math, sys, heapq, copy
from collections import defaultdict, deque

WHITE, GRAY, BLACK = 0, 1, 2

# ---------------- Data structures ----------------

class Node:
    __slots__ = ("id","typ","w","p","in_deg","done","height","color","pred")
    def __init__(self, id, typ, w, p):
        self.id = id
        self.typ = typ  # 'p', 'r', 'd', 'g'
        self.w = float(w)
        self.p = float(p)  # local success prob
        self.in_deg = 0
        self.done = 0
        # SAT fields
        if typ in ("d","g"):
            self.height = math.inf
        elif typ == "r":
            self.height = -math.inf
        elif typ == "p":
            self.height = self.w
        else:
            raise ValueError(f"Unknown node type {typ}")
        self.color = WHITE
        self.pred = None  # best predecessor for OR/g (SAT backtrack)

# ---------------- Graph IO ----------------

def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = {n["id"]: Node(n["id"], n["type"], n.get("w", 0.0), n.get("p", 1.0)) for n in data["nodes"]}
    out_edges = defaultdict(list)  # u -> list of (v, w_uv, p_uv)
    in_edges = defaultdict(list)   # v -> list of (u, w_uv, p_uv)
    edge_prob = {}                 # (u,v) -> p(u,v)
    edge_w = {}                    # (u,v) -> w(u,v)
    for e in data["edges"]:
        if len(e) == 3:
            u, v, w = e; p_uv = 1.0
        else:
            u, v, w, p_uv = e
        w = float(w); p_uv = float(p_uv)
        out_edges[u].append((v, w, p_uv))
        in_edges[v].append((u, w, p_uv))
        edge_prob[(u, v)] = p_uv
        edge_w[(u, v)] = w
        nodes[v].in_deg += 1
    goal = data["goal"]
    if goal not in nodes:
        raise ValueError("Goal node not found in nodes")
    return nodes, out_edges, in_edges, edge_prob, edge_w, goal

# ---------------- SAT to arbitrary goal ----------------

def sat_to_goal(nodes_in, out_edges, in_edges, goal):
    """Run SAT using 'goal' as target; return minimal-cost (height-min) trace to 'goal'."""
    nodes = {k: copy.copy(v) for k, v in nodes_in.items()}
    pq = []; counter = 0
    for v in nodes.values():
        v.in_deg = len(in_edges.get(v.id, []))
        v.done = 0; v.color = WHITE; v.pred = None
        if v.typ == "p":
            v.height = v.w; v.color = GRAY
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
        for vid, w_uv, _p in out_edges.get(uid, ()):
            v = nodes[vid]
            temp = u.height + w_uv + v.w
            if v.typ in ("d","g"):  # OR -> min
                if temp < v.height:
                    v.height = temp; v.pred = uid
                if v.color == WHITE:
                    v.color = GRAY; heapq.heappush(pq, (v.height, counter, vid)); counter += 1
                elif v.color == GRAY:
                    heapq.heappush(pq, (v.height, counter, vid)); counter += 1
            elif v.typ == "r":     # AND -> max (wait all parents)
                if temp > v.height:
                    v.height = temp
                v.done += 1
                if v.done == v.in_deg and v.color == WHITE:
                    v.color = GRAY; heapq.heappush(pq, (v.height, counter, vid)); counter += 1
            else:
                raise ValueError(f"Unknown node type {v.typ}")
    return False, {"height": None, "nodes": [], "edges": []}

def reconstruct(nodes, in_edges, goal):
    """Backtrack to form a DAG trace: OR keeps only pred; AND keeps all preds."""
    T_nodes = set(); T_edges = []
    q = deque([goal])
    while q:
        v = q.popleft()
        if v in T_nodes:
            continue
        T_nodes.add(v)
        nv = nodes[v]
        if nv.typ in ("d","g"):  # OR
            if nv.pred is not None:
                T_edges.append((nv.pred, v)); q.append(nv.pred)
        elif nv.typ == "r":      # AND
            for pu, _, _ in in_edges[v]:
                T_edges.append((pu, v)); q.append(pu)
        elif nv.typ == "p":
            pass
    return {"height": nodes[goal].height, "nodes": sorted(T_nodes), "edges": T_edges}

# ---------------- Probability on a DAG trace ----------------

def compute_probability_of_trace(trace, nodes, edge_prob):
    """Topo-DP on trace using independence formulas; returns P(g), P dict."""
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
    g = next((v for v in trace["nodes"] if nodes[v].typ == "g"), None)
    return P.get(g, 0.0), P

# ---------------- SUM-cost ----------------

def cost_sum(trace, nodes, edge_w):
    """Sum of node weights + sum of edge weights over the trace (unique nodes/edges)."""
    c_nodes = sum(nodes[v].w for v in trace["nodes"])
    c_edges = sum(edge_w.get((u, v), 0.0) for (u, v) in trace["edges"])
    return c_nodes + c_edges

# ---------------- Helpers ----------------

def has_path(children, src, dst):
    """Check reachability (used for cycle check) in current tentative DAG."""
    if src == dst:
        return True
    seen = {src}; dq = deque([src])
    while dq:
        x = dq.popleft()
        for y in children.get(x, []):
            if y == dst:
                return True
            if y not in seen:
                seen.add(y); dq.append(y)
    return False

# ---------------- Greedy augmentation with HARD budget (only) ----------------

def greedy_augment_sum_hard(nodes, out_edges, in_edges, edge_prob, edge_w, base_trace,
                            budget, max_iters=100, eps=1e-9):
    """
    Only consider candidates whose tentative trace cost_sum <= budget.
    Among feasible candidates, pick the one with maximal ΔP(g) (tie-break by minimal ΔC).
    Stop when no feasible candidate yields ΔP(g) > eps.
    """
    if budget is None:
        raise ValueError("HARD budget is required. Please pass --budget B.")

    trace_nodes = set(base_trace["nodes"])
    trace_edges = set(base_trace["edges"])
    history = []
    subtrace_cache = {}

    for it in range(1, max_iters + 1):
        current = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
        Pg_before, _ = compute_probability_of_trace(current, nodes, edge_prob)
        C_before = cost_sum(current, nodes, edge_w)

        # adjacency for cycle checking
        children = defaultdict(list)
        for u, v in trace_edges:
            children[u].append(v)

        best = None  # (gain, -dC, (u,d), add_nodes, add_edges, Pg_new, C_new)
        OR_nodes = [v for v in trace_nodes if nodes[v].typ in ("d","g")]

        for d in OR_nodes:
            for (u, w_uv, p_uv) in in_edges.get(d, []):
                if (u, d) in trace_edges:
                    continue
                add_nodes, add_edges = set(), set()

                # ensure u is provable (attach minimal subtrace to u if needed)
                if u not in trace_nodes:
                    if u not in subtrace_cache:
                        ok, subT = sat_to_goal(nodes, out_edges, in_edges, u)
                        if not ok:
                            continue
                        subtrace_cache[u] = (set(subT["nodes"]), set(subT["edges"]))
                    su_nodes, su_edges = subtrace_cache[u]
                    add_nodes |= su_nodes
                    add_edges |= su_edges

                # add candidate edge
                add_edges.add((u, d))

                # cycle check: d must not reach u after adding
                tmp_nodes = trace_nodes | add_nodes
                tmp_edges = trace_edges | add_edges
                tmp_children = defaultdict(list)
                for x, y in tmp_edges:
                    tmp_children[x].append(y)
                if has_path(tmp_children, d, u):
                    continue

                tentative = {"nodes": sorted(tmp_nodes), "edges": sorted(tmp_edges)}
                Pg_new, _ = compute_probability_of_trace(tentative, nodes, edge_prob)
                gain = Pg_new - Pg_before
                if gain <= eps:
                    continue

                C_new = cost_sum(tentative, nodes, edge_w)
                if C_new > budget + 1e-12:
                    continue  # violates HARD budget

                dC = C_new - C_before
                rank = (gain, -dC)  # max gain, then prefer smaller cost increase
                if best is None or rank > best[0]:
                    best = (rank, gain, dC, (u, d), add_nodes, add_edges, Pg_new, C_new)

        if best is None:
            # no feasible positive-gain candidate
            break

        # apply the best
        _rank, gain, dC, (u, d), add_nodes, add_edges, Pg_after, C_after = best
        trace_nodes |= add_nodes
        trace_edges |= add_edges
        history.append({
            "iter": it,
            "added_edge": [u, d],
            "delta_Pg": gain,
            "delta_C": dC,
            "Pg_after": Pg_after,
            "cost_after": C_after
        })

    final_trace = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
    final_Pg, Pnode = compute_probability_of_trace(final_trace, nodes, edge_prob)
    final_C = cost_sum(final_trace, nodes, edge_w)
    return final_trace, final_Pg, final_C, Pnode, history

# ---------------- CLI ----------------

def pretty_print(base_trace, base_Pg, base_C, aug_trace, aug_Pg, aug_C, history, budget):
    print("=== Base (SAT) trace ===")
    print("Nodes:", ", ".join(base_trace["nodes"]))
    print("Edges:")
    for u, v in base_trace["edges"]:
        print(f"  {u} -> {v}")
    print(f"P(g) base = {base_Pg:.6f} ; Cost_sum base = {base_C:.6f}")
    print(f"HARD budget = {budget:.6f}\n")

    if history:
        print("=== Greedy augmentation steps ===")
        for h in history:
            u, d = h["added_edge"]
            print(f"iter {h['iter']}: add {u} -> {d}, ΔP = {h['delta_Pg']:.6g}, "
                  f"ΔC = {h['delta_C']:.6g}, P(g) -> {h['Pg_after']:.6g}, Cost -> {h['cost_after']:.6g}")
        print()
    else:
        print("No augmentation performed (no feasible candidate with ΔP > eps under the budget).\n")

    print("=== Final augmented trace ===")
    print("Nodes:", ", ".join(aug_trace["nodes"]))
    print("Edges:")
    for u, v in aug_trace["edges"]:
        print(f"  {u} -> {v}")
    print(f"P(g) final = {aug_Pg:.6f} ; Cost_sum final = {aug_C:.6f}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json", help="Path to probability-annotated graph JSON")
    ap.add_argument("--budget", type=float, required=True, help="HARD budget B on SUM-cost (node+edge weights)")
    ap.add_argument("--iters", type=int, default=100, help="Max greedy iterations")
    ap.add_argument("--eps", type=float, default=1e-9, help="Minimal positive ΔP to accept a candidate")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()

    nodes, out_edges, in_edges, edge_prob, edge_w, goal = load_graph(args.graph_json)

    # 1) Baseline SAT trace to the goal
    ok, base_trace = sat_to_goal(nodes, out_edges, in_edges, goal)
    if not ok:
        print("NO ATTACK TRACE"); sys.exit(1)

    # 2) Evaluate baseline Pg and cost
    base_Pg, _ = compute_probability_of_trace(base_trace, nodes, edge_prob)
    base_C = cost_sum(base_trace, nodes, edge_w)

    # 3) Greedy augmentation with HARD budget (sum-cost)
    aug_trace, aug_Pg, aug_C, Pnode, history = greedy_augment_sum_hard(
        nodes, out_edges, in_edges, edge_prob, edge_w, base_trace,
        budget=args.budget, max_iters=args.iters, eps=args.eps
    )

    if args.pretty:
        pretty_print(base_trace, base_Pg, base_C, aug_trace, aug_Pg, aug_C, history, args.budget)
    else:
        out = {
            "base": {"trace": base_trace, "P_g": base_Pg, "cost_sum": base_C},
            "final": {"trace": aug_trace, "P_g": aug_Pg, "cost_sum": aug_C},
            "history": history,
            "budget": args.budget
        }
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
