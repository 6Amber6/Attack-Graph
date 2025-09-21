
#!/usr/bin/env python3
"""
SAT with probability evaluation (phase 1).

What this does:
- Uses the same SAT algorithm (AND=max, OR=min) to build a shortest attack trace (DAG).
- Then, under the independence assumption and *given that trace* (OR keeps one parent),
  it computes the success probability P(g) using the standard formulas:
    p-node:  P(p) = p(p)
    AND (r): P(r) = p(r) * product_{u in pred_T(r)} [ P(u) * p(u,r) ]
    OR (d/g, single-parent in trace): P(d) = p(d) * (1 - product_{u in pred_T(d)} [1 - P(u) * p(u,d)] )
  Since the SAT trace has at most one parent for OR, this reduces to
    P(d) = p(d) * P(u) * p(u,d)  (if exactly one parent).

Note:
- This script does NOT yet "add multiple parents" into OR nodes.
  So it evaluates the probability of the SAT-style trace.
  Next step (phase 2) would be a greedy MVAT heuristic that adds extra parents.

Usage:
  python3 sat_prob.py graph_prob_a.json
  python3 sat_prob.py graph_prob_a.json --pretty     # pretty print the trace 
"""

import json, math, sys, heapq
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
        # OR/g starts at +inf; AND starts at -inf; p will be set to w
        if typ in ("d", "g"):
            self.height = math.inf
        elif typ == "r":
            self.height = -math.inf
        elif typ == "p":
            self.height = self.w
        else:
            raise ValueError(f"Unknown node type {typ}")
        self.color = WHITE
        self.pred = None  # best predecessor id for OR/g

def load_graph(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # node probability default = 1.0 if missing
    nodes = {n["id"]: Node(n["id"], n["type"], n.get("w", 0.0), n.get("p", 1.0)) for n in data["nodes"]}
    out_edges = defaultdict(list)  # u -> list of (v, w_uv, p_uv)
    in_edges = defaultdict(list)   # v -> list of (u, w_uv, p_uv)
    edge_prob = {}                 # (u,v) -> p(u,v)
    for e in data["edges"]:
        # support both [u,v,w] and [u,v,w,p]
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

def sat(nodes, out_edges, in_edges, goal):
    # priority queue of (height, counter, node_id); counter breaks ties & avoids compare errors
    pq = []
    counter = 0

    # initialize: push all primitives; others remain WHITE for now
    for v in nodes.values():
        if v.typ == "p":
            v.color = GRAY
            heapq.heappush(pq, (v.height, counter, v.id))
            counter += 1

    while pq:
        h, _, uid = heapq.heappop(pq)
        u = nodes[uid]
        if u.color == BLACK:
            continue  # lazy deletion
        u.color = BLACK

        if uid == goal:
            return True, reconstruct(nodes, in_edges, goal)

        for vid, w_uv, p_uv in out_edges.get(uid, ()):
            v = nodes[vid]
            # temp cost via u -> v
            temp = u.height + w_uv + v.w

            if v.typ in ("d", "g"):  # OR: take minimum; can be pushed as soon as first candidate arrives
                if temp < v.height:
                    v.height = temp
                    v.pred = uid  # remember best predecessor
                if v.color == WHITE:
                    v.color = GRAY
                    heapq.heappush(pq, (v.height, counter, vid))
                    counter += 1
                elif v.color == GRAY:
                    heapq.heappush(pq, (v.height, counter, vid))
                    counter += 1
                else:
                    pass

            elif v.typ == "r":  # AND: take maximum; only push when all predecessors processed
                if temp > v.height:
                    v.height = temp
                v.done += 1
                if v.done == v.in_deg and v.color == WHITE:
                    v.color = GRAY
                    heapq.heappush(pq, (v.height, counter, vid))
                    counter += 1
            else:
                raise ValueError(f"Unexpected node type for {vid}: {v.typ}")

    # if we exit loop without returning, no attack trace
    return False, {"height": None, "nodes": [], "edges": []}

def reconstruct(nodes, in_edges, goal):
    """Backtrack a valid shortest attack trace from goal.
       - For OR/g nodes: keep exactly the chosen predecessor edge (pred)
       - For AND nodes: keep ALL incoming edges
       The result is acyclic by construction.
    """
    T_nodes = set()
    T_edges = []  # list of (u, v)

    q = deque([goal])
    while q:
        v = q.popleft()
        if v in T_nodes:
            continue
        T_nodes.add(v)
        nv = nodes[v]

        if nv.typ in ("d", "g"):  # OR
            pu = nv.pred
            if pu is not None:
                T_edges.append((pu, v))
                q.append(pu)

        elif nv.typ == "r":  # AND: include all incoming prerequisites
            for pu, _, _ in in_edges[v]:
                T_edges.append((pu, v))
                q.append(pu)

        elif nv.typ == "p":
            pass  # source, stop

    return {"height": nodes[goal].height, "nodes": sorted(T_nodes), "edges": T_edges}

def compute_probability_of_trace(trace, nodes, edge_prob, in_edges):
    """Given a *trace* (DAG) and node/edge probabilities, compute P(g).
       We do a topo-like DP over the trace edges.
    """
    # Build incoming list restricted to trace
    preds = defaultdict(list)  # v -> list of predecessors in trace
    for u, v in trace["edges"]:
        preds[v].append(u)

    # Build children within trace
    children = defaultdict(list)
    for u, v in trace["edges"]:
        children[u].append(v)

    # compute indegree within trace
    indeg = {v: len(preds[v]) for v in trace["nodes"]}
    for v in trace["nodes"]:
        indeg.setdefault(v, 0)

    # Kahn topo + DP
    from collections import deque
    q = deque([v for v in trace["nodes"] if indeg[v] == 0])
    P = {}

    while q:
        v = q.popleft()
        nv = nodes[v]
        if nv.typ == "p":
            P[v] = nv.p
        elif nv.typ == "r":  # AND
            if len(preds[v]) == 0:
                P[v] = 0.0
            else:
                prod = 1.0
                for u in preds[v]:
                    prod *= P.get(u, 0.0) * edge_prob.get((u, v), 1.0)
                P[v] = nv.p * prod
        elif nv.typ in ("d", "g"):  # OR
            if len(preds[v]) == 0:
                P[v] = 0.0
            else:
                fail_prod = 1.0
                for u in preds[v]:
                    fail_prod *= (1.0 - P.get(u, 0.0) * edge_prob.get((u, v), 1.0))
                P[v] = nv.p * (1.0 - fail_prod)
        else:
            P[v] = 0.0

        for w in children[v]:
            indeg[w] -= 1
            if indeg[w] == 0:
                q.append(w)

    goal = None
    for v in trace["nodes"]:
        if nodes[v].typ == "g":
            goal = v
            break
    if goal is None:
        return 0.0, P
    return P.get(goal, 0.0), P

def pretty_print(trace, prob_g, Pnode):
    print(f"SAT height: {trace['height']}")
    print(f"Trace success probability P(g): {prob_g:.6f}")
    print("Nodes in trace:", ", ".join(trace["nodes"]))
    print("Edges in trace:")
    for u, v in trace["edges"]:
        print(f"  {u} -> {v}")
    print("Node probabilities on trace:")
    for v in trace["nodes"]:
        print(f"  P({v}) = {Pnode.get(v, 0.0):.6f}")

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json", help="Path to probability-annotated graph JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty print the trace and probabilities")
    args = ap.parse_args()

    nodes, out_edges, in_edges, edge_prob, goal = load_graph(args.graph_json)
    ok, trace = sat(nodes, out_edges, in_edges, goal)
    if not ok:
        print("NO ATTACK TRACE")
        sys.exit(1)

    prob_g, Pnode = compute_probability_of_trace(trace, nodes, edge_prob, in_edges)
    if args.pretty:
        pretty_print(trace, prob_g, Pnode)
    else:
        out = {"height": trace["height"], "P_g": prob_g, "nodes": trace["nodes"], "edges": trace["edges"]}
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
