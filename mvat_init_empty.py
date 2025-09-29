#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Complete Greedy MVAT that considers all possible pathconsidering all possible path combinations and selecting the one with maximum probability gain.
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
        if v in nodes:
            nodes[v].in_deg += 1
    goal = data["goal"]
    if goal not in nodes:
        raise ValueError("Goal node not found in nodes")
    return nodes, out_edges, in_edges, edge_prob, edge_w, goal

# ---------------- SAT to arbitrary goal ----------------

def sat_to_goal(nodes_in, out_edges, in_edges, goal):
    """Run SAT using 'goal' as the target; return minimal-cost (height-min) trace to 'goal'."""
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
            return True, reconstruct_sat(nodes, in_edges, goal)
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

def reconstruct_sat(nodes, in_edges, goal):
    """Backtrack to form a DAG trace for SAT: OR keeps only pred; AND keeps all preds."""
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
    return {"height": nodes[goal].height, "nodes": sorted(T_nodes), "edges": sorted(set(T_edges))}

# ---------------- Probability on a DAG trace ----------------

def compute_probability_of_trace(trace, nodes, edge_prob):
    """Topo-DP on the given DAG trace; returns P(g), P dict."""
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

# ---------------- Cost (SUM) ----------------

def cost_sum(trace, nodes, edge_w):
    """Sum of node weights + sum of edge weights over the trace (unique nodes/edges)."""
    c_nodes = sum(nodes[v].w for v in trace["nodes"])
    c_edges = sum(edge_w.get((u, v), 0.0) for (u, v) in trace["edges"])
    return c_nodes + c_edges

# ---------------- Helpers ----------------

def has_path(children, src, dst):
    """Check reachability (cycle check) in a DAG: is dst reachable from src?"""
    if src == dst:  # trivial
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

def to_dot_full_graph(nodes, out_edges, edge_w, edge_prob, path):
    """Export the full original graph."""
    lines = ["digraph G {", '  rankdir=TB;']
    for v in nodes.values():
        shape = "box" if v.typ=="p" else ("diamond" if v.typ=="g" else ("ellipse" if v.typ=="d" else "hexagon"))
        lines.append(f'  "{v.id}" [shape={shape}, label="{v.id}\\n(type={v.typ}, w={v.w}, p={v.p})"];')
    for u, lst in out_edges.items():
        for v, w, p in lst:
            lines.append(f'  "{u}" -> "{v}" [label="w={w}, p={p}"];')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def to_dot_trace(trace, nodes, edge_w, edge_prob, Pnode, path):
    """Export a TRACE as DOT with labels."""
    lines = ["digraph Trace {", '  rankdir=TB;']
    for v in trace["nodes"]:
        nv = nodes[v]
        shape = "box" if nv.typ=="p" else ("diamond" if nv.typ=="g" else ("ellipse" if nv.typ=="d" else "hexagon"))
        pv = Pnode.get(v, 0.0)
        lines.append(f'  "{v}" [shape={shape}, label="{v}\\n(type={nv.typ}, w={nv.w}, p={nv.p}, P={pv:.3f})"];')
    for u, v in trace["edges"]:
        w = edge_w.get((u,v), 0.0); p = edge_prob.get((u,v), 1.0)
        lines.append(f'  "{u}" -> "{v}" [label="w={w}, p={p}"];')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

# ---------------- Complete Greedy augmentation ----------------

def find_all_paths_to_goal(nodes, out_edges, in_edges, goal):
    """Find all possible paths to the goal node."""
    paths = []
    
    def dfs(current, path, visited):
        if current == goal:
            paths.append(path[:])
            return
        
        if current in visited:
            return
        
        visited.add(current)
        for next_node, _, _ in out_edges.get(current, []):
            path.append(next_node)
            dfs(next_node, path, visited)
            path.pop()
        visited.remove(current)
    
    # Start from all leaf nodes (nodes with no incoming edges)
    leaf_nodes = [node_id for node_id in nodes.keys() if len(in_edges.get(node_id, [])) == 0]
    
    for leaf in leaf_nodes:
        dfs(leaf, [leaf], set())
    
    return paths

def greedy_augment_empty(nodes, out_edges, in_edges, edge_prob, edge_w, goal,
                         budget, max_iters=100, eps=1e-9):
    """
    Start from T={g}. Consider all possible path combinations and select the one with maximum probability gain.
    
    The key insight is that we need to consider all possible paths to the goal, not just single edge additions.
    """
    # Find all possible paths to the goal
    all_paths = find_all_paths_to_goal(nodes, out_edges, in_edges, goal)
    
    print(f"Found {len(all_paths)} possible paths to goal:")
    for i, path in enumerate(all_paths):
        print(f"  Path {i+1}: {' -> '.join(path)}")
    
    # Evaluate each path
    best_path = None
    best_prob = 0.0
    best_cost = float('inf')
    
    for path in all_paths:
        # Convert path to trace
        trace_nodes = set(path)
        trace_edges = []
        for i in range(len(path) - 1):
            trace_edges.append((path[i], path[i+1]))
        
        trace = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
        
        # Check if path is valid (no cycles, within budget)
        cost = cost_sum(trace, nodes, edge_w)
        if cost > budget + 1e-12:
            print(f"Path {' -> '.join(path)}: cost {cost:.6f} exceeds budget {budget:.6f}")
            continue
        
        # Check for cycles - a path is valid if it's a DAG
        # We don't need to check for cycles in a simple path
        has_cycle = False
        
        if has_cycle:
            print(f"Path {' -> '.join(path)}: contains cycle")
            continue
        
        # Compute probability
        prob, _ = compute_probability_of_trace(trace, nodes, edge_prob)
        
        print(f"Path {' -> '.join(path)}: P(g)={prob:.6f}, cost={cost:.6f}")
        
        # Select path with maximum probability
        if prob > best_prob or (prob == best_prob and cost < best_cost):
            best_path = trace
            best_prob = prob
            best_cost = cost
    
    if best_path is None:
        # Fallback to original algorithm
        trace_nodes = set([goal])
        trace_edges = set()
        history = []
        subtrace_cache = {}

        for it in range(1, max_iters + 1):
            current = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
            Pg_before, _ = compute_probability_of_trace(current, nodes, edge_prob)
            C_before = cost_sum(current, nodes, edge_w)

            # Build children map for cycle checks
            children = defaultdict(list)
            for u, v in trace_edges:
                children[u].append(v)

            best = None  # (rank, gain, dC, (u,d), add_nodes, add_edges, Pg_new, C_new)

            # Consider all OR/g nodes currently in the trace
            OR_nodes = [v for v in trace_nodes if nodes[v].typ in ("d","g")]
            for d in OR_nodes:
                for (u, w_uv, p_uv) in in_edges.get(d, []):
                    if (u, d) in trace_edges:
                        continue
                    add_nodes, add_edges = set(), set()

                    # If u not yet in trace, attach minimal SAT subtrace to reach u
                    if u not in trace_nodes:
                        if u not in subtrace_cache:
                            ok, subT = sat_to_goal(nodes, out_edges, in_edges, u)
                            if not ok:
                                continue  # unreachable
                            subtrace_cache[u] = (set(subT["nodes"]), set(subT["edges"]))
                        su_nodes, su_edges = subtrace_cache[u]
                        add_nodes |= su_nodes
                        add_edges |= su_edges

                    # Add candidate edge u->d
                    add_edges.add((u, d))

                    # Cycle check on the tentative graph
                    tmp_nodes = trace_nodes | add_nodes
                    tmp_edges = trace_edges | add_edges
                    tmp_children = defaultdict(list)
                    for x, y in tmp_edges:
                        tmp_children[x].append(y)
                    if has_path(tmp_children, d, u):  # would introduce a cycle
                        continue

                    tentative = {"nodes": sorted(tmp_nodes), "edges": sorted(tmp_edges)}
                    C_new = cost_sum(tentative, nodes, edge_w)
                    if C_new > budget + 1e-12:
                        continue  # violates HARD budget

                    Pg_new, _ = compute_probability_of_trace(tentative, nodes, edge_prob)
                    gain = Pg_new - Pg_before
                    if gain <= eps:
                        continue

                    dC = C_new - C_before
                    # Use probability gain as the primary sorting criterion
                    rank = (gain, -dC)  # prefer larger ΔP, then smaller ΔC
                    if best is None or rank > best[0]:
                        best = (rank, gain, dC, (u, d), add_nodes, add_edges, Pg_new, C_new)

            if best is None:
                break  # no feasible positive-gain candidate

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
    else:
        print(f"Selected best path: P(g)={best_prob:.6f}, cost={best_cost:.6f}")
        final_Pg, Pnode = compute_probability_of_trace(best_path, nodes, edge_prob)
        final_C = cost_sum(best_path, nodes, edge_w)
        return best_path, final_Pg, final_C, Pnode, []

def greedy_augment_from_sat(nodes, out_edges, in_edges, edge_prob, edge_w, base_trace,
                            budget, max_iters=100, eps=1e-9):
    """Optional: start from SAT trace, then augment (for comparison)."""
    trace_nodes = set(base_trace["nodes"])
    trace_edges = set(base_trace["edges"])
    history = []
    subtrace_cache = {}

    for it in range(1, max_iters + 1):
        current = {"nodes": sorted(trace_nodes), "edges": sorted(trace_edges)}
        Pg_before, _ = compute_probability_of_trace(current, nodes, edge_prob)
        C_before = cost_sum(current, nodes, edge_w)

        children = defaultdict(list)
        for u, v in trace_edges:
            children[u].append(v)

        best = None
        OR_nodes = [v for v in trace_nodes if nodes[v].typ in ("d","g")]

        for d in OR_nodes:
            for (u, w_uv, p_uv) in in_edges.get(d, []):
                if (u, d) in trace_edges:
                    continue
                add_nodes, add_edges = set(), set()

                if u not in trace_nodes:
                    if u not in subtrace_cache:
                        ok, subT = sat_to_goal(nodes, out_edges, in_edges, u)
                        if not ok:
                            continue
                        subtrace_cache[u] = (set(subT["nodes"]), set(subT["edges"]))
                    su_nodes, su_edges = subtrace_cache[u]
                    add_nodes |= su_nodes
                    add_edges |= su_edges

                add_edges.add((u, d))

                tmp_nodes = trace_nodes | add_nodes
                tmp_edges = trace_edges | add_edges
                tmp_children = defaultdict(list)
                for x, y in tmp_edges:
                    tmp_children[x].append(y)
                if has_path(tmp_children, d, u):
                    continue

                tentative = {"nodes": sorted(tmp_nodes), "edges": sorted(tmp_edges)}
                C_new = cost_sum(tentative, nodes, edge_w)
                if C_new > budget + 1e-12:
                    continue

                Pg_new, _ = compute_probability_of_trace(tentative, nodes, edge_prob)
                gain = Pg_new - Pg_before
                if gain <= eps:
                    continue

                dC = C_new - C_before
                rank = (gain, -dC)
                if best is None or rank > best[0]:
                    best = (rank, gain, dC, (u, d), add_nodes, add_edges, Pg_new, C_new)

        if best is None:
            break

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

def pretty_print(tag, trace, Pg, C, history, budget):
    print(f"=== {tag} trace ===")
    print("Nodes:", ", ".join(trace["nodes"]))
    print("Edges:")
    for u, v in trace["edges"]:
        print(f"  {u} -> {v}")
    print(f"P(g) = {Pg:.6f} ; Cost_sum = {C:.6f} ; HARD budget = {budget:.6f}\n")

    if history:
        print("=== Greedy augmentation steps ===")
        for h in history:
            u, d = h["added_edge"]
            print(f"iter {h['iter']}: add {u} -> {d}, ΔP = {h['delta_Pg']:.6g}, "
                  f"ΔC = {h['delta_C']:.6g}, P(g) -> {h['Pg_after']:.6g}, Cost -> {h['cost_after']:.6g}")
        print()

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json", help="Path to probability-annotated graph JSON")
    ap.add_argument("--budget", type=float, required=True, help="HARD budget B on SUM-cost (node+edge weights)")
    ap.add_argument("--init", choices=["empty","sat"], default="empty", help="Initialization mode")
    ap.add_argument("--iters", type=int, default=100, help="Max greedy iterations")
    ap.add_argument("--eps", type=float, default=1e-9, help="Minimal positive ΔP to accept a candidate")
    ap.add_argument("--pretty", action="store_true")
    ap.add_argument("--dot-base", metavar="OUT_DOT", help="Export DOT of base trace")
    ap.add_argument("--dot-final", metavar="OUT_DOT", help="Export DOT of final trace")
    ap.add_argument("--dot-full", metavar="OUT_DOT", help="Export DOT of the full original graph (optional)")
    args = ap.parse_args()

    nodes, out_edges, in_edges, edge_prob, edge_w, goal = load_graph(args.graph_json)

    if args.dot_full:
        to_dot_full_graph(nodes, out_edges, edge_w, edge_prob, args.dot_full)

    if args.init == "sat":
        ok, base_trace = sat_to_goal(nodes, out_edges, in_edges, goal)
        if not ok:
            print("NO ATTACK TRACE to goal via SAT"); sys.exit(1)
    else:
        base_trace = {"nodes": sorted({goal}), "edges": []}

    base_Pg, base_Pnode = compute_probability_of_trace(base_trace, nodes, edge_prob)
    base_C = cost_sum(base_trace, nodes, edge_w)

    if args.dot_base:
        to_dot_trace(base_trace, nodes, edge_w, edge_prob, base_Pnode, args.dot_base)

    if args.init == "sat":
        final_trace, final_Pg, final_C, Pnode, history = greedy_augment_from_sat(
            nodes, out_edges, in_edges, edge_prob, edge_w, base_trace,
            budget=args.budget, max_iters=args.iters, eps=args.eps
        )
        tag = "Final (from SAT)"
    else:
        final_trace, final_Pg, final_C, Pnode, history = greedy_augment_empty(
            nodes, out_edges, in_edges, edge_prob, edge_w, goal,
            budget=args.budget, max_iters=args.iters, eps=args.eps
        )
        tag = "Final (from EMPTY)"

    if args.dot_final:
        to_dot_trace(final_trace, nodes, edge_w, edge_prob, Pnode, args.dot_final)

    if args.pretty:
        pretty_print("Base", base_trace, base_Pg, base_C, [], args.budget)
        pretty_print(tag, final_trace, final_Pg, final_C, history, args.budget)
    else:
        out = {
            "init": args.init,
            "budget": args.budget,
            "base": {"trace": base_trace, "P_g": base_Pg, "cost_sum": base_C},
            "final": {"trace": final_trace, "P_g": final_Pg, "cost_sum": final_C},
            "history": history
        }
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
