
#!/usr/bin/env python3
"""
SAT (Shortest Attack Trace) baseline implementation.

- Node types:
  p: primitive fact (source)
  r: rule (AND)
  d: derived fact (OR)
  g: goal (treated like OR)

- Weights:
  Each node v has weight w(v) >= 0
  Each edge (u,v) has weight w(u,v) >= 0

- Height semantics:
  OR (d/g): height(v) = min over predecessors u of [height(u) + w(u,v) + w(v)]
  AND (r):  height(v) = max over predecessors u of [height(u) + w(u,v) + w(v)]
  p: height(p) = w(p)

- The algorithm works on graphs that may have cycles. The produced attack *trace*
  is always a DAG (acyclic), built by backtracking from the goal:
    - For OR nodes: keep exactly ONE selected predecessor edge (the best one)
    - For AND nodes: keep ALL incoming edges (all prerequisites)

Input format (JSON):
{
  "nodes": [
    {"id": "p1", "type": "p", "w": 0.0},
    {"id": "r1", "type": "r", "w": 0.0},
    {"id": "d1", "type": "d", "w": 0.0},
    {"id": "g",  "type": "g", "w": 0.0}
  ],
  "edges": [
    ["p1", "r1", 1.0],
    ["r1", "d1", 1.0],
    ["d1", "g",  1.0]
  ],
  "goal": "g"
}

Usage:
  python3 sat.py graph_a.json
  python3 sat.py graph_b_no_trace.json
  python3 sat.py graph_a.json --pretty    # pretty print the trace 
"""

import json, math, sys, heapq
from collections import defaultdict, deque

WHITE, GRAY, BLACK = 0, 1, 2

class Node:
    __slots__ = ("id","typ","w","in_deg","done","height","color","pred")
    def __init__(self, id, typ, w):
        self.id = id
        self.typ = typ  # 'p', 'r', 'd', 'g'
        self.w = float(w)
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
    nodes = {n["id"]: Node(n["id"], n["type"], n.get("w", 0.0)) for n in data["nodes"]}
    out_edges = defaultdict(list)  # u -> list of (v, w_uv)
    in_edges = defaultdict(list)   # v -> list of (u, w_uv)
    for u, v, w in data["edges"]:
        out_edges[u].append((v, float(w)))
        in_edges[v].append((u, float(w)))
        nodes[v].in_deg += 1
    goal = data["goal"]
    if goal not in nodes:
        raise ValueError("Goal node not found in nodes")
    return nodes, out_edges, in_edges, goal

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
        # If this popped height is worse than the current known height (for OR),
        # OR better than current height (for AND) - we still accept, because:
        # - OR only decreases; once popped, it is the best-known
        # - AND only increases; we only push AND when done == in_deg (fully settled)
        u.color = BLACK

        if uid == goal:
            return True, reconstruct(nodes, out_edges, in_edges, goal)

        for vid, w_uv in out_edges.get(uid, ()):
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
                    # push again (lazy update); old entry will be skipped when popped
                    heapq.heappush(pq, (v.height, counter, vid))
                    counter += 1
                else:
                    # v already BLACK with a better or equal height; ignore
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

def reconstruct(nodes, out_edges, in_edges, goal):
    """Backtrack a valid shortest attack trace from goal.
       - For OR/g nodes: keep exactly the chosen predecessor edge (pred)
       - For AND nodes: keep ALL incoming edges
       The result is acyclic by construction.
    """
    T_nodes = set()
    T_edges = []  # list of (u, v, w_uv)

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
                # find the edge weight pu -> v
                for uu, w_uv in in_edges[v]:
                    if uu == pu:
                        T_edges.append((pu, v, w_uv))
                        break
                q.append(pu)

        elif nv.typ == "r":  # AND: include all incoming prerequisites
            for pu, w_uv in in_edges[v]:
                T_edges.append((pu, v, w_uv))
                q.append(pu)

        elif nv.typ == "p":
            pass  # source, stop

    # sort nodes for stable output
    return {"height": nodes[goal].height, "nodes": sorted(T_nodes), "edges": T_edges}

def pretty_print(trace):
    print(f"SAT height: {trace['height']}")
    print("Nodes in trace:", ", ".join(trace["nodes"]))
    print("Edges in trace:")
    for u, v, w in trace["edges"]:
        print(f"  {u} -> {v} (w={w})")

def to_dot(trace, path="trace.dot"):
    # Generate a Graphviz DOT for the ATTACK TRACE (not the whole graph)
    lines = ["digraph Trace {"]
    lines.append('  rankdir=LR;')
    for u, v, w in trace["edges"]:
        lines.append(f'  "{u}" -> "{v}" [label="{w}"];')
    lines.append("}")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("graph_json", help="Path to graph JSON")
    ap.add_argument("--pretty", action="store_true", help="Pretty print the trace")
    ap.add_argument("--dot", metavar="OUT_DOT", help="Export the attack trace as Graphviz DOT")
    args = ap.parse_args()

    nodes, out_edges, in_edges, goal = load_graph(args.graph_json)
    ok, trace = sat(nodes, out_edges, in_edges, goal)
    if not ok:
        print("NO ATTACK TRACE")
        sys.exit(1)

    if args.pretty:
        pretty_print(trace)
    else:
        # default: concise machine-friendly JSON
        print(json.dumps(trace, indent=2))

    if args.dot:
        outp = to_dot(trace, args.dot)
        print(f"Wrote DOT to: {outp}")

if __name__ == "__main__":
    main()
