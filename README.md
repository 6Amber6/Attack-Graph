# Attack Graph — SAT Baseline & Probability Extension

- **`sat.py`**: Baseline implementation of **Shortest Attack Trace (SAT)** on a weighted attack graph  
  - The original graph may contain cycles; the output attack trace is always acyclic (a DAG).
  - AND nodes take the **max**, OR/goal nodes take the **min**.
  - Supports node weights `w(v)` and edge weights `w(u,v)` (non-negative).
  - Reconstructs and outputs the shortest attack trace; can export DOT (trace subgraph only).

- **`sat_prob.py`**: **Probability extension (Phase 1)** that computes success probability on the SAT trace  
  - Reads graphs with success probabilities (nodes/edges have `p`).
  - Still uses SAT to select an acyclic trace with **single-parent OR**.
  - Computes **P(g)** (trace success probability) on that trace under the independence assumption.
  - Serves as a comparison baseline for later MVAT (where OR may take multiple parents).
 
- **`mvat_sum_budget_hard.py`**: Greedy MVAT (multi-parent OR) with SUM-cost and HARD budget only.
  - Builds a baseline SAT trace using weights w (AND=max, OR=min).
  - Computes its success probability P(g) under independence (node/edge probs p).
  - Greedily augments OR/goal nodes by adding extra parent edges to maximize ΔP(g),
    subject to:
    * Acyclicity (no cycles introduced).
    * HARD budget on SUM-cost: C_sum(T) <= B, where
        C_sum(T) = sum_{v in T} w(v) + sum_{(u,v) in T} w(u,v)
  - If a candidate parent u is not in the current trace, the algorithm attaches the
    minimal-cost subtrace to u (obtained by running SAT with u as the goal).

