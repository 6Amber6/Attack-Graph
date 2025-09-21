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
