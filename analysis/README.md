# Analysis scripts for the agent-driven communication study (30 Sep 2026)

Run from `~/agent-scaling` with the vLLM venv active (`driven_analysis.py` labels messages with the local model):

- `fixed_comm.py` — what the paper's fixed-round setups send (message counts, sizes, tables, uptake of IDs).
- `driven_analysis.py [run dirs...] [out.json]` — agent-driven runs: outcomes, channel use, timing, shared-state keys, message kinds.
- `hard.py` — summarizes the 3+-change runs and builds the web page from `agent-driven.html` + `swimlanes.json`.

The published page: https://claude.ai/artifact/6ckTqEhNESDYmG3B5oyzc7 (built from `agent-driven-final.html`).
