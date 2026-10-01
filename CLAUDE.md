# agent-scaling

Multi-agent harness inspired by "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296):
single, independent, centralized, decentralized and hybrid topologies, compared with the paper's
metrics (turns, tokens/overhead, message density, redundancy, error amplification).
Current stage: single agent (`agent.py`) plus the paper's four multi-agent topologies (`mas.py`,
`agent.py --topology independent|centralized|decentralized|hybrid`, WorkBench only), following the
authors' released code (github.com/ybkim95/agent-scaling). Our own extension, `agent_driven.py`
(`--topology agent-driven --comm all|text|a2a|shared|none [--coordinate]`), drops the fixed rounds:
peers share one workspace and choose whether and how to communicate. `compare.py` tabulates runs with the
paper's metrics; `show_trace.py` prints a task's trace as a timeline; `trace_viz.py` builds an animated
HTML replay of one task across runs. Toy tasks in `tasks.jsonl`, outputs in `runs/`.
Benchmark: WorkBench (`workbench.py`, `agent.py --benchmark workbench --limit N`), one of the
paper's six; its repo is cloned by `setup.sh` into `/scratch/$USER/WorkBench` and scored with its own
state-based evaluator (accuracy plus side-effect rate).
Second benchmark, in progress: BrowseComp-Plus (`browsecomp.py`, `agent.py --benchmark browsecomp`, with every
topology: `mas.py` BrowseCompEnv, and agent-driven teams that share one answer slot where the last submission counts): the paper's 100 questions over a fixed 100K-document collection, dense search with the paper's
Qwen3-Embedding-4B index, graded by the local model with the paper's grader prompt. `BROWSECOMP=1` before
`source setup.sh` (or `sbatch --export=ALL,BROWSECOMP=1 run.sbatch ...`) downloads it into `/scratch/$USER/browsecomp`
and serves the query embedder as a second vLLM on PORT+1; `browsecomp_check.py` checks search. The decrypted
questions must never leave `/scratch`: not in git, not on artifact pages.

## Cluster: Princeton CS Neuronic (Slurm)
- Nodes: 8x NVIDIA L40 (46 GB each, PCIe, no NVLink), 512 GB RAM, 3.5 TB local SSD at `/scratch`,
  10 Gbps Ethernet (no InfiniBand, so never shard a model across nodes).
- Home is only 16 GB: keep code there; put the venv, HF cache and model weights in `/scratch/$USER`
  (local to one node, may be wiped between jobs).
- Compute nodes have internet access. Interactive jobs (`salloc`) are capped at 2 hours.
- Never run GPU or heavy work on the login node (`neuronic`). Do installs and downloads inside a job.

## Workflow
- Code lives in the private repo github.com/mghilea/agent-scaling. It's edited both on the Mac and
  on the cluster (the cluster clone pushes with a read-write deploy key scoped to this repo).
  `git pull` before starting work, commit in small steps, and push when a change works, so the
  other machine doesn't diverge. `runs/` and `logs/` contents are not in git.
- Real runs go through Slurm: `sbatch run.sbatch [20b|120b] [agent.py args]`, submitted from
  `~/agent-scaling`. Interactive `salloc` is for setup and debugging.

## Model serving
- After `salloc`, run `source ~/agent-scaling/setup.sh [20b|120b]`: it sets env vars, installs uv/vLLM
  and downloads weights into `/scratch/$USER` if missing, and starts vLLM. Re-sourcing is safe.
- The cluster shell has `noclobber` set: use `>|` to overwrite files in shell redirects.
- vLLM on the compute node, OpenAI-compatible API at `localhost:8000`.
- gpt-oss-20b on 1 GPU for development; gpt-oss-120b with `--tensor-parallel-size 2` as the main model.
- Tool calling needs `--enable-auto-tool-choice --tool-call-parser openai` on `vllm serve`.
