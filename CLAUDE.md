# agent-scaling

Multi-agent harness inspired by "Towards a Science of Scaling Agent Systems" (arXiv 2512.08296):
single, independent, centralized, decentralized and hybrid topologies, compared with the paper's
metrics (turns, tokens/overhead, message density, redundancy, error amplification).
Current stage: single-agent baseline (`agent.py`, tasks in `tasks.jsonl`, outputs in `runs/`).

## Cluster: Princeton CS Neuronic (Slurm)
- Nodes: 8x NVIDIA L40 (46 GB each, PCIe, no NVLink), 512 GB RAM, 3.5 TB local SSD at `/scratch`,
  10 Gbps Ethernet (no InfiniBand, so never shard a model across nodes).
- Home is only 16 GB: keep code there; put the venv, HF cache and model weights in `/scratch/$USER`
  (local to one node, may be wiped between jobs).
- Compute nodes have internet access. Interactive jobs (`salloc`) are capped at 2 hours.
- Never run GPU or heavy work on the login node (`neuronic`). Do installs and downloads inside a job.

## Model serving
- After `salloc`, run `source ~/agent-scaling/setup.sh [20b|120b]`: it sets env vars, installs uv/vLLM
  and downloads weights into `/scratch/$USER` if missing, and starts vLLM. Re-sourcing is safe.
- The cluster shell has `noclobber` set: use `>|` to overwrite files in shell redirects.
- vLLM on the compute node, OpenAI-compatible API at `localhost:8000`.
- gpt-oss-20b on 1 GPU for development; gpt-oss-120b with `--tensor-parallel-size 2` as the main model.
- Tool calling needs `--enable-auto-tool-choice --tool-call-parser openai` on `vllm serve`.
