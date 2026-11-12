#!/bin/bash
# Batch job: set up the node, start vLLM, run the agent on the task set, exit.
# Submit from ~/agent-scaling (the log path below is relative to where you submit):
#
#   sbatch run.sbatch                                  # gpt-oss-20b, all tasks
#   sbatch run.sbatch 20b --reasoning-effort low       # extra args go to agent.py
#   sbatch --gres=gpu:2 -c 16 --mem=128G run.sbatch 120b
#   sbatch --export=ALL,BROWSECOMP=1 run.sbatch 20b --benchmark browsecomp   # also sets up BrowseComp-Plus
#   sbatch run.sbatch 20b --benchmark appworld --dataset dev                 # sets up AppWorld itself
#
# Watch it:   tail -f logs/agent-<jobid>.out    (vLLM's own log: logs/vllm-<jobid>.log)
# Cancel it:  scancel <jobid>
#SBATCH --job-name=agent
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), started $(date)"

size="${1:-20b}"
[[ $# -gt 0 ]] && shift

export VLLM_LOG="$SLURM_SUBMIT_DIR/logs/vllm-$SLURM_JOB_ID.log"
export PORT=$((20000 + SLURM_JOB_ID % 10000 * 2))  # a port of its own (even, so PORT+1 is free for the BrowseComp embedder)
# AppWorld runs in its own venv (its dependencies clash with vLLM's), which setup.sh builds with APPWORLD=1.
py=python
if [[ " $* " == *" --benchmark appworld "* ]]; then export APPWORLD=1; fi
source ./setup.sh "$size" || exit 1
if [[ " $* " == *" --benchmark appworld "* ]]; then py="$APPWORLD_PYTHON"; fi

"$py" agent.py "$@" --base-url "http://localhost:$PORT/v1"
echo "finished $(date)"
