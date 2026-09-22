#!/bin/bash
# Batch job: set up the node, start vLLM, run the agent on the task set, exit.
# Submit from ~/agent-scaling (the log path below is relative to where you submit):
#
#   sbatch run.sbatch                                  # gpt-oss-20b, all tasks
#   sbatch run.sbatch 20b --reasoning-effort low       # extra args go to agent.py
#   sbatch --gres=gpu:2 -c 16 --mem=128G run.sbatch 120b
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
source ./setup.sh "$size" || exit 1

python agent.py "$@"
echo "finished $(date)"
