#!/bin/bash
# Batch job for Claude models (Anthropic API): no GPU and no vLLM; sets up AppWorld and runs agent.py with its Python.
# Submit from ~/agent-scaling; the arguments go to agent.py (--max-dollars is required for Claude models):
#
#   sbatch claude.sbatch --benchmark appworld --dataset dev --model claude-opus-5-5 --max-dollars 25 \
#       --only "$(python analysis/aw_big_tasks.py)" --tag big01
#
# The API key comes from ~/.config/anthropic/api_key (chmod 600) unless ANTHROPIC_API_KEY is set. Never commit it.
# Watch it:   tail -f logs/claude-<jobid>.out      Cancel it:  scancel <jobid>
#SBATCH --job-name=claude
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), started $(date)"
export APPWORLD=1 APPWORLD_ONLY=1
source ./setup.sh || exit 1
"$APPWORLD_PYTHON" agent.py "$@"
echo "finished $(date)"
