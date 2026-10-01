#!/bin/bash
# Batch job: start vLLM once, then run several agent.py configurations against it at the same time.
# Submit from ~/agent-scaling, with a suite file holding one agent.py argument list per line:
#
#   sbatch suite.sbatch 20b suites/hard-a.txt
#
# Each line gets its own log, logs/suite-<jobid>-<n>.log; this job's own log is logs/suite-<jobid>.out.
#SBATCH --job-name=suite
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), started $(date)"
size="${1:-20b}"
suite="$2"
[[ -f "$suite" ]] || { echo "no suite file: $suite" >&2; exit 1; }

export VLLM_LOG="$SLURM_SUBMIT_DIR/logs/vllm-$SLURM_JOB_ID.log"
source ./setup.sh "$size" || exit 1

n=0
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    n=$((n + 1))
    echo "[$n] python agent.py $line"
    # shellcheck disable=SC2086  # each line is an argument list
    python agent.py $line --quiet >| "logs/suite-$SLURM_JOB_ID-$n.log" 2>&1 &
    sleep 5
done < "$suite"
wait
echo "finished $(date)"
