#!/bin/bash
# Batch job: start vLLM once, then run several agent.py configurations against it at the same time.
# Submit from ~/agent-scaling, with a suite file holding one agent.py argument list per line:
#
#   sbatch suite.sbatch 20b suites/hard-a.txt
#   sbatch suite.sbatch 20b suites/hard-a.txt --tag rep03   # extra args go to every line
#
# Each job serves vLLM on its own port, so several suite jobs can share a node.
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
extra=("${@:3}")
export PORT=$((20000 + SLURM_JOB_ID % 10000 * 2))  # even, so PORT+1 (BrowseComp's embedder) is never another job's PORT

export VLLM_LOG="$SLURM_SUBMIT_DIR/logs/vllm-$SLURM_JOB_ID.log"
source ./setup.sh "$size" || exit 1

n=0
pids=()
while IFS= read -r line; do
    [[ -z "$line" || "$line" == \#* ]] && continue
    n=$((n + 1))
    echo "[$n] python agent.py $line ${extra[*]}"
    # shellcheck disable=SC2086  # each line is an argument list
    python agent.py $line "${extra[@]}" --base-url "http://localhost:$PORT/v1" --quiet \
        >| "logs/suite-$SLURM_JOB_ID-$n.log" 2>&1 &
    pids+=($!)
    sleep 5
done < "$suite"
# Wait for the agent runs only: a bare `wait` would also wait for the vLLM server that
# setup.sh started in the background, holding the GPU until the time limit.
wait "${pids[@]}"
echo "finished $(date)"
