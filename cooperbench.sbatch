#!/bin/bash
# Batch job: CooperBench (cooperbench/apptainer.patch: containers on Apptainer) against our vLLM model.
# Submit from ~/agent-scaling; the arguments go to `cooperbench run` (the model and backend are set here):
#
#   sbatch cooperbench.sbatch -n solo-20b-core --setting solo -s core
#   sbatch cooperbench.sbatch -n smoke -r go_chi_task -t 26 -f 1,2 --setting solo
#
# The agent is CooperBench's mini_swe_agent_v2 on gpt-oss-20b (cooperbench/gpt-oss-20b.yaml). Logs are written on the
# node and copied to runs/cooperbench/<name>/ at the end (CooperBench's dataset license is unknown: keep its task
# contents out of git and off pages).
#SBATCH --job-name=cooperbench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), started $(date)"
export VLLM_LOG="$SLURM_SUBMIT_DIR/logs/vllm-$SLURM_JOB_ID.log"
export PORT=$((20000 + SLURM_JOB_ID % 10000 * 2))
export COOPERBENCH=1
source ./setup.sh 20b || exit 1
export HOSTED_VLLM_API_BASE="http://localhost:$PORT/v1" HOSTED_VLLM_API_KEY=EMPTY

logs="$WORK/cooperbench/logs"
cd "$COOPERBENCH_SRC" || exit 1
"$COOPERBENCH_BIN" run -a mini_swe_agent_v2 -m hosted_vllm/gpt-oss-20b --backend apptainer \
    --agent-config "$SLURM_SUBMIT_DIR/cooperbench/gpt-oss-20b.yaml" --log-dir "$logs" "$@"
status=$?
mkdir -p "$SLURM_SUBMIT_DIR/runs/cooperbench"
cp -r "$logs"/. "$SLURM_SUBMIT_DIR/runs/cooperbench/"
echo "finished $(date), exit $status"
