#!/bin/bash
# CPU job: analyze the repeated 64-task hard suite and rebuild the hard-task page.
# Submitted with --dependency=afterany:<the suite jobs>; needs no GPU.
#
#   sbatch --dependency=afterany:$(tr ' ' : < logs/repeats-jobids.txt) analysis/repeats.sbatch
#SBATCH --job-name=repeats-analysis
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

cd "$SLURM_SUBMIT_DIR" || exit 1
echo "job $SLURM_JOB_ID on $(hostname -s), commit $(git rev-parse --short HEAD), started $(date)"
WORK="/scratch/$USER"
export WORKBENCH_DIR="$WORK/WorkBench"
mkdir -p "$WORK"
# Scoring only needs WorkBench and pandas: use the vLLM venv if this node has one, else a small one.
py="$WORK/venv/bin/python"
if ! "$py" -c "import pandas, openai" 2>/dev/null; then
    py="$WORK/venv-cpu/bin/python"
    if ! "$py" -c "import pandas, openai" 2>/dev/null; then
        [[ -x "$WORK/bin/uv" ]] || curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh || exit 1
        "$WORK/bin/uv" venv --allow-existing --python 3.12 "$WORK/venv-cpu" || exit 1
        "$WORK/bin/uv" pip install --python "$py" "pandas==2.3.3" openai || exit 1
    fi
fi
if [[ ! -d "$WORKBENCH_DIR/.git" ]]; then
    git clone -q https://github.com/olly-styles/WorkBench.git "$WORKBENCH_DIR" || exit 1
    git -C "$WORKBENCH_DIR" checkout -q 49c7dfd || exit 1
fi
"$py" analysis/hard64_repeats.py && "$py" analysis/build_hard64_page.py
echo "finished $(date)"
