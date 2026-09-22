# Prepare a Neuronic compute node and start vLLM. Source it (don't run it with
# bash) so the paths and Python environment stay active in your shell.
#
#   salloc -c 8 --mem=64G --gres=gpu:1 --time=02:00:00
#   source ~/agent-scaling/setup.sh            # gpt-oss-20b on 1 GPU
#
#   salloc -c 16 --mem=128G --gres=gpu:2 --time=02:00:00
#   source ~/agent-scaling/setup.sh 120b       # gpt-oss-120b split over 2 GPUs
#
# Safe to re-run: each step is skipped if it's already done on this node.
# In a second terminal on the same node, re-sourcing it just restores the environment.
# Overrides: PORT (default 8000), TP (tensor-parallel size, default = GPUs allocated),
# VLLM_VERSION (default 0.30.0, the version first tested on Neuronic), VLLM_LOG (log path).
# For unattended runs, use run.sbatch instead, which sources this for you.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with: source $0" >&2
    exit 1
fi

_agent_setup() {
    local size="${1:-20b}"
    local name="gpt-oss-$size"
    local port="${PORT:-8000}"
    local vllm_version="${VLLM_VERSION:-0.30.0}"

    # ---- sanity checks ----
    if [[ "$(hostname -s)" == "neuronic" ]]; then
        echo "You're on the login node. Run salloc first, then source this on the compute node." >&2
        return 1
    fi
    if ! nvidia-smi -L >/dev/null 2>&1; then
        echo "No GPU visible. Did you pass --gres=gpu:N to salloc?" >&2
        return 1
    fi
    local gpus tp
    gpus=$(nvidia-smi -L | wc -l)
    tp="${TP:-$gpus}"
    if [[ "$size" == "120b" && "$tp" -lt 2 ]]; then
        echo "gpt-oss-120b needs 2 L40s. Allocate with --gres=gpu:2." >&2
        return 1
    fi

    # ---- environment (everything lives on the node's local SSD) ----
    export WORK="/scratch/$USER"
    export UV_CACHE_DIR="$WORK/uv-cache" HF_HOME="$WORK/hf"
    [[ ":$PATH:" == *":$WORK/bin:"* ]] || export PATH="$WORK/bin:$PATH"
    mkdir -p "$WORK/models" || return 1

    if [[ ! -x "$WORK/bin/uv" ]]; then
        echo "==> Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh || return 1
    fi

    if [[ ! -x "$WORK/venv/bin/vllm" ]]; then
        echo "==> Installing vLLM $vllm_version (takes a few minutes on a fresh node)"
        uv venv --allow-existing --python 3.12 "$WORK/venv" || return 1
        uv pip install --python "$WORK/venv/bin/python" "vllm==$vllm_version" || return 1
    fi
    source "$WORK/venv/bin/activate"

    local model_dir="$WORK/models/$name"
    if [[ ! -f "$model_dir/.complete" ]]; then
        echo "==> Downloading openai/$name"
        hf download "openai/$name" --exclude "original/*" --exclude "metal/*" \
            --local-dir "$model_dir" || return 1
        touch "$model_dir/.complete"
    fi

    # ---- vLLM server ----
    if curl -sf "localhost:$port/health" >/dev/null; then
        if curl -s "localhost:$port/v1/models" | grep -q "\"$name\""; then
            echo "==> vLLM is already serving $name on port $port"
        else
            echo "A different model is running on port $port. Stop it with:" >&2
            echo "    pkill -u $USER -f 'vllm serve'" >&2
            echo "then source this script again." >&2
            return 1
        fi
    else
        local log="${VLLM_LOG:-$WORK/vllm-$name.log}"
        local pid
        pid=$(pgrep -u "$USER" -f "vllm serve" | head -n 1)
        if [[ -n "$pid" ]]; then
            echo "==> vLLM is still starting (pid $pid); waiting for it"
        else
            echo "==> Starting vLLM: $name, tensor-parallel $tp, port $port"
            echo "    log: $log"
            # >| overwrites the old log even with noclobber on
            nohup vllm serve "$model_dir" --served-model-name "$name" --port "$port" \
                --tensor-parallel-size "$tp" \
                --enable-auto-tool-choice --tool-call-parser openai >| "$log" 2>&1 &
            pid=$!
        fi
        local start=$SECONDS
        echo -n "    waiting for the server"
        until curl -sf "localhost:$port/health" >/dev/null; do
            if ! kill -0 "$pid" 2>/dev/null; then
                echo
                echo "vLLM exited. Last lines of $log:" >&2
                tail -n 30 "$log" >&2
                return 1
            fi
            if (( SECONDS - start > 900 )); then
                echo
                echo "Still not up after 15 minutes. Check: tail -f $log" >&2
                return 1
            fi
            sleep 5
            echo -n "."
        done
        echo " ready ($(( SECONDS - start ))s)"
    fi

    echo
    echo "Model:  $name at http://localhost:$port/v1"
    if [[ -n "$SLURM_JOB_ID" ]]; then
        echo "Job:    $SLURM_JOB_ID on $(hostname -s), ends $(squeue -h -j "$SLURM_JOB_ID" -o %e)"
    fi
    echo "Next:   python ~/agent-scaling/agent.py"
}

# Last command, so `source setup.sh || exit 1` sees whether setup succeeded
_agent_setup "$@"
