# Prepare a Neuronic compute node and start vLLM. Source it (don't run it with
# bash) so the paths and Python environment stay active in your shell.
#
#   salloc -c 8 --mem=64G --gres=gpu:1 --time=02:00:00
#   source ~/agent-scaling/setup.sh            # gpt-oss-20b on 1 GPU
#
#   salloc -c 16 --mem=128G --gres=gpu:2 --time=02:00:00
#   source ~/agent-scaling/setup.sh 120b       # gpt-oss-120b split over 2 GPUs
#   source ~/agent-scaling/setup.sh qwen3-coder-30b   # Qwen3-Coder-30B-A3B-Instruct (BF16) over 2 GPUs
#   source ~/agent-scaling/setup.sh qwen3.8-27b       # Qwen3.8-27B, dense, thinking (BF16) over 2 GPUs
#   salloc -c 16 --mem=192G --gres=gpu:4 ...; source ~/agent-scaling/setup.sh qwen3.5-122b   # Qwen3.5-122B-A10B (FP8), 4 GPUs
#
# Safe to re-run: each step is skipped if it's already done on this node.
# In a second terminal on the same node, re-sourcing it just restores the environment.
# /scratch deletes files more than two weeks old but leaves their directories, so each step checks the
# files it needs rather than the directory, and redoes the step if any are gone.
# Overrides: PORT (default 8000), TP (tensor-parallel size, default = GPUs allocated),
# VLLM_VERSION (default 0.30.0, the version first tested on Neuronic), VLLM_LOG (log path),
# VLLM_GPU_UTIL (share of GPU memory for the model, default 0.9, or 0.7 with BROWSECOMP).
# BROWSECOMP=1 also sets up BrowseComp-Plus (agent.py --benchmark browsecomp): its documents, search index and
# questions in /scratch/$USER/browsecomp, and its query embedder as a second vLLM on port PORT+1 (EMBED_PORT).
# APPWORLD=1 also sets up AppWorld (agent.py --benchmark appworld): its own venv ($APPWORLD_PYTHON; its pinned
# dependencies would clash with vLLM's, so run agent.py with that Python) and its data in /scratch/$USER/appworld
# ($APPWORLD_ROOT). The data must not be redistributed unencrypted: keep it out of git and off artifact pages.
# COOPERBENCH=1 also sets up CooperBench (github.com/cooperbench/CooperBench, pinned, with cooperbench/apptainer.patch,
# which runs its containers on Apptainer since the cluster has no Docker): the source in $WORK/cooperbench-src, its own
# venv ($COOPERBENCH_BIN is its `cooperbench` command), task images unpacked on first use under $WORK/cooperbench/images.
# APPWORLD_ONLY=1 (with APPWORLD=1) skips the GPU, vLLM and the model: just uv and AppWorld, for runs against an API
# model (agent.py --model claude-..., see claude.sbatch), which need no GPU.
# For unattended runs, use run.sbatch instead, which sources this for you.

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    echo "Run this with: source $0" >&2
    exit 1
fi

_agent_setup() {
    local size="${1:-20b}"
    # The model: its served name, Hugging Face repo, vLLM tool-call parser and extra serve flags, and GPUs it needs.
    local name repo parser min_gpus=1 extra=()
    case "$size" in
        20b|120b) name="gpt-oss-$size"; repo="openai/$name"; parser=openai; [[ "$size" == 120b ]] && min_gpus=2 ;;
        qwen3-coder-30b)  # 61 GB in BF16: 2 L40s; context capped at 128k (native 256k) to leave room for several agents
            name="$size"; repo="Qwen/Qwen3-Coder-30B-A3B-Instruct"; parser=qwen3_coder; min_gpus=2
            extra=(--max-model-len 131072) ;;
        # Qwen3.5-family thinking models (Qwen3.8-27B has Qwen3.5's architecture), served as their model cards say:
        # qwen3 reasoning parser (thinking comes back separately; agent.py passes it back as reasoning_content),
        # qwen3_coder tool parser, the vision encoder skipped (--language-model-only). Context capped at 128k (native 256k).
        qwen3.8-27b)    # dense 27B, 56 GB in BF16: 2 L40s
            name="$size"; repo="Qwen/Qwen3.8-27B"; parser=qwen3_coder; min_gpus=2
            extra=(--max-model-len 131072 --reasoning-parser qwen3 --language-model-only) ;;
        qwen3.5-122b)   # Qwen3.5-122B-A10B (10B active), 127 GB in FP8: 4 L40s
            name="$size"; repo="Qwen/Qwen3.5-122B-A10B-FP8"; parser=qwen3_coder; min_gpus=4
            extra=(--max-model-len 131072 --reasoning-parser qwen3 --language-model-only) ;;
        *) echo "Unknown model '$size': use 20b, 120b, qwen3-coder-30b, qwen3.8-27b or qwen3.5-122b" >&2; return 1 ;;
    esac
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
    if [[ "$tp" -lt "$min_gpus" ]]; then
        echo "$name needs $min_gpus L40s. Allocate with --gres=gpu:$min_gpus." >&2
        return 1
    fi

    # ---- environment (everything lives on the node's local SSD) ----
    export WORK="/scratch/$USER"
    export UV_CACHE_DIR="$WORK/uv-cache" HF_HOME="$WORK/hf"
    # A Hugging Face token (free, read-only) in ~/.config/huggingface/token, chmod 600, lifts the rate limit on anonymous
    # downloads; HF_HOME is per node, so it is passed as HF_TOKEN. Never commit it.
    if [[ -z "$HF_TOKEN" && -r "$HOME/.config/huggingface/token" ]]; then
        HF_TOKEN="$(tr -d '[:space:]' < "$HOME/.config/huggingface/token")"
        export HF_TOKEN
    fi
    [[ ":$PATH:" == *":$WORK/bin:"* ]] || export PATH="$WORK/bin:$PATH"
    mkdir -p "$WORK/models" || return 1

    # Two jobs landing on one fresh node must not install or download into $WORK at the same time.
    local lock
    exec {lock}>>"$WORK/.setup.lock"  # >> so noclobber doesn't refuse the existing file
    flock "$lock"
    if [[ ! -x "$WORK/bin/uv" ]]; then
        echo "==> Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh || { exec {lock}>&-; return 1; }
    fi

    # A venv the purge emptied (or one left half-built) can't be repaired: uv stops at the packages' missing
    # metadata. So unless vllm imports, start over, with an empty download cache as well: the cache may have lost
    # files too, and uv hardlinks from it, which would give the new venv the old files' dates.
    if [[ ! -x "$WORK/venv/bin/vllm" ]] || ! "$WORK/venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
        echo "==> Installing vLLM $vllm_version (takes a few minutes on a fresh node)"
        rm -rf "$WORK/venv" "$WORK/uv-cache"
        uv venv --python 3.12 "$WORK/venv" || { exec {lock}>&-; return 1; }
        uv pip install --python "$WORK/venv/bin/python" "vllm==$vllm_version" || { exec {lock}>&-; return 1; }
        if [[ ! -x "$WORK/venv/bin/vllm" ]] || ! "$WORK/venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
            echo "vLLM install is broken in $WORK/venv" >&2; exec {lock}>&-; return 1
        fi
    fi
    source "$WORK/venv/bin/activate"

    # WorkBench benchmark (agent.py --benchmark workbench): its tools, tasks and evaluator.
    # Pinned so scores stay comparable; pandas is pinned to the version in WorkBench's uv.lock.
    export WORKBENCH_DIR="$WORK/WorkBench"
    _clone https://github.com/olly-styles/WorkBench.git "$WORKBENCH_DIR" 49c7dfd || { exec {lock}>&-; return 1; }
    if ! python -c "import pandas" 2>/dev/null; then
        echo "==> Installing pandas for WorkBench"
        uv pip install --python "$WORK/venv/bin/python" "pandas==2.3.3" || { exec {lock}>&-; return 1; }
    fi

    local model_dir="$WORK/models/$name"
    if ! _intact "$model_dir"; then
        echo "==> Downloading $repo"
        # Many jobs starting on fresh nodes at once can hit Hugging Face's rate limit for anonymous downloads (the
        # cluster shares one IP): wait and retry, with growing pauses (2, 4, ... 14 min; about an hour in all).
        local tries=0
        until hf download "$repo" --exclude "original/*" --exclude "metal/*" --local-dir "$model_dir"; do
            tries=$((tries + 1))
            if (( tries >= 8 )); then echo "Downloading $repo failed $tries times" >&2; exec {lock}>&-; return 1; fi
            echo "==> Download failed (Hugging Face's rate limit?); retrying in $((tries * 2)) minutes" >&2
            sleep $((tries * 120))
        done
        _mark_intact "$model_dir"
    fi
    if [[ -n "$BROWSECOMP" ]]; then
        _browsecomp_data || { exec {lock}>&-; return 1; }
    fi
    if [[ -n "$APPWORLD" ]]; then
        _appworld || { exec {lock}>&-; return 1; }
    fi
    if [[ -n "$COOPERBENCH" ]]; then
        _cooperbench || { exec {lock}>&-; return 1; }
    fi
    exec {lock}>&-  # closing the file releases the lock

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
        pid=$(pgrep -u "$USER" -f "vllm serve .*--port $port( |$)" | head -n 1)
        if [[ -n "$pid" ]]; then
            echo "==> vLLM is still starting (pid $pid); waiting for it"
        else
            echo "==> Starting vLLM: $name, tensor-parallel $tp, port $port"
            echo "    log: $log"
            # >| overwrites the old log even with noclobber on
            nohup vllm serve "$model_dir" --served-model-name "$name" --port "$port" \
                --tensor-parallel-size "$tp" --gpu-memory-utilization "${VLLM_GPU_UTIL:-$( [[ -n "$BROWSECOMP" ]] && echo 0.70 || echo 0.90 )}" \
                --enable-auto-tool-choice --tool-call-parser "$parser" "${extra[@]}" >| "$log" 2>&1 &
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
            if (( SECONDS - start > 1500 )); then  # the 127 GB model over 4 GPUs takes a while
                echo
                echo "Still not up after 25 minutes. Check: tail -f $log" >&2
                return 1
            fi
            sleep 5
            echo -n "."
        done
        echo " ready ($(( SECONDS - start ))s)"
    fi

    if [[ -n "$BROWSECOMP" ]]; then
        export EMBED_PORT="${EMBED_PORT:-$((port + 1))}"
        _browsecomp_embedder || return 1
    fi

    echo
    echo "Model:  $name at http://localhost:$port/v1"
    if [[ -n "$SLURM_JOB_ID" ]]; then
        echo "Job:    $SLURM_JOB_ID on $(hostname -s), ends $(squeue -h -j "$SLURM_JOB_ID" -o %e)"
    fi
    echo "Next:   python ~/agent-scaling/agent.py"
}

# A download is complete when its .complete marker lists the files it fetched and they're all still there.
# (Markers from before this check are empty, so those downloads are checked again once; hf skips files it has.)
_intact() {
    local f
    [[ -s "$1/.complete" ]] || return 1
    while IFS= read -r f; do
        [[ -f "$1/$f" ]] || return 1
    done < "$1/.complete"
}

_mark_intact() {
    (cd "$1" && find . -type f ! -name .complete ! -path "*/.cache/*" -printf '%P\n') >| "$1/.complete"
}

# A git clone at a pinned commit. A clone the purge emptied still has its .git directory, so check that git can
# read it, that it's at the commit, and that no tracked file is missing; otherwise clone it again.
_clone() {
    local url="$1" dir="$2" commit="$3" changes
    if changes=$(git -C "$dir" status --porcelain --untracked-files=no 2>/dev/null) && [[ -z "$changes" ]] \
        && [[ "$(git -C "$dir" rev-parse HEAD 2>/dev/null)" == "$commit"* ]]; then
        return 0
    fi
    echo "==> Cloning $url"
    rm -rf "$dir"
    git clone -q "$url" "$dir" && git -C "$dir" checkout -q "$commit"
}

# BrowseComp-Plus data: documents, the Qwen3-Embedding-4B index, the questions (decrypted on this node only),
# the embedding model, and the paper's code for its 100-question sample and grader prompt.
_browsecomp_data() {
    export BROWSECOMP_DIR="$WORK/browsecomp" PAPER_REPO="$WORK/agent-scaling-paper"
    local d="$BROWSECOMP_DIR"
    mkdir -p "$d" || return 1
    if ! _intact "$d"; then
        echo "==> Downloading BrowseComp-Plus (about 6 GB)"
        hf download Tevatron/browsecomp-plus-corpus --repo-type dataset --local-dir "$d/corpus" || return 1
        hf download Tevatron/browsecomp-plus-indexes --repo-type dataset --include "qwen3-embedding-4b/*" \
            --local-dir "$d/indexes" || return 1
        hf download Tevatron/browsecomp-plus --repo-type dataset --local-dir "$d/queries" || return 1
        _mark_intact "$d"
    fi
    if ! _intact "$WORK/models/Qwen3-Embedding-4B"; then
        echo "==> Downloading Qwen/Qwen3-Embedding-4B"
        hf download Qwen/Qwen3-Embedding-4B --local-dir "$WORK/models/Qwen3-Embedding-4B" || return 1
        _mark_intact "$WORK/models/Qwen3-Embedding-4B"
    fi
    _clone https://github.com/ybkim95/agent-scaling.git "$PAPER_REPO" 6f3bfb7 || return 1
    python -c "import pyarrow" 2>/dev/null || uv pip install --python "$WORK/venv/bin/python" pyarrow || return 1
    [[ -f "$d/questions.jsonl" ]] || python "$(dirname "${BASH_SOURCE[0]}")/browsecomp.py" || return 1
}

# The query embedder: Qwen3-Embedding-4B as a pooling model on the same GPU, next to the chat model.
_browsecomp_embedder() {
    local log="${VLLM_LOG:-$WORK/vllm.log}"
    log="${log%.log}-embed.log"
    if curl -sf "localhost:$EMBED_PORT/health" >/dev/null; then
        # Something already answers on this port: use it only if it is the embedder.
        if ! curl -s "localhost:$EMBED_PORT/v1/models" | grep -q '"qwen3-embedding-4b"'; then
            echo "Port $EMBED_PORT is serving something other than the query embedder. Set EMBED_PORT to a free port." >&2
            return 1
        fi
        echo "==> The query embedder is already running on port $EMBED_PORT"
    else
        echo "==> Starting the query embedder on port $EMBED_PORT"
        nohup vllm serve "$WORK/models/Qwen3-Embedding-4B" --served-model-name qwen3-embedding-4b --port "$EMBED_PORT" \
            --runner pooling --gpu-memory-utilization 0.25 --max-model-len 2048 >| "$log" 2>&1 &
        local pid=$! start=$SECONDS
        until curl -sf "localhost:$EMBED_PORT/health" >/dev/null; do
            kill -0 "$pid" 2>/dev/null || { echo "embedder exited; see $log" >&2; tail -n 20 "$log" >&2; return 1; }
            (( SECONDS - start > 900 )) && { echo "embedder not up after 15 minutes; see $log" >&2; return 1; }
            sleep 5
        done
        echo "    embedder ready ($(( SECONDS - start ))s)"
    fi
}

# AppWorld: its package in a venv of its own, the encrypted part of its code unpacked (`appworld install`), and its
# data. The venv is rebuilt if appworld or openai no longer imports (e.g. after the /scratch purge).
_appworld() {
    export APPWORLD_ROOT="$WORK/appworld" APPWORLD_PYTHON="$WORK/appworld-venv/bin/python"
    local version="${APPWORLD_VERSION:-0.1.3.post1}"
    if ! "$APPWORLD_PYTHON" -c "import appworld, openai" >/dev/null 2>&1 || [[ ! -f "$WORK/appworld-venv/.unpacked" ]]; then
        echo "==> Installing AppWorld $version"
        rm -rf "$WORK/appworld-venv"
        uv venv --python 3.11 "$WORK/appworld-venv" || return 1
        uv pip install --python "$APPWORLD_PYTHON" "appworld==$version" openai anthropic || return 1
        (cd "$WORK" && "$WORK/appworld-venv/bin/appworld" install) || return 1
        touch "$WORK/appworld-venv/.unpacked"
    fi
    # The Anthropic SDK, for Claude models (claude_agent.py); venvs built before it was added get it here.
    if ! "$APPWORLD_PYTHON" -c "import anthropic" >/dev/null 2>&1; then
        uv pip install --python "$APPWORLD_PYTHON" anthropic || return 1
    fi
    mkdir -p "$APPWORLD_ROOT" || return 1
    if ! _intact "$APPWORLD_ROOT/data"; then
        echo "==> Downloading AppWorld data"
        (cd "$APPWORLD_ROOT" && "$WORK/appworld-venv/bin/appworld" download data --root "$APPWORLD_ROOT") || return 1
        _mark_intact "$APPWORLD_ROOT/data"
    fi
}

# CooperBench, pinned, with our Apptainer backend (cooperbench/apptainer.patch), in a venv of its own (Python 3.12).
_cooperbench() {
    local commit="${COOPERBENCH_COMMIT:-63b9d44}" src="$WORK/cooperbench-src" patch="$HOME/agent-scaling/cooperbench/apptainer.patch"
    export COOPERBENCH_SRC="$src" COOPERBENCH_BIN="$WORK/cooperbench-venv/bin/cooperbench" COOPERBENCH_PYTHON="$WORK/cooperbench-venv/bin/python"
    export APPTAINER_CACHEDIR="$WORK/cooperbench/apptainer-cache" APPTAINER_TMPDIR="$WORK/cooperbench/apptainer-tmp"
    mkdir -p "$APPTAINER_CACHEDIR" "$APPTAINER_TMPDIR" || return 1
    if [[ ! -d "$src/.git" ]]; then
        echo "==> Cloning CooperBench ($commit)"
        git clone -q https://github.com/cooperbench/CooperBench.git "$src" || return 1
        git -C "$src" checkout -q "$commit" || return 1
    fi
    if ! git -C "$src" apply --check -R "$patch" 2>/dev/null; then  # not applied yet, or an earlier version of it is
        git -C "$src" checkout -q -- . && git -C "$src" clean -qfd -- src  # back to the pinned commit (ignored files stay)
        git -C "$src" apply "$patch" || { echo "cooperbench/apptainer.patch doesn't apply to $src" >&2; return 1; }
    fi
    if ! "$WORK/cooperbench-venv/bin/python" -c "import cooperbench.infra.apptainer" >/dev/null 2>&1; then
        echo "==> Installing CooperBench"
        rm -rf "$WORK/cooperbench-venv"
        uv venv -q --python 3.12 "$WORK/cooperbench-venv" || return 1
        uv pip install -q --python "$WORK/cooperbench-venv/bin/python" -e "$src" || return 1
    fi
    if ! "$WORK/cooperbench-venv/bin/python" -c "import anthropic" >/dev/null 2>&1; then  # Claude runs (claude.sbatch)
        uv pip install -q --python "$WORK/cooperbench-venv/bin/python" anthropic || return 1
    fi
}

# AppWorld only (APPWORLD_ONLY=1): no GPU or vLLM, for runs against an API model.
# No GPU, vLLM or model (APPWORLD_ONLY=1, or API_ONLY=1): just uv and the benchmarks asked for with APPWORLD=1 and/or
# COOPERBENCH=1, for runs against an API model (claude.sbatch).
_appworld_only_setup() {
    if [[ "$(hostname -s)" == "neuronic" ]]; then
        echo "You're on the login node. Run salloc (no GPU needed) or sbatch claude.sbatch." >&2
        return 1
    fi
    export WORK="/scratch/$USER"
    export UV_CACHE_DIR="$WORK/uv-cache"
    [[ ":$PATH:" == *":$WORK/bin:"* ]] || export PATH="$WORK/bin:$PATH"
    mkdir -p "$WORK" || return 1
    local lock
    exec {lock}>>"$WORK/.setup.lock"
    flock "$lock"
    if [[ ! -x "$WORK/bin/uv" ]]; then
        echo "==> Installing uv"
        curl -LsSf https://astral.sh/uv/install.sh \
            | env UV_INSTALL_DIR="$WORK/bin" UV_NO_MODIFY_PATH=1 sh || { exec {lock}>&-; return 1; }
    fi
    if [[ -n "$APPWORLD" || -n "$APPWORLD_ONLY" ]]; then
        _appworld || { exec {lock}>&-; return 1; }
        echo "AppWorld ready: $APPWORLD_PYTHON (data in $APPWORLD_ROOT)"
    fi
    if [[ -n "$COOPERBENCH" ]]; then
        _cooperbench || { exec {lock}>&-; return 1; }
        echo "CooperBench ready: $COOPERBENCH_PYTHON"
    fi
    exec {lock}>&-
}

# Last command, so `source setup.sh || exit 1` sees whether setup succeeded
if [[ -n "$APPWORLD_ONLY" || -n "$API_ONLY" ]]; then
    _appworld_only_setup
else
    _agent_setup "$@"
fi
