# Shared helpers for the inference scripts (sourced, not run directly).

ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd)"
CKPT_DIR="$ROOT/inference/checkpoints"

# Download the required checkpoints from HuggingFace (IVPL/CaReDiff).
# Files already present under inference/checkpoints/ are not downloaded again.
download_checkpoints() {  # $1: pattern, e.g. "generic/offline/*"
    python - "$1" "$CKPT_DIR" <<'EOF'
import sys
from huggingface_hub import snapshot_download
pattern, local_dir = sys.argv[1], sys.argv[2]
snapshot_download("IVPL/CaReDiff", allow_patterns=[pattern], local_dir=local_dir)
print(f"[inference] checkpoints ready: {local_dir}/{pattern.rstrip('/*')}")
EOF
}

# The post-processor weights are loaded on every stage=test run but are
# distributed separately (see inference/README.md, step 1).
require_pretrained_models() {  # $1: track dir (generic|personalised)
    if [ ! -f "$ROOT/$1/code/pretrained_models/post_processor/checkpoint.pth" ]; then
        echo "ERROR: $ROOT/$1/code/pretrained_models/post_processor/checkpoint.pth not found."
        echo "Download the 'pretrained_models' archive (link in inference/README.md, step 1)"
        echo "and extract it into $ROOT/$1/code/ first."
        exit 1
    fi
}

require_data_dir() {
    if [ ! -d "$DATA_DIR/test" ]; then
        echo "ERROR: '$DATA_DIR/test' not found."
        echo "Set DATA_DIR (first argument, or the variable at the top of this script)"
        echo "to a dataset root that contains a test/ split in the MARS layout."
        echo "See inference/README.md for the expected directory structure."
        exit 1
    fi
}
