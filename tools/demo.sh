#!/usr/bin/env bash
# Capture the terminal demo: six real commands, real output, real exit codes.
# Writes docs/evidence/demo-manifest.json plus the captured transcript the
# GIF renderer replays. Nothing here is staged; a command's exit code is
# recorded as it happened, including the refusal that exits 2 on purpose.
set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_DIR="docs/evidence"
CAPTURE="$OUT_DIR/demo-capture.txt"
MANIFEST="$OUT_DIR/demo-manifest.json"
RUN=/tmp/demo_run.json
TINY=/tmp/demo_tiny.json
mkdir -p "$OUT_DIR"
: > "$CAPTURE"

declare -a CMDS=(
  "stageattr run --out /tmp/demo_run.json --seeds 3"
  "stageattr verdict --run /tmp/demo_run.json"
  "stageattr attribute --run /tmp/demo_run.json --qid q012"
  "stageattr run --out /tmp/demo_tiny.json --mix faithful=0.97,ignore=0.01,wrong_chunk=0.01,abstain=0.01 --k-ctx 8 && stageattr verdict --run /tmp/demo_tiny.json"
  "python3 tools/check_numbers.py"
  "python3 tools/prove_checker.py"
)

codes=()
for cmd in "${CMDS[@]}"; do
  {
    echo "\$ $cmd"
    bash -c "$cmd" 2>&1
    code=$?
    echo "[exit $code]"
    echo ""
  } >> "$CAPTURE"
  codes+=("$code")
done

{
  echo "{"
  echo '  "note": "replay of captured stdout; exit codes recorded as they happened",'
  echo '  "segments": ['
  for i in "${!CMDS[@]}"; do
    sep=","
    [ "$i" -eq $((${#CMDS[@]} - 1)) ] && sep=""
    printf '    {"command": %s, "exit_code": %s}%s\n' \
      "$(python3 -c 'import json,sys;print(json.dumps(sys.argv[1]))' "${CMDS[$i]}")" \
      "${codes[$i]}" "$sep"
  done
  echo "  ]"
  echo "}"
} > "$MANIFEST"

rm -f "$RUN" "$TINY"
echo "captured ${#CMDS[@]} segments to $CAPTURE"
