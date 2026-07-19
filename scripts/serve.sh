#!/usr/bin/env bash
# Run both pages: the landing on 8600 and the demo on 8501.
#
# The demo is supervised. Streamlit died three times during development -- twice
# from two AMRFinderPlus runs overlapping, which is now prevented by a lock, but a
# live demo should not depend on that having been the only cause. If it exits, it
# comes back within a second and the URL keeps working.
#
#   ./scripts/serve.sh            both
#   ./scripts/serve.sh demo       demo only
#   ./scripts/serve.sh landing    landing only
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

DEMO_PORT="${DEMO_PORT:-8501}"
LANDING_PORT="${LANDING_PORT:-8600}"
WHAT="${1:-both}"

[ -f .tools/env.sh ] && . .tools/env.sh

# One thread per BLAS operation: the models are tiny, and the extra threads only
# compete with the annotation subprocess for memory.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=()
cleanup() { for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done; }
trap cleanup EXIT INT TERM

start_landing() {
  echo "landing  http://localhost:$LANDING_PORT"
  python3 -m http.server "$LANDING_PORT" --directory landing >/dev/null 2>&1 &
  pids+=($!)
}

start_demo() {
  echo "demo     http://localhost:$DEMO_PORT"
  (
    while true; do
      uv run streamlit run app/streamlit_app.py \
        --server.address=0.0.0.0 --server.port="$DEMO_PORT" \
        --server.headless=true --browser.gatherUsageStats=false \
        >> /tmp/genome-firewall-demo.log 2>&1
      code=$?
      echo "[$(date +%H:%M:%S)] demo exited ($code), restarting" \
        >> /tmp/genome-firewall-demo.log
      sleep 1
    done
  ) &
  pids+=($!)
}

case "$WHAT" in
  demo)    start_demo ;;
  landing) start_landing ;;
  *)       start_landing; start_demo ;;
esac

echo "log      /tmp/genome-firewall-demo.log"
echo "ctrl-c to stop"
wait
