#!/usr/bin/env bash
# Everything runs on this machine; one HTTPS address reaches it from anywhere.
#
#   ./scripts/public.sh
#
#       landing   https://<name>.trycloudflare.com/
#       demo      https://<name>.trycloudflare.com/app
#
# Three processes: Streamlit under the /app path, Caddy joining the landing and
# the demo into one origin, and a Cloudflare tunnel publishing that origin over
# HTTPS. Nothing is deployed and no account is needed.
#
# ABOUT THE ADDRESS. A quick tunnel keeps its name for as long as the process
# lives, and takes a new one when restarted. To fix it permanently you need a
# Cloudflare account and a domain:
#
#   cloudflared tunnel login
#   cloudflared tunnel create genome-firewall
#   cloudflared tunnel route dns genome-firewall demo.yourdomain.com
#   cloudflared tunnel run --url http://localhost:8080 genome-firewall
#
# Until then, restart it as rarely as possible and copy the address once.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export PROXY_PORT="${PROXY_PORT:-8080}"
export DEMO_PORT="${DEMO_PORT:-8501}"
export LANDING_DIR="${LANDING_DIR:-landing}"

BIN="$ROOT/.tools/bin"
LOG=/tmp/genome-firewall
[ -f .tools/env.sh ] && . .tools/env.sh

for tool in cloudflared caddy; do
  [ -x "$BIN/$tool" ] || { echo "missing $BIN/$tool -- see deploy/DEPLOY.md"; exit 1; }
done

# One thread per BLAS op: the models are tiny and the threads only compete with
# the annotation subprocess for memory.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

pids=()
cleanup() {
  for p in "${pids[@]:-}"; do kill "$p" 2>/dev/null; done
  wait 2>/dev/null
}
trap cleanup EXIT INT TERM

echo "starting Streamlit under /app ..."
(
  while true; do
    uv run streamlit run app/streamlit_app.py \
      --server.address=127.0.0.1 --server.port="$DEMO_PORT" \
      --server.baseUrlPath=app \
      --server.headless=true --browser.gatherUsageStats=false \
      >> "$LOG-demo.log" 2>&1
    echo "[$(date +%H:%M:%S)] demo exited ($?), restarting" >> "$LOG-demo.log"
    sleep 1
  done
) &
pids+=($!)

echo "starting Caddy on :$PROXY_PORT ..."
"$BIN/caddy" run --config deploy/Caddyfile --adapter caddyfile \
  >> "$LOG-caddy.log" 2>&1 &
pids+=($!)

# wait for the origin before publishing it, so the tunnel does not come up
# pointing at a port nothing is listening on
for _ in $(seq 30); do
  curl -sf -o /dev/null "http://localhost:$PROXY_PORT/" && break
  sleep 1
done

echo "opening the tunnel ..."
"$BIN/cloudflared" tunnel --url "http://localhost:$PROXY_PORT" --no-autoupdate \
  > "$LOG-tunnel.log" 2>&1 &
pids+=($!)

URL=""
for _ in $(seq 40); do
  URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$LOG-tunnel.log" \
        | head -1 || true)
  [ -n "$URL" ] && break
  sleep 1
done

echo
if [ -n "$URL" ]; then
  echo "  landing   $URL/"
  echo "  demo      $URL/app"
else
  echo "  tunnel did not report an address; see $LOG-tunnel.log"
fi
echo "  local     http://localhost:$PROXY_PORT/"
echo
echo "  logs      $LOG-demo.log  $LOG-caddy.log  $LOG-tunnel.log"
echo "  ctrl-c to stop everything"
wait
