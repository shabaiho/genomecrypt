#!/usr/bin/env bash
# Point the landing page's demo buttons at a deployed Streamlit instance.
#
#   ./deploy/set-demo-url.sh https://user-genome-firewall.hf.space
#   ./deploy/set-demo-url.sh                    # back to localhost:8501
#
# The landing ships with localhost so it works before anything is deployed. That
# is also why this script exists: a landing deployed with localhost buttons sends
# every visitor nowhere, and nothing about the page looks wrong.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FILE="$ROOT/landing/index.html"
NEW="${1:-http://localhost:8501}"

[ -f "$FILE" ] || { echo "not found: $FILE"; exit 1; }

# strip a trailing slash only from a full URL; a bare path like /app is passed
# through untouched, since stripping it there silently rewrote /app/ back to /app
case "$NEW" in
  http*) NEW="${NEW%/}" ;;
esac

# the current target may be a full URL or a bare path like /app
current=$(grep -oE 'href="(https?://|/)[^"#]*"' "$FILE" \
          | grep -vE 'github|ncbi|bv-brc' | head -1 | sed 's/href="//; s/"//')

if [ -z "$current" ]; then
  echo "no demo link found in $FILE"
  exit 1
fi

sed -i "s|href=\"$current\"|href=\"$NEW\"|g" "$FILE"
n=$(grep -c "href=\"$NEW\"" "$FILE")

echo "was: $current"
echo "now: $NEW"
echo "$n link(s) updated in landing/index.html"

case "$NEW" in
  *localhost*) echo "note: localhost only works on the machine running the demo" ;;
esac
