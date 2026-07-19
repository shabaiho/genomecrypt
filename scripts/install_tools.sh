#!/usr/bin/env bash
# Install the non-Python tools AMRFinderPlus needs, into ./.tools -- no docker,
# no conda, no root. uv handles every Python dependency; this handles the three
# binaries uv cannot install.
#
#   ./scripts/install_tools.sh          # install everything + fetch the AMR database
#   source .tools/env.sh                # put them on PATH for the current shell
#
# Costs ~350MB on disk and ~5 min the first time. Re-running is a no-op for
# anything already installed.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOLS="$ROOT/.tools"
BIN="$TOOLS/bin"
mkdir -p "$BIN" "$TOOLS/src"

# AMRFinderPlus ships prebuilt binaries only. v4.x is linked against
# GLIBCXX_3.4.32 (gcc 13+, i.e. Ubuntu 24.04+); on older distros it dies with a
# linker error. We try newest-first and fall back to the last v3 release, which
# builds against GLIBCXX_3.4.30. features.py normalizes both output schemas, so
# either version produces identical downstream features.
AMR_VERSIONS="${AMR_VERSIONS:-4.2.7 4.0.23 3.12.8}"
BLAST_VERSION="${BLAST_VERSION:-2.17.0}"
HMMER_VERSION="${HMMER_VERSION:-3.4}"
CURL="curl -fL --retry 3 --progress-bar"

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }

# --- BLAST+ : amrfinder shells out to blastn/blastp/tblastn/makeblastdb -------
if [ ! -x "$BIN/blastn" ]; then
  say "BLAST+ $BLAST_VERSION (~250MB download)"
  url="https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/$BLAST_VERSION/ncbi-blast-$BLAST_VERSION+-x64-linux.tar.gz"
  $CURL "$url" -o "$TOOLS/src/blast.tar.gz"
  tar -xzf "$TOOLS/src/blast.tar.gz" -C "$TOOLS/src"
  # keep only what amrfinder calls -- the full suite is ~1.5GB unpacked
  for b in blastn blastp tblastn makeblastdb blastx; do
    cp "$TOOLS/src/ncbi-blast-$BLAST_VERSION+/bin/$b" "$BIN/"
  done
  rm -rf "$TOOLS/src/ncbi-blast-$BLAST_VERSION+" "$TOOLS/src/blast.tar.gz"
else
  echo "BLAST+ already installed"
fi

# --- HMMER : only distributed as source, so we compile it (needs gcc + make) --
if [ ! -x "$BIN/hmmsearch" ]; then
  say "HMMER $HMMER_VERSION (compiling, ~2 min)"
  command -v gcc >/dev/null || { echo "gcc required to build HMMER"; exit 1; }
  curl -fL --retry 3 "http://eddylab.org/software/hmmer/hmmer-$HMMER_VERSION.tar.gz" \
    -o "$TOOLS/src/hmmer.tar.gz"
  tar -xzf "$TOOLS/src/hmmer.tar.gz" -C "$TOOLS/src"
  (
    cd "$TOOLS/src/hmmer-$HMMER_VERSION"
    ./configure --prefix="$TOOLS" >/dev/null
    make -j"$(nproc)" >/dev/null
    make install >/dev/null
  )
  rm -rf "$TOOLS/src/hmmer-$HMMER_VERSION" "$TOOLS/src/hmmer.tar.gz"
else
  echo "HMMER already installed"
fi

# --- AMRFinderPlus ------------------------------------------------------------
if [ ! -x "$BIN/amrfinder" ]; then
  installed=""
  for v in $AMR_VERSIONS; do
    say "AMRFinderPlus $v"
    rm -rf "$TOOLS/amrfinder"; mkdir -p "$TOOLS/amrfinder"
    url="https://github.com/ncbi/amr/releases/download/amrfinder_v$v/amrfinder_binaries_v$v.tar.gz"
    if ! $CURL "$url" -o "$TOOLS/src/amrfinder.tar.gz"; then
      echo "  download failed, trying older release"; continue
    fi
    tar -xzf "$TOOLS/src/amrfinder.tar.gz" -C "$TOOLS/amrfinder"
    rm -f "$TOOLS/src/amrfinder.tar.gz"
    # verify it actually RUNS here -- a prebuilt binary that links against a
    # newer libstdc++ downloads fine and then fails at the first invocation
    if "$TOOLS/amrfinder/amrfinder" --version >/dev/null 2>&1; then
      installed="$v"
      break
    fi
    echo "  v$v is not runnable on this system (libstdc++ too old), falling back"
  done
  [ -n "$installed" ] || { echo "no runnable AMRFinderPlus release found"; exit 1; }
  # the release ships loose binaries + data/; symlink them onto our PATH
  for f in "$TOOLS/amrfinder"/*; do
    [ -f "$f" ] && [ -x "$f" ] && ln -sf "$f" "$BIN/$(basename "$f")"
  done
  echo "installed AMRFinderPlus $installed"
else
  echo "AMRFinderPlus already installed ($("$BIN/amrfinder" --version 2>/dev/null))"
fi

# --- env.sh -------------------------------------------------------------------
cat > "$TOOLS/env.sh" <<EOF
# source this to use the tools installed by scripts/install_tools.sh
export PATH="$BIN:\$PATH"
EOF

export PATH="$BIN:$PATH"

# --- AMR database (~200MB, versioned -- pin it in your writeup) ----------------
say "AMR database"
if [ -d "$TOOLS/amrfinder/data" ] && [ -n "$(ls -A "$TOOLS/amrfinder/data" 2>/dev/null)" ]; then
  echo "database present:"; ls "$TOOLS/amrfinder/data"
else
  amrfinder -u
fi

say "done"
echo "run:  source .tools/env.sh  &&  amrfinder --version"
