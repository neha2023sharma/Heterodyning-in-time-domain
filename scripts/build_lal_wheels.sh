#!/usr/bin/env bash
# Build lal + lalsimulation from the custom fork and produce local wheels.
#
# Usage (from repo root):
#   bash scripts/build_lal_wheels.sh
#
# After this succeeds, run the full test suite with:
#   uv run --group test-integration pytest tests/test_likelihood.py -v

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAL_FORK="https://github.com/neha2023sharma/lalsuite.git"
LAL_COMMIT="a13e410022d5db89c64983c2bf9c1c1da54f7cdb"
BUILD_DIR="$REPO_ROOT/.lal-build"
CLONE_DIR="$BUILD_DIR/lalsuite"
INSTALL_PREFIX="$BUILD_DIR/install"
WHEEL_DIR="$REPO_ROOT/lal-wheels"
NCPU=$(sysctl -n hw.physicalcpu 2>/dev/null || echo 4)

# ── 1. System dependencies ────────────────────────────────────────────────────

echo "==> Checking build dependencies..."

missing=()
command -v autoconf  &>/dev/null || missing+=(autoconf)
command -v automake  &>/dev/null || missing+=(automake)
command -v libtool   &>/dev/null || missing+=(libtool)
command -v swig      &>/dev/null || missing+=(swig)
command -v gsl-config &>/dev/null || missing+=(gsl)
pkg-config --exists fftw3 2>/dev/null  || missing+=(fftw)

if [ ${#missing[@]} -gt 0 ]; then
    echo "==> Installing via Homebrew: ${missing[*]}"
    brew install "${missing[@]}"
fi

echo "    autoconf: $(autoconf --version | head -1)"
echo "    automake: $(automake --version | head -1)"
echo "    swig:     $(swig -version 2>&1 | head -1)"
echo "    gsl:      $(gsl-config --version)"
echo "    fftw3:    $(pkg-config --modversion fftw3)"

# ── 2. Clone ──────────────────────────────────────────────────────────────────

mkdir -p "$BUILD_DIR"
if [ -d "$CLONE_DIR/.git" ]; then
    echo "==> lalsuite already cloned at $CLONE_DIR"
else
    echo "==> Cloning lalsuite fork..."
    git clone "$LAL_FORK" "$CLONE_DIR"
fi
git -C "$CLONE_DIR" checkout "$LAL_COMMIT"
echo "==> At commit: $(git -C "$CLONE_DIR" rev-parse HEAD)"

# ── 3. Build helper ───────────────────────────────────────────────────────────

build_package () {
    local pkg="$1"          # e.g. "lal" or "lalsimulation"
    local extra_flags="${2:-}"
    local src="$CLONE_DIR/$pkg"
    local bld="$BUILD_DIR/$pkg-build"

    echo ""
    echo "══ Building $pkg ══════════════════════════════════════════════════"

    # Bootstrap (generate configure script from configure.ac)
    if [ ! -f "$src/configure" ]; then
        echo "==> Running autoreconf in $src..."
        (cd "$src" && LIBTOOLIZE=true autoreconf --install --force 2>&1)
    fi

    mkdir -p "$bld"
    echo "==> Configuring $pkg..."
    (cd "$bld" && "$src/configure" \
        --prefix="$INSTALL_PREFIX" \
        --enable-swig-python \
        --disable-doxygen \
        --disable-gcc-flags \
        PKG_CONFIG_PATH="$INSTALL_PREFIX/lib/pkgconfig:$(brew --prefix)/lib/pkgconfig" \
        LDFLAGS="-L$INSTALL_PREFIX/lib" \
        CPPFLAGS="-I$INSTALL_PREFIX/include" \
        $extra_flags \
        2>&1 | tail -5)

    echo "==> Building $pkg (using $NCPU cores)..."
    make -C "$bld" -j"$NCPU" 2>&1 | tail -5

    echo "==> Installing $pkg..."
    make -C "$bld" install 2>&1 | tail -3
}

# ── 4. Build lal then lalsimulation ──────────────────────────────────────────

build_package "lal"
build_package "lalsimulation"

# ── 5. Point the uv venv at the installed packages via a .pth file ───────────
# The compiled lal/lalsimulation Python packages (including the .so SWIG
# extensions) are already installed under $INSTALL_PREFIX. Rather than
# repackaging them into wheels, we add their parent directory to the uv
# venv's sys.path via a .pth file, and set DYLD_LIBRARY_PATH so the
# extensions can find the shared C libraries at runtime.

echo ""
echo "==> Syncing uv test-integration env..."
cd "$REPO_ROOT"
uv sync --group test-integration

PY_VER=$(uv run python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
INSTALLED_PY="$INSTALL_PREFIX/lib/python${PY_VER}/site-packages"

if [ ! -d "$INSTALLED_PY/lal" ]; then
    echo "ERROR: expected lal Python package not found at $INSTALLED_PY/lal"
    echo "  Check that the autotools build installed Python bindings."
    exit 1
fi

VENV_SITE=$(uv run python -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$VENV_SITE/lal-custom.pth"

echo "==> Writing .pth file: $PTH_FILE"
echo "$INSTALLED_PY" > "$PTH_FILE"

# The venv may contain a stale lalsimulation .so (installed by a previous uv sync
# or manual pip install) that shadows our freshly built one.  uv installs name the
# extension _lalsimulation.cpython-3XX-darwin.so while ours is _lalsimulation.so;
# Python picks the cpython-named file first.  Replace it with our build.
echo "==> Replacing stale lal/lalsimulation in venv with freshly built versions..."
for pkg in lal lalsimulation; do
    SRC="$INSTALLED_PY/$pkg"
    DST="$VENV_SITE/$pkg"
    if [ -d "$DST" ]; then
        # Copy the freshly built .so files over any cpython-named stale copies
        for so_src in "$SRC"/*.so; do
            so_name=$(basename "$so_src")
            # Also replace the cpython-tagged variant if it exists
            cpython_so=$(ls "$DST"/_*cpython*.so 2>/dev/null | head -1)
            if [ -n "$cpython_so" ]; then
                echo "    replacing $(basename "$cpython_so") in $pkg"
                cp "$so_src" "$cpython_so"
            fi
            cp "$so_src" "$DST/$so_name"
        done
        # Copy any updated .py files
        cp "$SRC"/*.py "$DST/"
    fi
done

# Also write a small activation snippet so DYLD_LIBRARY_PATH is set when
# running tests — stored as a conftest.py hook.
CONFTEST="$REPO_ROOT/tests/conftest.py"
cat > "$CONFTEST" <<PYEOF
# Auto-generated by scripts/build_lal_wheels.sh — do not edit manually.
# Sets DYLD_LIBRARY_PATH so lal's .so extensions find their C libraries.
import os
os.environ.setdefault(
    "DYLD_LIBRARY_PATH",
    "$INSTALL_PREFIX/lib" + ":" + os.environ.get("DYLD_LIBRARY_PATH", ""),
)
PYEOF

echo ""
echo "✓ lal installed at:   $INSTALLED_PY/lal"
echo "✓ lalsimulation at:   $INSTALLED_PY/lalsimulation"
echo "✓ .pth:               $PTH_FILE"
echo "✓ conftest.py:        $CONFTEST"
echo ""
echo "✓ Done. Run integration tests with:"
echo "  uv run --group test-integration pytest tests/test_likelihood.py -v"
