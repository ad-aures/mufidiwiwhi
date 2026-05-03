#!/usr/bin/env bash
# Install Mufidiwiwhi GUI from a PyInstaller-built binary.
#
# Usage:
#   ./install_ubuntu.sh                  # install from ./dist (default)
#   ./install_ubuntu.sh --url URL        # download a binary or tarball
#   ./install_ubuntu.sh --uninstall      # remove the launcher and binary
#
# The script never installs system packages and never asks for sudo;
# everything goes under $HOME (~/.local). Re-running is safe.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="mufidiwiwhi-gui"
INSTALL_BIN_DIR="$HOME/.local/bin"
INSTALL_LIB_DIR="$HOME/.local/share/mufidiwiwhi"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="$HOME/.local/share/applications"
ICON_SRC="$REPO_ROOT/mufidiwiwhi.svg"
DESKTOP_SRC="$REPO_ROOT/packaging/mufidiwiwhi.desktop"

URL=""
DO_UNINSTALL=0
for arg in "$@"; do
    case "$arg" in
        --url=*)      URL="${arg#--url=}" ;;
        --url)        shift; URL="${1:-}" ;;
        --uninstall)  DO_UNINSTALL=1 ;;
        -h|--help)
            sed -n '2,9p' "$0"
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            exit 2
            ;;
    esac
done

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!! \033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m!! \033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Uninstall
# ---------------------------------------------------------------------------
if [[ "$DO_UNINSTALL" -eq 1 ]]; then
    log "Removing launcher and binary"
    rm -f "$DESKTOP_DIR/$APP_NAME.desktop"
    rm -f "$ICON_DIR/mufidiwiwhi.svg"
    rm -f "$INSTALL_BIN_DIR/$APP_NAME"
    rm -rf "$INSTALL_LIB_DIR"
    command -v update-desktop-database >/dev/null 2>&1 \
        && update-desktop-database "$DESKTOP_DIR" || true
    command -v gtk-update-icon-cache >/dev/null 2>&1 \
        && gtk-update-icon-cache -t "$HOME/.local/share/icons/hicolor" || true
    log "Done."
    exit 0
fi

# ---------------------------------------------------------------------------
# Pick the source: --url, or local ./dist
# ---------------------------------------------------------------------------
SRC=""
TMPDIR=""
cleanup() { [[ -n "$TMPDIR" && -d "$TMPDIR" ]] && rm -rf "$TMPDIR"; }
trap cleanup EXIT

if [[ -n "$URL" ]]; then
    log "Downloading $URL"
    TMPDIR="$(mktemp -d)"
    DOWNLOAD="$TMPDIR/download"
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar "$URL" -o "$DOWNLOAD"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -O "$DOWNLOAD" "$URL"
    else
        die "Neither curl nor wget is available."
    fi
    case "$URL" in
        *.tar.gz|*.tgz)
            log "Extracting tarball"
            tar -xzf "$DOWNLOAD" -C "$TMPDIR"
            SRC="$(find "$TMPDIR" -maxdepth 4 -type f -name "$APP_NAME" | head -n 1)"
            [[ -z "$SRC" ]] && SRC="$(find "$TMPDIR" -maxdepth 4 -type d -name "$APP_NAME" | head -n 1)"
            ;;
        *.zip)
            log "Extracting zip"
            command -v unzip >/dev/null 2>&1 || die "unzip is required for .zip URLs."
            unzip -q "$DOWNLOAD" -d "$TMPDIR"
            SRC="$(find "$TMPDIR" -maxdepth 4 -type f -name "$APP_NAME" | head -n 1)"
            [[ -z "$SRC" ]] && SRC="$(find "$TMPDIR" -maxdepth 4 -type d -name "$APP_NAME" | head -n 1)"
            ;;
        *)
            SRC="$DOWNLOAD"
            ;;
    esac
else
    if [[ -f "$REPO_ROOT/dist/$APP_NAME" ]]; then
        SRC="$REPO_ROOT/dist/$APP_NAME"             # one-file build
    elif [[ -d "$REPO_ROOT/dist/$APP_NAME" ]]; then
        SRC="$REPO_ROOT/dist/$APP_NAME"             # one-folder build
    else
        die "No prebuilt binary found in $REPO_ROOT/dist. Build it first with:
        pyinstaller packaging/mufidiwiwhi-gui-onefile.spec --noconfirm
or pass --url URL to fetch a release artefact."
    fi
fi

[[ -e "$SRC" ]] || die "Source not found after download: $SRC"

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
mkdir -p "$INSTALL_BIN_DIR" "$INSTALL_LIB_DIR" "$ICON_DIR" "$DESKTOP_DIR"

EXEC_PATH=""
if [[ -f "$SRC" ]]; then
    log "Installing single-file binary to $INSTALL_BIN_DIR/$APP_NAME"
    install -m 0755 "$SRC" "$INSTALL_BIN_DIR/$APP_NAME"
    EXEC_PATH="$INSTALL_BIN_DIR/$APP_NAME"
else
    log "Installing single-folder bundle to $INSTALL_LIB_DIR"
    rm -rf "$INSTALL_LIB_DIR"
    cp -a "$SRC" "$INSTALL_LIB_DIR"
    [[ -f "$INSTALL_LIB_DIR/$APP_NAME" ]] || die "Bundle missing $APP_NAME at $INSTALL_LIB_DIR"
    chmod +x "$INSTALL_LIB_DIR/$APP_NAME"
    log "Symlinking $INSTALL_BIN_DIR/$APP_NAME -> $INSTALL_LIB_DIR/$APP_NAME"
    ln -sf "$INSTALL_LIB_DIR/$APP_NAME" "$INSTALL_BIN_DIR/$APP_NAME"
    EXEC_PATH="$INSTALL_LIB_DIR/$APP_NAME"
fi

if [[ -f "$ICON_SRC" ]]; then
    log "Installing icon to $ICON_DIR/mufidiwiwhi.svg"
    install -m 0644 "$ICON_SRC" "$ICON_DIR/mufidiwiwhi.svg"
else
    warn "Icon source not found at $ICON_SRC; launcher will use the default icon."
fi

if [[ -f "$DESKTOP_SRC" ]]; then
    log "Installing launcher to $DESKTOP_DIR/$APP_NAME.desktop"
    sed -e "s|^Exec=.*|Exec=$EXEC_PATH %F|" \
        -e "s|^Icon=.*|Icon=mufidiwiwhi|" \
        "$DESKTOP_SRC" \
        > "$DESKTOP_DIR/$APP_NAME.desktop"
    chmod 0644 "$DESKTOP_DIR/$APP_NAME.desktop"
else
    warn "Desktop template missing at $DESKTOP_SRC; you will not get a launcher entry."
fi

# Refresh caches if the helpers are present (these never fail the install).
command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$DESKTOP_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 \
    && gtk-update-icon-cache -t "$HOME/.local/share/icons/hicolor" || true

log "Done."
echo
echo "  Binary:   $EXEC_PATH"
echo "  Launcher: $DESKTOP_DIR/$APP_NAME.desktop"
echo "  Icon:     $ICON_DIR/mufidiwiwhi.svg"
echo
echo "If $INSTALL_BIN_DIR is on your PATH, just run: $APP_NAME"
echo "Otherwise launch from your application menu (Mufidiwiwhi)."
