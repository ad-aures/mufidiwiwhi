#!/usr/bin/env bash
# Install the Mufidiwiwhi GUI on Ubuntu / Debian-style systems.
#
# Pulls the latest published release from Codeberg, copies the
# binary to ~/.local/bin, drops an icon and a .desktop entry under
# ~/.local/share. No sudo, no system packages.
#
# One-liner:
#   curl -fsSL https://codeberg.org/adaures/mufidiwiwhi/raw/branch/main/install_ubuntu.sh | bash

set -euo pipefail

REPO="adaures/mufidiwiwhi"
API="https://codeberg.org/api/v1/repos/$REPO/releases/latest"
ICON_URL="https://codeberg.org/$REPO/raw/branch/main/mufidiwiwhi.svg"
APP="mufidiwiwhi-gui"
BIN_DIR="$HOME/.local/bin"
ICON_DIR="$HOME/.local/share/icons/hicolor/scalable/apps"
DESKTOP_DIR="$HOME/.local/share/applications"

log()  { printf '\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m!!\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m!!\033[0m %s\n' "$*" >&2; exit 1; }

command -v curl >/dev/null 2>&1 || die "curl is required."
command -v python3 >/dev/null 2>&1 || die "python3 is required."

log "Resolving latest release"
JSON="$(curl -fsSL "$API")" || die "Could not reach $API"

read -r TAG ASSET_URL ASSET_NAME < <(python3 -c '
import json, sys
data = json.loads(sys.argv[1])
tag = data.get("tag_name", "")
for a in data.get("assets") or []:
    name = a.get("name", "")
    if name.endswith("-linux-x86_64"):
        print(tag, a.get("browser_download_url", ""), name)
        break
' "$JSON")

[[ -n "${ASSET_URL:-}" ]] || die "No *-linux-x86_64 asset on the latest release."

mkdir -p "$BIN_DIR" "$ICON_DIR" "$DESKTOP_DIR"

log "Downloading $ASSET_NAME"
curl -fL --progress-bar "$ASSET_URL" -o "$BIN_DIR/$APP"
chmod +x "$BIN_DIR/$APP"

log "Downloading icon"
curl -fsSL "$ICON_URL" -o "$ICON_DIR/mufidiwiwhi.svg" \
    || warn "Icon download failed; launcher will use the default icon."

log "Writing launcher"
cat > "$DESKTOP_DIR/$APP.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Mufidiwiwhi
Comment=Multi-speaker podcast transcription
Exec=$BIN_DIR/$APP %F
Icon=mufidiwiwhi
Terminal=false
Categories=AudioVideo;Audio;Utility;
MimeType=audio/wav;audio/flac;audio/mpeg;audio/ogg;
StartupWMClass=$APP
EOF
chmod 0644 "$DESKTOP_DIR/$APP.desktop"

command -v update-desktop-database >/dev/null 2>&1 \
    && update-desktop-database "$DESKTOP_DIR" || true
command -v gtk-update-icon-cache >/dev/null 2>&1 \
    && gtk-update-icon-cache -t "$HOME/.local/share/icons/hicolor" || true

log "Installed $TAG to $BIN_DIR/$APP"
echo
echo "  Run from the apps menu (Mufidiwiwhi) or: $APP"
echo
echo "  To uninstall, remove:"
echo "    $BIN_DIR/$APP"
echo "    $ICON_DIR/mufidiwiwhi.svg"
echo "    $DESKTOP_DIR/$APP.desktop"
