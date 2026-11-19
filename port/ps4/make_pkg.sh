#!/usr/bin/env bash
# The installable PS4 .pkg (port/ps4/PLAN.md, phase 5): the host (dhd_ps4.elf, build_host64.py
# --target ps4) as eboot.bin, and nothing of the game: the player's own disc image goes to
# /data/dhd/ on the console (ps4_host.c).
# Usage: port/ps4/make_pkg.sh <dhd_ps4.elf> <output dir>
set -euo pipefail

ELF="$1"
OUTDIR="$(mkdir -p "$2" && cd "$2" && pwd)"
PS4_DIR="$(cd "$(dirname "$0")" && pwd)"
TOOLS="$OO_PS4_TOOLCHAIN/bin/linux"

TITLE="Downhill PS4"
VERSION="01.00"
TITLE_ID="DHDR00001"
CONTENT_ID="IV0000-DHDR00001_00-DOWNHILLREVIVED0"

export DOTNET_SYSTEM_GLOBALIZATION_INVARIANT=1
export LD_LIBRARY_PATH="${PS4_HOSTLIBS:-$HOME/ps4dev/hostlibs/usr/lib}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

STAGE="$OUTDIR/pkg-stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/sce_sys/about" "$STAGE/sce_module"

"$TOOLS/create-fself" -in="$ELF" -out="$OUTDIR/dhd.oelf" --eboot "$STAGE/eboot.bin" --paid 0x3800000000000011
cp "$PS4_DIR/sce_sys/icon0.png" "$STAGE/sce_sys/"
cp "$PS4_DIR/sce_sys/about/right.sprx" "$STAGE/sce_sys/about/"
cp "$PS4_DIR/sce_module/libc.prx" "$PS4_DIR/sce_module/libSceFios2.prx" "$STAGE/sce_module/"

SFO="$STAGE/sce_sys/param.sfo"
"$TOOLS/PkgTool.Core" sfo_new "$SFO"
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" APP_TYPE  --type Integer --maxsize 4   --value 1
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" APP_VER   --type Utf8    --maxsize 8   --value "$VERSION"
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" ATTRIBUTE --type Integer --maxsize 4   --value 0
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" CATEGORY  --type Utf8    --maxsize 4   --value gd
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" CONTENT_ID --type Utf8   --maxsize 48  --value "$CONTENT_ID"
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" DOWNLOAD_DATA_SIZE --type Integer --maxsize 4 --value 0
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" SYSTEM_VER --type Integer --maxsize 4  --value 0
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" TITLE     --type Utf8    --maxsize 128 --value "$TITLE"
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" TITLE_ID  --type Utf8    --maxsize 12  --value "$TITLE_ID"
"$TOOLS/PkgTool.Core" sfo_setentry "$SFO" VERSION   --type Utf8    --maxsize 8   --value "$VERSION"

cd "$STAGE"
"$TOOLS/create-gp4" -out pkg.gp4 --content-id="$CONTENT_ID" \
    --files "eboot.bin sce_sys/about/right.sprx sce_sys/param.sfo sce_sys/icon0.png sce_module/libc.prx sce_module/libSceFios2.prx"
"$TOOLS/PkgTool.Core" pkg_build pkg.gp4 "$OUTDIR"
cp -f "$OUTDIR/$CONTENT_ID.pkg" "$OUTDIR/DownhillPS4-$VERSION.pkg"
echo "OK: $OUTDIR/DownhillPS4-$VERSION.pkg"
