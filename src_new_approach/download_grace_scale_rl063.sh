#!/bin/bash
# ============================================================
# Script: download_grace_scale_rl063.sh
# Purpose: Download JPL GRACE/GRACE-FO RL06.3v04 Scale Factor
# Author: Saurav Bhattarai (talchabhadel-lab setup)
# ============================================================

# === Configuration ===
DATA_DIR="$HOME/Documents/ORISE/GRACE_RL063"
FILE_NAME="JPL_MSCNv04_CRImascon_ScaleFactors.nc"
DATA_URL="https://archive.podaac.earthdata.nasa.gov/podaac-ops-cumulus-protected/TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4/${FILE_NAME}"

# === Create folder if it doesn’t exist ===
mkdir -p "$DATA_DIR"
cd "$DATA_DIR" || exit

echo "============================================================"
echo "   NASA JPL GRACE RL06.3v04 SCALE FACTOR DOWNLOADER"
echo "============================================================"
echo "Destination folder: $DATA_DIR"
echo ""
read -p "👉 Paste your Earthdata TOKEN here: " TOKEN

# === Sanity check ===
if [[ -z "$TOKEN" ]]; then
    echo "❌ Error: No token entered. Exiting."
    exit 1
fi

# === Download using wget ===
echo ""
echo "⬇️  Downloading Scale Factor file from PO.DAAC..."
wget --header="Authorization: Bearer ${TOKEN}" "$DATA_URL" -O "$FILE_NAME"

# === Check if file downloaded successfully ===
if [[ -f "$FILE_NAME" ]]; then
    echo ""
    echo "✅ Download complete!"
    echo "Saved file: $DATA_DIR/$FILE_NAME"
    echo ""
    echo "You can inspect it with:"
    echo "  ncdump -h $FILE_NAME | head"
else
    echo ""
    echo "⚠️  Download failed. Please check your token or network."
fi

echo "============================================================"
