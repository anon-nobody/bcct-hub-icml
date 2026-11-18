#!/usr/bin/env bash
# Convert all text in figure PDFs to vector outlines so the paper PDF contains no Type-3 fonts
# (ICML requires Type-1 fonts only; matplotlib embeds DejaVu as Type 3 by default).
# Usage: bash scripts/figures/outline_fonts.sh paper/figures/fig_validation.pdf [more.pdf ...]
set -euo pipefail
for f in "$@"; do
  tmp="$(mktemp --suffix=.pdf)"
  gs -q -dNOPAUSE -dBATCH -dSAFER -sDEVICE=pdfwrite -dNoOutputFonts -dCompatibilityLevel=1.5 \
     -sOutputFile="$tmp" "$f"
  mv "$tmp" "$f"
  echo "outlined: $f"
done
