#!/usr/bin/env bash
# layout_dataset.sh — reorganise the Kaggle-mounted dataset into the
# directory layout the paper's data.py expects.
#
# Kaggle dataset layout (after auto-unzip on Kaggle, or kaggle CLI on Ubuntu):
#   <SRC>/eng_{train,val,test}_{lex,nonlex}/{lex|nonlex}/<SIGNER>_<gender>_<age>_<lang>_<type>/<idx>/<frames>
#   <SRC>/eng_{train,val,test}_{lex,nonlex}/{lex|nonlex}/<SIGNER>_..../gt.txt
#
# Paper's data.py glob: `<DST>/{lex,nonlex}/*/<frames>` AND `<DST>/{lex,nonlex}/gt.txt`
# i.e. <DST>/train/lex/<signer>/<idx>/<frames>, <DST>/train/lex/<signer>/gt.txt.
#
# This script SYMLINKS the existing dataset into the paper's expected
# location -- no copy, no disk usage, runs in seconds.
#
# Usage:
#   SRC=/path/to/wita-full-english-122signers DST=$HOME/wita-data/english \
#       ./layout_dataset.sh

set -euo pipefail

: "${SRC:?need SRC=path/to/dataset/root with eng_*_* subdirs}"
: "${DST:?need DST=$HOME/wita-data/english}"

mkdir -p "$DST"

for split in train val test; do
    for subset in lex nonlex; do
        src_dir="${SRC}/eng_${split}_${subset}/${subset}"
        dst_dir="${DST}/${split}/${subset}"
        if [[ ! -d "$src_dir" ]]; then
            echo "skip: no source dir $src_dir"
            continue
        fi
        echo "linking ${split}/${subset}: $src_dir -> $dst_dir"
        mkdir -p "$(dirname "$dst_dir")"
        # Remove a previous symlink/empty dir before linking.
        if [[ -L "$dst_dir" || -d "$dst_dir" ]]; then
            rm -rf "$dst_dir"
        fi
        ln -s "$src_dir" "$dst_dir"
        # Quick sanity: count clip dirs.
        n_clips=$(find -L "$dst_dir" -mindepth 2 -maxdepth 2 -type d | wc -l)
        n_gt=$(find -L "$dst_dir" -name gt.txt | wc -l)
        echo "  clips=$n_clips  gt_files=$n_gt"
    done
done

echo
echo "Done.  Verify the paper's data.py expectation:"
echo "  glob: ${DST}/<split>/<lex|nonlex>/<signer>/<word_idx>"
ls "${DST}" 2>/dev/null || true
