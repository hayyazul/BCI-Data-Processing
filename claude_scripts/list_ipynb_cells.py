#!/usr/bin/env python3
"""List cells in a Jupyter notebook with index, type, and a source preview.

Usage: python list_ipynb_cells.py <notebook.ipynb> [--full] [--preview N]
  --full       Print full cell source (no truncation).
  --preview N  Show first N chars of each cell (default 500).
"""
import json, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('notebook')
    ap.add_argument('--full', action='store_true')
    ap.add_argument('--preview', type=int, default=500)
    args = ap.parse_args()

    with open(args.notebook) as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        src = ''.join(cell['source'])
        print(f'=== Cell {i} ({cell["cell_type"]}) ===')
        print(src if args.full else src[:args.preview])
        print()

if __name__ == '__main__':
    main()
