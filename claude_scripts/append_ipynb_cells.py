#!/usr/bin/env python3
"""Append a range of cells from one Jupyter notebook to another.

Usage:
  python append_ipynb_cells.py <src.ipynb> <dst.ipynb> <start> <end> [--keep-outputs]

  start, end : inclusive cell indices (0-based) from the source notebook.
  --keep-outputs : preserve execution_count and outputs on copied code cells
                   (default: reset them so the destination notebook is clean).

Example:
  python append_ipynb_cells.py analysis.ipynb preprocess.ipynb 9 18
"""
import json, sys, argparse, copy

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('dst')
    ap.add_argument('start', type=int)
    ap.add_argument('end', type=int, help='inclusive')
    ap.add_argument('--keep-outputs', action='store_true')
    args = ap.parse_args()

    with open(args.src) as f:
        src_nb = json.load(f)
    with open(args.dst) as f:
        dst_nb = json.load(f)

    n_src = len(src_nb['cells'])
    if not (0 <= args.start <= args.end < n_src):
        sys.exit(f"Bad range [{args.start}, {args.end}]; src has {n_src} cells.")

    to_append = []
    for i in range(args.start, args.end + 1):
        cell = copy.deepcopy(src_nb['cells'][i])
        if cell['cell_type'] == 'code' and not args.keep_outputs:
            cell['execution_count'] = None
            cell['outputs'] = []
        to_append.append(cell)

    dst_nb['cells'].extend(to_append)

    with open(args.dst, 'w') as f:
        json.dump(dst_nb, f, indent=1)
        f.write('\n')

    print(f"Appended {len(to_append)} cells ({args.start}..{args.end}) "
          f"from {args.src} to {args.dst}. New cell count: {len(dst_nb['cells'])}.")

if __name__ == '__main__':
    main()
