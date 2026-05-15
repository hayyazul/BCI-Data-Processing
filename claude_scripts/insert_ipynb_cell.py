"""Insert a new cell at a given index in a Jupyter notebook.

Usage:
    python claude_scripts/insert_ipynb_cell.py <notebook.ipynb> <insert_at_index> <cell_type> <source_file>

cell_type: 'code' or 'markdown'. The cell is inserted at insert_at_index
(existing cells at and after that index shift down by one).
"""
import json
import sys


def main():
    nb_path, idx_str, cell_type, src_path = sys.argv[1:5]
    idx = int(idx_str)
    with open(nb_path) as f:
        nb = json.load(f)
    with open(src_path) as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    cell = {'cell_type': cell_type, 'metadata': {}, 'source': lines}
    if cell_type == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    nb['cells'].insert(idx, cell)
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')


if __name__ == '__main__':
    main()
