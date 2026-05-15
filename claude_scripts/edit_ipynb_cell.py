"""Replace the source of a single cell in a Jupyter notebook by index.

Usage:
    python claude_scripts/edit_ipynb_cell.py <notebook.ipynb> <cell_index> <source_file>

Reads the new cell source from <source_file> (raw text), assigns it to
notebook['cells'][cell_index]['source'] as a list of lines (each line keeping
its trailing newline except possibly the last), clears outputs/execution_count
on code cells, and writes the notebook back in place.
"""
import json
import sys


def main():
    nb_path, idx_str, src_path = sys.argv[1], sys.argv[2], sys.argv[3]
    idx = int(idx_str)
    with open(nb_path) as f:
        nb = json.load(f)
    with open(src_path) as f:
        text = f.read()
    lines = text.splitlines(keepends=True)
    cell = nb['cells'][idx]
    cell['source'] = lines
    if cell['cell_type'] == 'code':
        cell['outputs'] = []
        cell['execution_count'] = None
    with open(nb_path, 'w') as f:
        json.dump(nb, f, indent=1)
        f.write('\n')


if __name__ == '__main__':
    main()
