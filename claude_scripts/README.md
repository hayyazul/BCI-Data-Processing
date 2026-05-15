# claude_scripts

Reusable utility scripts created by Claude for tasks that recur across sessions.

- `list_ipynb_cells.py` — List cells in a Jupyter notebook with index, type, and source preview. Use `--full` for full source, `--preview N` to set preview length.
- `append_ipynb_cells.py` — Append an inclusive range of cells from one notebook to another. Resets code cell outputs/execution counts by default; pass `--keep-outputs` to preserve them.
- `edit_ipynb_cell.py` — Replace a single cell's source in a notebook by index. Usage: `python claude_scripts/edit_ipynb_cell.py <nb.ipynb> <idx> <source_file>`. Clears code-cell outputs/execution counts.
- `insert_ipynb_cell.py` — Insert a new cell at a given index. Usage: `python claude_scripts/insert_ipynb_cell.py <nb.ipynb> <insert_at> <code|markdown> <source_file>`.
