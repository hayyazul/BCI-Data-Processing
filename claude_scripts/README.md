# claude_scripts

Reusable utility scripts created by Claude for tasks that recur across sessions.

- `list_ipynb_cells.py` — List cells in a Jupyter notebook with index, type, and source preview. Use `--full` for full source, `--preview N` to set preview length.
- `append_ipynb_cells.py` — Append an inclusive range of cells from one notebook to another. Resets code cell outputs/execution counts by default; pass `--keep-outputs` to preserve them.
