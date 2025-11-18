# Tree Generator and File Distribution

This project generates random rooted trees (arborescences), creates folder structures on disk to represent the trees, and distributes positive/negative sample files across folders according to configurable hyperparameters.

What I changed
- Reduced comment verbosity and converted main docstrings/messages to concise English.
- Standardized folder names to English (`Folder_` and `folder_`).
- Added README, .gitignore and MIT license.

Quick start
1. Create a virtualenv and install requirements from `requirements.txt` (if present).

2. Run the main script to generate trees and distribute files:

```bash
python improved_main.py
```

Notes
- The main entrypoint is `improved_main.py`.
- Visualization functions use Graphviz; install `graphviz` system package if you want PNG outputs.
- If you want all comments and internal variable names converted to English, I can do a further pass to rename variables and update all references.
