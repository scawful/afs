# afs_studio

Native C++17 visualization and training management application for AFS.

## Build

```bash
# From AFS repo root
cmake -S apps/studio -B build/studio
cmake --build build/studio --target afs_studio
```

## Run

```bash
./build/studio/afs_studio --data ~/src/training
```

## Install (local)

```bash
cmake --install build/studio --prefix ~/.local
# Ensure ~/.local/bin is on PATH
```

## CLI helpers

```bash
python -m afs studio build
python -m afs studio run --build
python -m afs studio install --prefix ~/.local
python -m afs studio alias
```

## Quick aliases

```bash
export AFS_ROOT=~/src/trunk/lab/afs
alias afs-studio='PYTHONPATH="$AFS_ROOT/src" python -m afs studio run --build'
alias afs-studio-build='PYTHONPATH="$AFS_ROOT/src" python -m afs studio build'
```

## Data sources

- Training data path: `AFS_TRAINING_ROOT` if set, otherwise `~/src/training` or `~/.context/training` (override with CLI arg).
- Context graph: `AFS_GRAPH_PATH` or `${AFS_CONTEXT_ROOT}/index/afs_graph.json` (defaults to `~/src/context` or `~/.context`).
- Dataset registry: `AFS_DATASET_REGISTRY` or `${AFS_TRAINING_ROOT}/index/dataset_registry.json`.
- Resource index: `AFS_RESOURCE_INDEX` or `${AFS_TRAINING_ROOT}/index/resource_index.json`.

## Flags

- `--data` or `--data-path`: override training root
- `--version`: print version and exit

## Features

- **Dashboard**: Training metrics overview
- **Analysis**: Quality score trends, domain breakdown
- **Training Hub**: Real-time training status
- **Sample Review**: Data quality inspection
- **Text Editor**: Built-in code editor
- **Shortcut System**: Customizable keyboard shortcuts (Ctrl+/)

## Dependencies (auto-fetched)

- Dear ImGui (docking branch)
- ImPlot
- GLFW
- nlohmann/json
