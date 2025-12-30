# afs_studio

Native C++17 visualization and training management application for AFS.

## Build

```bash
# From project root
cmake -B build -S . -DAFS_BUILD_STUDIO=ON
cmake --build build --target afs_studio
```

## Run

```bash
./build/apps/studio/afs_studio
```

## Data sources

- Training data path: `~/src/training` if present, otherwise `~/.context/training` (override with CLI arg).
- Context graph: `AFS_GRAPH_PATH` or `${AFS_CONTEXT_ROOT}/index/afs_graph.json` (defaults to `~/src/context` or `~/.context`).
- Dataset registry: `AFS_DATASET_REGISTRY` or `${AFS_TRAINING_ROOT}/index/dataset_registry.json`.

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
