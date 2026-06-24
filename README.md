# SATURN ImpacTB

Cross-species single-cell integration with [SATURN](https://github.com/snap-stanford/SATURN) on the ImpacTB harmonized pipeline (**human → macaque → mouse**). Preprocessing mirrors `scMODAL_ImpacTB_HVGDownsample`: subject-balanced downsampling to the smallest species count, then a per-species HVG union (`seurat_v3`, default 3,000 genes/species).

Workflow is driven by a [marimo](https://marimo.io) notebook (`impac_tb_saturn.py`).

## Requirements

- Python ≥ 3.10 (marimo 0.23+; use 3.10–3.11 if you need the deprecated `louvain` package for vendor vignettes)
- CUDA GPU recommended for training (CPU possible for smoke tests)
- Harmonized ImpacTB AnnData outputs (`integration_manifest.csv` + per-species `.h5ad`)
- SATURN protein embeddings (one-time download, see below)

## Quick start

### 1. Clone this repo

```bash
git clone https://github.com/GWMcElfresh/saturn.git
cd saturn
```

Upstream SATURN is **vendored** under `vendor/SATURN/` and pinned in [`VENDOR_SHA`](VENDOR_SHA) (commit `6906abf` from [snap-stanford/SATURN](https://github.com/snap-stanford/SATURN)). To refresh upstream later:

```bash
cd vendor/SATURN
git init && git remote add origin https://github.com/snap-stanford/SATURN.git
git fetch origin && git checkout $(cat ../../VENDOR_SHA)
```

### 2. Create environment

**With [uv](https://docs.astral.sh/uv/) (recommended on HPC):**

```bash
uv sync
source .venv/bin/activate   # or: uv run ...
```

**With pip:**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

For GPU FAISS (optional, if your cluster provides it):

```bash
pip install -e ".[gpu]"
```

PyTorch CUDA 12.4 wheels are configured in `pyproject.toml` via the `pytorch-cu124` index. On CPU-only machines, install torch from [pytorch.org](https://pytorch.org) before the rest of the stack.

### 3. Download protein embeddings

SATURN needs per-species ESM protein embedding files (`.pt`). Stanford hosts a precomputed bundle:

```bash
mkdir -p data
curl -L http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz | tar -xz -C data/
# Files land in data/protein_embeddings/
```

ImpacTB uses:

| Species in manifest | Embedding file |
|---------------------|----------------|
| `human` | `Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt` |
| `macaque` | `Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM1b.pt` |
| `mouse` | `Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt` |

Override macaque subspecies if needed:

```bash
export MACAQUE_EMBEDDING_SPECIES=macaca_fascicularis
```

Gene symbols in your `.h5ad` must appear in each species’ embedding dict. See [SATURN protein embeddings docs](https://github.com/snap-stanford/SATURN/tree/main/protein_embeddings) to generate custom files.

### 4. Point at harmonized data

```bash
export HARMONIZED_DIR=/path/to/harmonized_outputs/
export WORKING_DIR=/path/to/scratch/saturn_impac_tb   # must not be $HOME on HPC
```

Default `HARMONIZED_DIR` targets OHSU gscratch; override for your system.

### 5. Run

**Interactive notebook:**

```bash
marimo edit impac_tb_saturn.py
```

**Headless / batch:**

```bash
marimo run impac_tb_saturn.py
```

**Validate inputs without training:**

```bash
SATURN_DRY_RUN=1 marimo run impac_tb_saturn.py
```

**Module smoke test** (label resolution + species map):

```bash
python smoke_check.py
```

## Environment variables

### Data / preprocessing (shared with scMODAL ImpacTB)

| Variable | Default | Description |
|----------|---------|-------------|
| `HARMONIZED_DIR` | gscratch harmonized path | Directory with `integration_manifest.csv` |
| `WORKING_DIR` | gscratch scratch path | Cache, outputs, temp files |
| `MAX_CELLS_PER_SPECIES` | `0` (auto = min species) | Downsample cap per species |
| `N_TOP_GENES_PER_SPECIES` | `3000` | HVG count before union |
| `HVG_FLAVOR` | `seurat_v3` | scanpy HVG method |
| `TRAINING_RANDOM_SEED` | `42` | Downsample RNG |
| `TRAINING_SUBJECT_COL` | auto-detect | Subject/donor column override |

### Labels

| Variable | Default | Description |
|----------|---------|-------------|
| `IN_LABEL_COL` | auto | Force a specific `obs` column for SATURN metric learning |
| `LEIDEN_RESOLUTION` | `0.5` | Resolution when computing on-the-fly Leiden labels |
| `N_NEIGHBORS` | `30` | Neighbors for on-the-fly clustering |

Auto-detection order: cell-type columns → existing cluster columns (resolution nearest 0.5) → computed Leiden (`saturn_leiden_proxy`). All species get a unified `saturn_label` column for SATURN.

### SATURN training

| Variable | Default | Description |
|----------|---------|-------------|
| `SATURN_SEED` | `0` | Training seed |
| `SATURN_PRETRAIN_EPOCHS` | `50` | Pretrain epochs |
| `SATURN_EPOCHS` | `25` | Metric-learning epochs |
| `SATURN_BATCH_SIZE` | `1024` | Training batch size |
| `SATURN_PRETRAIN_BATCH_SIZE` | `1024` | Pretrain batch size |
| `SATURN_NUM_MACROGENES` | `2000` | Macrogene count |
| `SATURN_EMBEDDING_MODEL` | `ESM1b` | Protein embedding model |
| `SATURN_DEVICE_NUM` | `0` | CUDA device index |
| `SATURN_DRY_RUN` | off | `1` = skip `train-saturn.py` |
| `MACAQUE_EMBEDDING_SPECIES` | `macaca_mulatta` | Macaque embedding key |

## Repository layout

```
saturn/
  impac_tb_saturn.py      # marimo workflow
  impactb_preprocess.py   # downsample + HVG union + cache
  label_resolve.py        # label auto-detect + on-the-fly Leiden
  species_map.py          # manifest → protein embedding paths
  smoke_check.py          # quick self-check
  pyproject.toml
  vendor/SATURN/          # vendored snap-stanford/SATURN (see VENDOR_SHA)
  data/
    protein_embeddings/   # downloaded .pt files (not in git)
    in_data.csv           # generated at runtime
```

## Outputs

Under `$WORKING_DIR`:

- `cache/downsampled_hvg/` — cached downsampled + HVG-subset AnnData
- `cache/saturn_inputs/` — per-species h5ad for SATURN
- `model_outputs/` — SATURN run artifacts, UMAP plots, `run_summary.json`

## Upstream references

- [SATURN paper / code](https://github.com/snap-stanford/SATURN)
- [Protein embeddings bundle](http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz)
- ImpacTB harmonized inputs from the scMODAL / GENE_HARMONIZE pipeline

## License

ImpacTB wrapper code in this repository is provided for research use. Upstream SATURN is [MIT licensed](https://github.com/snap-stanford/SATURN/blob/main/LICENSE). Respect Stanford’s data redistribution terms for downloaded embeddings.
