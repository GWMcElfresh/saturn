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
export HARMONIZED_DIR=/path/to/harmonized_outputs/   # optional; default: ./outputs/harmonized/harmonized_outputs
export WORKING_DIR=/path/to/scratch/work             # must not be $HOME on HPC
export SATURN_OUTPUT_DIR=/path/to/saturn_outputs     # optional; default: ./saturn_outputs
```

Defaults assume harmonized outputs live under `outputs/harmonized/` in this repo and preprocessing cache under `./cache/`. Override for other layouts or HPC scratch paths.

**HPC data setup** (one-time on cluster): symlink harmonized upstream outputs into this repo:

```bash
mkdir -p /home/exacloud/gscratch/prime-seq/Bimber/GW/saturn/outputs/harmonized
ln -sfn /home/exacloud/gscratch/prime-seq/Bimber/GW/scModal_ImpacTB/outputs/harmonized/harmonized_outputs \
  /home/exacloud/gscratch/prime-seq/Bimber/GW/saturn/outputs/harmonized/harmonized_outputs
```

### 5. Run

| Use case | Command |
|----------|---------|
| Local interactive dev | `marimo edit impac_tb_saturn.py` |
| HPC interactive (SSH tunnel) | `bash submit.sh` |
| HPC batch (headless) | `bash submit_run.sh` |
| Dry run (no training) | `SATURN_DRY_RUN=1 python impac_tb_saturn.py` |

**Batch = `python notebook.py`. Interactive = `marimo edit`. Dashboard = `marimo run`.**

`submit_run.sh` runs `python impac_tb_saturn.py` inside the SLURM job (marimo script mode). Do not use `marimo run` for scheduled jobs — it starts a read-only web app and blocks until something opens the URL.

**Module smoke test** (label resolution + species map):

```bash
python smoke_check.py
```

#### HPC / SLURM

- [`submit_run.sh`](submit_run.sh) — headless batch (`python impac_tb_saturn.py`)
- [`submit.sh`](submit.sh) — interactive only (`marimo edit --headless` + SSH port forward)

Run `bash submit_run.sh` or `bash submit.sh` from the repo (not raw `sbatch`) so each job creates `${PROJECT_DIR}/logs/` and writes annotated files:

`logs/<step>-<YYYYMMDD-HHMMSS>-<jobid>.{out,err}`

Examples: `logs/batch-20250702-143022-12345.out`, `logs/interactive-20250702-150011-12346.out`.

Launch scripts source [`scripts/pipeline_env.sh`](scripts/pipeline_env.sh) via `${PROJECT_DIR}` (not `BASH_SOURCE`) so they work after SLURM copies the job script to `/var/spool/slurmd/`.

**Default SLURM allocations** (override with `sbatch` flags if needed; cluster cap 1 TB RAM):

| Script | CPUs | Memory | Walltime |
|--------|------|--------|----------|
| `submit_run.sh` (batch) | 8 | 512 GB | 24 h |
| `submit.sh` (interactive) | 4 | 128 GB | 8 h |

A successful batch log shows `SATURN_BATCH:` startup lines and `SATURN_IMPACTB:` progress, then artifacts under `$SATURN_OUTPUT_DIR` (default `${PROJECT_DIR}/saturn_outputs`, including `run_summary.json`). If the log shows `URL: http://localhost:...`, the job used `marimo run` by mistake and likely produced no outputs.

## Environment variables

### Data / preprocessing (shared with scMODAL ImpacTB)

| Variable | Default | Description |
|----------|---------|-------------|
| `HARMONIZED_DIR` | `./outputs/harmonized/harmonized_outputs` | Directory with `integration_manifest.csv` |
| `WORKING_DIR` | `${PROJECT_DIR}/work` (batch) | Temp files and scratch cwd |
| `SATURN_OUTPUT_DIR` | `${PROJECT_DIR}/saturn_outputs` (batch) | SATURN training outputs |
| `CACHE_SUBDIR` | `cache` | Preprocessed AnnData cache (under saturn root, not `WORKING_DIR`) |
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

Under `./cache/` (saturn project root):

- `cache/downsampled_hvg/` — cached downsampled + HVG-subset AnnData
- `cache/saturn_inputs/` — per-species h5ad for SATURN

Under `$SATURN_OUTPUT_DIR` (default `./saturn_outputs` at repo root):

- `saturn_results/` — SATURN run artifacts
- `umap_species.png`, `cell_clusters.tsv`, `macrogene_weights.tsv`, `run_summary.json`

## Upstream references

- [SATURN paper / code](https://github.com/snap-stanford/SATURN)
- [Protein embeddings bundle](http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz)
- ImpacTB harmonized inputs from the scMODAL / GENE_HARMONIZE pipeline

## License

ImpacTB wrapper code in this repository is provided for research use. Upstream SATURN is [MIT licensed](https://github.com/snap-stanford/SATURN/blob/main/LICENSE). Respect Stanford’s data redistribution terms for downloaded embeddings.
