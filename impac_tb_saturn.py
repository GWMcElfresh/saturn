import marimo

__generated_with = "0.23.10"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SATURN ImpacTB — Subject-Balanced Downsample + HVG Union

    Cross-species integration with [SATURN](https://github.com/snap-stanford/saturn) on **human → macaque → mouse**,
    using the same preprocessing as `scMODAL_ImpacTB_HVGDownsample` (subject-balanced downsample to min species count,
    HVG union `seurat_v3`). Labels: cell type → existing clusters (resolution ≈ 0.5) → on-the-fly Leiden.

    **Protein embeddings** (one-time): download the Stanford bundle into `data/protein_embeddings_export/` (ESM2 subfolder used by default).

    **Dry run** (skip training): `SATURN_DRY_RUN=1 python impac_tb_saturn.py`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Working directory

    Resolve a writable scratch location for caches, model outputs, and temp files.
    Priority: `WORKING_DIR` env var → Nextflow task dir → repo `work/` →
    cluster `TMPDIR` → local `work/` beside this notebook. Also redirects matplotlib,
    Numba, and XDG caches away from `$HOME` (required on many HPC systems).
    """)
    return


@app.cell
def _():
    import os
    import pathlib

    _SATURN_ROOT = pathlib.Path(__file__).resolve().parent
    _HOME = pathlib.Path.home()

    def _is_under_home(path: pathlib.Path) -> bool:
        try:
            path.resolve().relative_to(_HOME.resolve())
            return True
        except ValueError:
            return False

    def _resolve_working_dir() -> pathlib.Path:
        explicit = os.environ.get("WORKING_DIR", "").strip()
        if explicit:
            return pathlib.Path(explicit).expanduser().resolve()
        for key in ("NXF_TASK_WORKDIR", "NXF_WORK"):
            val = os.environ.get(key, "").strip()
            if val:
                candidate = pathlib.Path(val).expanduser().resolve()
                if not _is_under_home(candidate):
                    return candidate
        default = _SATURN_ROOT / "work"
        default.mkdir(parents=True, exist_ok=True)
        return default.resolve()

    WORKING_DIR = _resolve_working_dir()
    TMP_ROOT = WORKING_DIR / "tmp"
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    _cache_targets = {
        "TMPDIR": TMP_ROOT,
        "TEMP": TMP_ROOT,
        "TMP": TMP_ROOT,
        "NUMBA_CACHE_DIR": TMP_ROOT / "numba_cache",
        "MPLCONFIGDIR": TMP_ROOT / "matplotlib",
        "XDG_CACHE_HOME": TMP_ROOT / "xdg_cache",
    }
    os.environ["MPLBACKEND"] = "Agg"
    for _env_key, _path in _cache_targets.items():
        _path.mkdir(parents=True, exist_ok=True)
        os.environ[_env_key] = str(_path)
    os.chdir(WORKING_DIR)
    print(f"SATURN_IMPACTB: WORKING_DIR = {WORKING_DIR}", flush=True)
    return TMP_ROOT, WORKING_DIR, os, pathlib


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    Paths and hyperparameters from environment variables (see `README.md` for the full list).
    Key inputs: harmonized AnnData under `outputs/harmonized/` (override with `HARMONIZED_DIR`),
    HVG/downsample settings, label-resolution options, and SATURN training knobs (`SATURN_*`).
    Preprocessing cache lives in `./cache/` beside this notebook. `SATURN_DRY_RUN=1` validates
    inputs without launching training.
    """)
    return


@app.cell
def _(os, pathlib):
    SATURN_ROOT = pathlib.Path(__file__).resolve().parent
    VENDOR_SATURN = SATURN_ROOT / "vendor" / "SATURN"
    SATURN_EMBEDDING_MODEL = os.environ.get("SATURN_EMBEDDING_MODEL", "ESM2")
    _embeddings_default = (
        SATURN_ROOT / "data" / "protein_embeddings_export" / SATURN_EMBEDDING_MODEL
    )
    EMBEDDINGS_DIR = pathlib.Path(
        os.environ.get("EMBEDDINGS_DIR", str(_embeddings_default))
    )

    HARMONIZED_DIR = pathlib.Path(
        os.environ.get(
            "HARMONIZED_DIR",
            str(SATURN_ROOT / "outputs" / "harmonized" / "harmonized_outputs"),
        )
    )

    TRAINING_RANDOM_SEED = int(os.environ.get("TRAINING_RANDOM_SEED", "42"))
    MAX_CELLS_PER_SPECIES = int(os.environ.get("MAX_CELLS_PER_SPECIES", "0"))
    N_TOP_GENES_PER_SPECIES = int(os.environ.get("N_TOP_GENES_PER_SPECIES", "3000"))
    HVG_FLAVOR = os.environ.get("HVG_FLAVOR", "seurat_v3")
    TRAINING_SUBJECT_COL = (
        os.environ.get("TRAINING_SUBJECT_COL", "").strip() or None
    )
    CACHE_SUBDIR = pathlib.Path(os.environ.get("CACHE_SUBDIR", "cache"))

    LEIDEN_RESOLUTION = float(os.environ.get("LEIDEN_RESOLUTION", "0.5"))
    N_NEIGHBORS = int(os.environ.get("N_NEIGHBORS", "30"))
    IN_LABEL_COL = os.environ.get("IN_LABEL_COL", "").strip() or None

    SATURN_SEED = int(os.environ.get("SATURN_SEED", "0"))
    SATURN_NUM_MACROGENES = int(os.environ.get("SATURN_NUM_MACROGENES", "2000"))
    SATURN_PRETRAIN_EPOCHS = int(os.environ.get("SATURN_PRETRAIN_EPOCHS", "50"))
    SATURN_EPOCHS = int(os.environ.get("SATURN_EPOCHS", "25"))
    SATURN_BATCH_SIZE = int(os.environ.get("SATURN_BATCH_SIZE", "1024"))
    SATURN_PRETRAIN_BATCH_SIZE = int(
        os.environ.get("SATURN_PRETRAIN_BATCH_SIZE", str(SATURN_BATCH_SIZE))
    )
    SATURN_DEVICE_NUM = int(os.environ.get("SATURN_DEVICE_NUM", "0"))
    SATURN_DRY_RUN = os.environ.get("SATURN_DRY_RUN", "").strip() in {
        "1",
        "true",
        "True",
    }
    CT_MAP_PATH = os.environ.get("CT_MAP_PATH", "").strip() or None
    SATURN_OUTPUT_DIR = os.environ.get(
        "SATURN_OUTPUT_DIR", str(SATURN_ROOT / "saturn_outputs")
    ).strip()

    return (
        CACHE_SUBDIR,
        CT_MAP_PATH,
        EMBEDDINGS_DIR,
        HARMONIZED_DIR,
        HVG_FLAVOR,
        IN_LABEL_COL,
        LEIDEN_RESOLUTION,
        MAX_CELLS_PER_SPECIES,
        N_NEIGHBORS,
        N_TOP_GENES_PER_SPECIES,
        SATURN_BATCH_SIZE,
        SATURN_DEVICE_NUM,
        SATURN_DRY_RUN,
        SATURN_EMBEDDING_MODEL,
        SATURN_EPOCHS,
        SATURN_NUM_MACROGENES,
        SATURN_OUTPUT_DIR,
        SATURN_PRETRAIN_BATCH_SIZE,
        SATURN_PRETRAIN_EPOCHS,
        SATURN_ROOT,
        SATURN_SEED,
        TRAINING_RANDOM_SEED,
        TRAINING_SUBJECT_COL,
        VENDOR_SATURN,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Imports

    Load scanpy, PyTorch, and project helpers (`impactb_preprocess`, `label_resolve`,
    `species_map`). Vendor SATURN is added to `sys.path` when present.
    """)
    return


@app.cell
def _(SATURN_ROOT, VENDOR_SATURN):
    import json
    import shutil
    import subprocess
    import sys

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import torch

    matplotlib.use("Agg")
    sys.path.insert(0, str(SATURN_ROOT))
    if VENDOR_SATURN.is_dir():
        sys.path.insert(0, str(VENDOR_SATURN))

    from impactb_preprocess import build_or_load_cache, resolve_expression_matrix
    from label_resolve import ResolveLabelColumn
    from saturn_exports import (
        export_cell_clusters_tsv,
        export_macrogene_weights_tsv,
        find_genes_to_macrogenes_pkl,
        find_integrated_h5ad,
    )
    from species_map import (
        BuildInDataCsv,
        ProteinEmbeddingsDownloadCommand,
        RequiredEmbeddingPaths,
    )

    print(
        f"SATURN_IMPACTB: torch={torch.__version__} cuda={torch.cuda.is_available()}",
        flush=True,
    )
    return (
        BuildInDataCsv,
        ProteinEmbeddingsDownloadCommand,
        RequiredEmbeddingPaths,
        ResolveLabelColumn,
        build_or_load_cache,
        export_cell_clusters_tsv,
        export_macrogene_weights_tsv,
        find_genes_to_macrogenes_pkl,
        find_integrated_h5ad,
        json,
        np,
        pd,
        plt,
        resolve_expression_matrix,
        sc,
        shutil,
        subprocess,
        sys,
        torch,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Output directories

    Create `cache/` (preprocessed AnnData, SATURN input h5ads) beside this notebook and
    SATURN outputs under `SATURN_OUTPUT_DIR` (default `./saturn_outputs` under `WORKING_DIR`).
    Override with env `SATURN_OUTPUT_DIR` (absolute or relative to `WORKING_DIR`).
    """)
    return


@app.cell
def _(CACHE_SUBDIR, SATURN_OUTPUT_DIR, SATURN_ROOT, WORKING_DIR, pathlib):
    cache_dir = SATURN_ROOT / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    _output = pathlib.Path(SATURN_OUTPUT_DIR)
    out_dir = (
        _output.resolve()
        if _output.is_absolute()
        else (WORKING_DIR / _output).resolve()
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    saturn_inputs_dir = cache_dir / "saturn_inputs"
    saturn_inputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"SATURN_IMPACTB: cache_dir = {cache_dir}", flush=True)
    print(f"SATURN_IMPACTB: out_dir   = {out_dir}", flush=True)
    return cache_dir, out_dir, saturn_inputs_dir


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocess harmonized data

    Load per-species AnnData from `HARMONIZED_DIR` via `integration_manifest.csv`.
    Subject-balanced downsample to the smallest species count (or `MAX_CELLS_PER_SPECIES`),
    then per-species HVG selection and union (`seurat_v3` by default). Results are cached
    on disk so reruns skip recomputation when parameters match.
    """)
    return


@app.cell
def _(
    HARMONIZED_DIR,
    HVG_FLAVOR,
    MAX_CELLS_PER_SPECIES,
    N_TOP_GENES_PER_SPECIES,
    TRAINING_RANDOM_SEED,
    TRAINING_SUBJECT_COL,
    build_or_load_cache,
    cache_dir,
):
    from impactb_preprocess import load_manifest_adatas

    _raw_adatas, _species_order, _manifest = load_manifest_adatas(HARMONIZED_DIR)
    _max_cells = (
        MAX_CELLS_PER_SPECIES
        if MAX_CELLS_PER_SPECIES > 0
        else min(_a.n_obs for _a in _raw_adatas)
    )
    cache_result = build_or_load_cache(
        HARMONIZED_DIR,
        cache_dir,
        max_cells_per_species=_max_cells,
        training_random_seed=TRAINING_RANDOM_SEED,
        training_subject_col=TRAINING_SUBJECT_COL,
        n_top_genes_per_species=N_TOP_GENES_PER_SPECIES,
        hvg_flavor=HVG_FLAVOR,
    )
    adatas = cache_result["adatas"]
    species_order = cache_result["species_order"]
    manifest = cache_result["manifest"]
    n_genes_union = cache_result["n_genes_union"]
    print(
        f"SATURN_IMPACTB: downsample target={cache_result['max_cells_per_species']} cells/species",
        flush=True,
    )
    for _species, _adata in zip(species_order, adatas):
        print(
            f"  {_species}: {_adata.n_obs:,} cells × {_adata.n_vars:,} genes",
            flush=True,
        )
    return adatas, cache_result, manifest, n_genes_union, species_order


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Resolve training labels

    For each species, pick a label column for SATURN supervision: explicit `IN_LABEL_COL`
    if set, else cell-type annotations, else existing Leiden clusters near resolution 0.5,
    else on-the-fly Leiden at `LEIDEN_RESOLUTION`. Prints the chosen column and source per species.
    """)
    return


@app.cell
def _(
    IN_LABEL_COL,
    LEIDEN_RESOLUTION,
    N_NEIGHBORS,
    ResolveLabelColumn,
    TRAINING_RANDOM_SEED,
    adatas,
    pd,
    species_order,
):
    label_cols: dict[str, str] = {}
    label_sources: dict[str, str] = {}
    for _species, _adata in zip(species_order, adatas):
        _col, _source = ResolveLabelColumn(
            _adata,
            preferred_resolution=LEIDEN_RESOLUTION,
            explicit_col=IN_LABEL_COL,
            n_neighbors=N_NEIGHBORS,
            random_state=TRAINING_RANDOM_SEED,
        )
        label_cols[_species] = _col
        label_sources[_species] = _source
        print(
            f"SATURN_IMPACTB: labels species={_species} col={_col} source={_source}",
            flush=True,
        )
    label_summary = pd.DataFrame(
        [
            {
                "species": _s,
                "label_col": label_cols[_s],
                "label_source": label_sources[_s],
            }
            for _s in species_order
        ]
    )
    return label_cols, label_sources, label_summary


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Export SATURN input h5ads

    Copy each species AnnData to `cache/saturn_inputs/`, normalize the expression layer
    SATURN expects, and write labels into a unified `saturn_label` column. Displays a
    summary table of label columns and sources.
    """)
    return


@app.cell
def _(
    adatas,
    label_cols,
    label_summary,
    mo,
    resolve_expression_matrix,
    saturn_inputs_dir,
    species_order,
):
    SATURN_LABEL_COL = "saturn_label"
    h5ad_paths: dict[str, object] = {}
    expr_sources: dict[str, str] = {}
    for _species, _adata in zip(species_order, adatas):
        _adata_out, _src = resolve_expression_matrix(_adata.copy())
        _src_col = label_cols[_species]
        _adata_out.obs[SATURN_LABEL_COL] = _adata_out.obs[_src_col].astype(str)
        expr_sources[_species] = _src
        _out_path = saturn_inputs_dir / f"{_species}_saturn.h5ad"
        _adata_out.write_h5ad(_out_path)
        h5ad_paths[_species] = _out_path
    in_data_label_cols = {_s: SATURN_LABEL_COL for _s in species_order}
    mo.md(label_summary.to_markdown(index=False))
    return expr_sources, h5ad_paths, in_data_label_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Protein embeddings

    Verify per-species ESM embedding files under `EMBEDDINGS_DIR` (default
    `data/protein_embeddings_export/ESM2/`). Expects Stanford bundle names like
    `human_embedding.torch`. Stops with a download command if any are missing. In dry-run mode, reports success
    without proceeding to training.
    """)
    return


@app.cell
def _(
    EMBEDDINGS_DIR,
    ProteinEmbeddingsDownloadCommand,
    RequiredEmbeddingPaths,
    SATURN_DRY_RUN,
    SATURN_EMBEDDING_MODEL,
    mo,
    species_order,
):
    embedding_paths = RequiredEmbeddingPaths(
        species_order, EMBEDDINGS_DIR, SATURN_EMBEDDING_MODEL
    )
    print(
        f"SATURN_IMPACTB: embedding_model={SATURN_EMBEDDING_MODEL} "
        f"embeddings_dir={EMBEDDINGS_DIR}",
        flush=True,
    )
    for _species, _path in embedding_paths.items():
        print(f"SATURN_IMPACTB: embedding species={_species} path={_path}", flush=True)
    missing = [str(_p) for _p in embedding_paths.values() if not _p.exists()]
    if missing:
        print(
            f"SATURN_IMPACTB: ERROR missing protein embeddings ({len(missing)} files):",
            flush=True,
        )
        for _missing_path in missing:
            print(f"  {_missing_path}", flush=True)
        _cmd = ProteinEmbeddingsDownloadCommand(EMBEDDINGS_DIR)
        mo.stop(
            mo.md(
                f"**Missing protein embeddings** ({len(missing)} files).\n\n"
                f"```bash\n{_cmd}\n```\n\n"
                "Or set `EMBEDDINGS_DIR` / `SATURN_EMBEDDING_MODEL`, "
                "or paths via embedding_path in in_data.csv."
            )
        )
    if SATURN_DRY_RUN:
        _ = mo.md("**SATURN_DRY_RUN=1** — skipping training; inputs validated.")
    return embedding_paths, missing


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Build `in_data.csv`

    Write SATURN's species manifest (`data/in_data.csv`) linking each h5ad path,
    embedding file, and label column.
    """)
    return


@app.cell
def _(
    BuildInDataCsv,
    SATURN_ROOT,
    embedding_paths,
    h5ad_paths,
    in_data_label_cols,
    species_order,
):
    in_data_path = SATURN_ROOT / "data" / "in_data.csv"
    BuildInDataCsv(
        species_order, h5ad_paths, embedding_paths, in_data_label_cols, in_data_path
    )
    print(f"SATURN_IMPACTB: in_data.csv -> {in_data_path}", flush=True)
    return (in_data_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Train SATURN

    Launch vendor `train-saturn.py` with macrogene count, pretrain/fine-tune epochs,
    batch sizes, and reference-species label column. Skipped when `SATURN_DRY_RUN=1`.
    Outputs land in `SATURN_OUTPUT_DIR` (default `./saturn_outputs`).
    """)
    return


@app.cell
def _(
    SATURN_BATCH_SIZE,
    SATURN_DEVICE_NUM,
    SATURN_DRY_RUN,
    SATURN_EPOCHS,
    SATURN_EMBEDDING_MODEL,
    SATURN_NUM_MACROGENES,
    SATURN_PRETRAIN_BATCH_SIZE,
    SATURN_PRETRAIN_EPOCHS,
    SATURN_SEED,
    VENDOR_SATURN,
    find_integrated_h5ad,
    in_data_path,
    in_data_label_cols,
    n_genes_union,
    out_dir,
    species_order,
    subprocess,
    sys,
):
    integrated_path = None
    train_cmd = []
    if not SATURN_DRY_RUN:
        _train_script = VENDOR_SATURN / "train-saturn.py"
        if not _train_script.exists():
            raise RuntimeError(f"SATURN vendor not found: {_train_script}")
        _ref_label_col = in_data_label_cols[species_order[0]]
        _work_dir = f"{out_dir.resolve()}/"
        (out_dir / "tboard_log").mkdir(parents=True, exist_ok=True)
        train_cmd = [
            sys.executable,
            str(_train_script),
            "--in_data",
            str(in_data_path),
            "--work_dir",
            _work_dir,
            "--log_dir",
            "tboard_log/",
            "--embedding_model",
            SATURN_EMBEDDING_MODEL,
            "--hv_genes",
            str(n_genes_union),
            "--num_macrogenes",
            str(SATURN_NUM_MACROGENES),
            "--pretrain_epochs",
            str(SATURN_PRETRAIN_EPOCHS),
            "--epochs",
            str(SATURN_EPOCHS),
            "--batch_size",
            str(SATURN_BATCH_SIZE),
            "--pretrain_batch_size",
            str(SATURN_PRETRAIN_BATCH_SIZE),
            "--seed",
            str(SATURN_SEED),
            "--device_num",
            str(SATURN_DEVICE_NUM),
            "--ref_label_col",
            _ref_label_col,
            "--centroids_init_path",
            str(out_dir / "centroids_init.pkl"),
        ]
        print("SATURN_IMPACTB: launching train-saturn.py", flush=True)
        _result = subprocess.run(train_cmd, check=False, cwd=str(VENDOR_SATURN))
        if _result.returncode != 0:
            raise RuntimeError(f"train-saturn.py failed with code {_result.returncode}")
        integrated_path = find_integrated_h5ad(out_dir)
        if integrated_path is None:
            raise RuntimeError(
                f"No integrated h5ad found under {out_dir / 'saturn_results'} "
                "after train-saturn.py completed"
            )
        print(f"SATURN_IMPACTB: integrated h5ad = {integrated_path}", flush=True)
    return integrated_path, train_cmd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Post-training exports

    After training, load the integrated h5ad from `saturn_results/` under the output dir,
    run Leiden on the SATURN embedding, save `cell_clusters.tsv` and `macrogene_weights.tsv`,
    and write `umap_species.png`. Skipped in dry-run mode.
    """)
    return


@app.cell
def _(
    LEIDEN_RESOLUTION,
    SATURN_DRY_RUN,
    TRAINING_RANDOM_SEED,
    export_cell_clusters_tsv,
    export_macrogene_weights_tsv,
    find_genes_to_macrogenes_pkl,
    integrated_path,
    out_dir,
    plt,
    sc,
):
    integrated = None
    umap_species_png = None
    cell_clusters_tsv = None
    macrogene_weights_tsv = None
    if not SATURN_DRY_RUN and integrated_path is not None and integrated_path.exists():
        print("SATURN_IMPACTB: post-train exports starting", flush=True)
        integrated = sc.read_h5ad(integrated_path)
        sc.pp.neighbors(integrated, use_rep="X", n_neighbors=15)
        sc.tl.leiden(
            integrated,
            resolution=LEIDEN_RESOLUTION,
            random_state=TRAINING_RANDOM_SEED,
            key_added="leiden",
        )
        if "X_umap" not in integrated.obsm:
            sc.tl.umap(integrated)
        umap_species_png = out_dir / "umap_species.png"
        _fig, _ax = plt.subplots(figsize=(6, 5))
        sc.pl.umap(integrated, color="species", show=False, ax=_ax)
        _fig.savefig(umap_species_png, dpi=120, bbox_inches="tight")
        plt.close(_fig)
        print(f"SATURN_IMPACTB: wrote {umap_species_png}", flush=True)
        cell_clusters_tsv = out_dir / "cell_clusters.tsv"
        export_cell_clusters_tsv(integrated, cell_clusters_tsv)
        print(f"SATURN_IMPACTB: wrote {cell_clusters_tsv}", flush=True)
        _pkl = find_genes_to_macrogenes_pkl(out_dir)
        if _pkl is None:
            raise RuntimeError(
                f"No genes_to_macrogenes PKL found for {integrated_path.name} "
                f"under {out_dir / 'saturn_results'}"
            )
        macrogene_weights_tsv = out_dir / "macrogene_weights.tsv"
        export_macrogene_weights_tsv(_pkl, macrogene_weights_tsv)
        print(f"SATURN_IMPACTB: wrote {macrogene_weights_tsv}", flush=True)
    return (
        cell_clusters_tsv,
        integrated,
        macrogene_weights_tsv,
        umap_species_png,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Write run artifacts

    Copy manifest and shared-gene files, export downsample summary CSV, and write
    `run_summary.json` / `run_summary.csv` with paths, label choices, and the training
    command for reproducibility.
    """)
    return


@app.cell
def _(
    HARMONIZED_DIR,
    SATURN_DRY_RUN,
    cache_result,
    cell_clusters_tsv,
    expr_sources,
    integrated,
    json,
    label_cols,
    label_sources,
    macrogene_weights_tsv,
    manifest,
    n_genes_union,
    out_dir,
    pd,
    shutil,
    species_order,
    train_cmd,
    umap_species_png,
):
    shutil.copy2(
        HARMONIZED_DIR / "integration_manifest.csv",
        out_dir / "integration_manifest.csv",
    )
    if (HARMONIZED_DIR / "shared_genes.csv").exists():
        shutil.copy2(
            HARMONIZED_DIR / "shared_genes.csv", out_dir / "shared_genes.csv"
        )
    cache_result["training_downsample_summary"].to_csv(
        out_dir / "training_downsample_summary.csv", index=False
    )
    _summary = {
        "species_order": species_order,
        "n_genes_union": n_genes_union,
        "label_cols": label_cols,
        "in_data_label_col": "saturn_label",
        "label_sources": label_sources,
        "expr_sources": expr_sources,
        "saturn_dry_run": SATURN_DRY_RUN,
        "saturn_output_dir": str(out_dir),
        "train_cmd": train_cmd,
        "integrated_path": str(integrated.filename) if integrated is not None else None,
        "umap_species_png": str(umap_species_png) if umap_species_png else None,
        "cell_clusters_tsv": str(cell_clusters_tsv) if cell_clusters_tsv else None,
        "macrogene_weights_tsv": (
            str(macrogene_weights_tsv) if macrogene_weights_tsv else None
        ),
    }
    (out_dir / "run_summary.json").write_text(json.dumps(_summary, indent=2))
    pd.DataFrame([_summary]).to_csv(out_dir / "run_summary.csv", index=False)
    print(f"SATURN_IMPACTB: artifacts in {out_dir}", flush=True)
    return


if __name__ == "__main__":
    app.run()
