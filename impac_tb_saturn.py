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

  **Protein embeddings** (one-time): `curl -L http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz | tar -xz -C data/`

    **Dry run** (skip training): `SATURN_DRY_RUN=1 marimo run impac_tb_saturn.py`
    """)
    return


@app.cell
def _():
    import os
    import pathlib

    _DEFAULT_SCRATCH = (
        "/home/exacloud/gscratch/prime-seq/Bimber/GW/scModal_ImpacTB/saturn_impac_tb"
    )
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
        default = pathlib.Path(_DEFAULT_SCRATCH)
        if default.is_dir() or str(default).startswith("/home/exacloud/gscratch"):
            return default.resolve()
        for key in ("SLURM_TMPDIR", "TMPDIR", "TMP"):
            val = os.environ.get(key, "").strip()
            if not val:
                continue
            candidate = pathlib.Path(val).expanduser().resolve()
            if candidate.exists() and not _is_under_home(candidate):
                return (candidate / "saturn_work").resolve()
        # ponytail: local dev fallback when no HPC scratch is available
        local = pathlib.Path(__file__).resolve().parent / "work"
        local.mkdir(parents=True, exist_ok=True)
        return local.resolve()

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
    for env_key, path in _cache_targets.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[env_key] = str(path)
    os.chdir(WORKING_DIR)
    print(f"SATURN_IMPACTB: WORKING_DIR = {WORKING_DIR}", flush=True)
    return TMP_ROOT, WORKING_DIR, os, pathlib


@app.cell
def _(os, pathlib):
    SATURN_ROOT = pathlib.Path(__file__).resolve().parent
    VENDOR_SATURN = SATURN_ROOT / "vendor" / "SATURN"
    EMBEDDINGS_DIR = SATURN_ROOT / "data" / "protein_embeddings"

    HARMONIZED_DIR = pathlib.Path(
        os.environ.get(
            "HARMONIZED_DIR",
            "/home/exacloud/gscratch/prime-seq/Bimber/GW/scModal_ImpacTB/outputs/harmonized/harmonized_outputs/",
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
    SATURN_EMBEDDING_MODEL = os.environ.get("SATURN_EMBEDDING_MODEL", "ESM1b")
    SATURN_DEVICE_NUM = int(os.environ.get("SATURN_DEVICE_NUM", "0"))
    SATURN_DRY_RUN = os.environ.get("SATURN_DRY_RUN", "").strip() in {
        "1",
        "true",
        "True",
    }
    CT_MAP_PATH = os.environ.get("CT_MAP_PATH", "").strip() or None

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
        SATURN_PRETRAIN_BATCH_SIZE,
        SATURN_PRETRAIN_EPOCHS,
        SATURN_ROOT,
        SATURN_SEED,
        TRAINING_RANDOM_SEED,
        TRAINING_SUBJECT_COL,
        VENDOR_SATURN,
    )


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


@app.cell
def _(CACHE_SUBDIR, WORKING_DIR):
    cache_dir = WORKING_DIR / CACHE_SUBDIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_dir = WORKING_DIR / "model_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    saturn_inputs_dir = cache_dir / "saturn_inputs"
    saturn_inputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"SATURN_IMPACTB: cache_dir = {cache_dir}", flush=True)
    print(f"SATURN_IMPACTB: out_dir   = {out_dir}", flush=True)
    return cache_dir, out_dir, saturn_inputs_dir


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
        else min(a.n_obs for a in _raw_adatas)
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
        col, source = ResolveLabelColumn(
            _adata,
            preferred_resolution=LEIDEN_RESOLUTION,
            explicit_col=IN_LABEL_COL,
            n_neighbors=N_NEIGHBORS,
            random_state=TRAINING_RANDOM_SEED,
        )
        label_cols[_species] = col
        label_sources[_species] = source
        print(
            f"SATURN_IMPACTB: labels species={_species} col={col} source={source}",
            flush=True,
        )
    label_summary = pd.DataFrame(
        [
            {"species": s, "label_col": label_cols[s], "label_source": label_sources[s]}
            for s in species_order
        ]
    )
    return label_cols, label_sources, label_summary


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
    in_data_label_cols: dict[str, str] = {}
    for _species, _adata in zip(species_order, adatas):
        adata_out, src = resolve_expression_matrix(_adata.copy())
        src_col = label_cols[_species]
        adata_out.obs[SATURN_LABEL_COL] = adata_out.obs[src_col].astype(str)
        expr_sources[_species] = src
        out_path = saturn_inputs_dir / f"{_species}_saturn.h5ad"
        adata_out.write_h5ad(out_path)
        h5ad_paths[_species] = out_path
    label_cols = {s: SATURN_LABEL_COL for s in species_order}
    in_data_label_cols = label_cols.copy()
    mo.md(label_summary.to_markdown(index=False))
    return expr_sources, h5ad_paths, in_data_label_cols


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
    missing = [str(p) for p in embedding_paths.values() if not p.exists()]
    if missing:
        cmd = ProteinEmbeddingsDownloadCommand(EMBEDDINGS_DIR)
        mo.stop(
            mo.md(
                f"**Missing protein embeddings** ({len(missing)} files).\n\n"
                f"```bash\n{cmd}\n```\n\n"
                "Or set paths via embedding_path in in_data.csv."
            )
        )
    if SATURN_DRY_RUN:
        mo.md("**SATURN_DRY_RUN=1** — skipping training; inputs validated.")
    return embedding_paths, missing


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
    in_data_path,
    in_data_label_cols,
    n_genes_union,
    out_dir,
    species_order,
    subprocess,
    sys,
):
    run_name = None
    train_cmd = []
    if not SATURN_DRY_RUN:
        train_script = VENDOR_SATURN / "train-saturn.py"
        if not train_script.exists():
            raise RuntimeError(f"SATURN vendor not found: {train_script}")
        ref_label_col = in_data_label_cols[species_order[0]]
        train_cmd = [
            sys.executable,
            str(train_script),
            "--in_data",
            str(in_data_path),
            "--work_dir",
            str(out_dir),
            "--log_dir",
            str(out_dir / "tboard_log"),
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
            ref_label_col,
            "--centroids_init_path",
            str(out_dir / "centroids_init.pkl"),
        ]
        print("SATURN_IMPACTB: launching train-saturn.py", flush=True)
        result = subprocess.run(train_cmd, check=False, cwd=str(VENDOR_SATURN))
        if result.returncode != 0:
            raise RuntimeError(f"train-saturn.py failed with code {result.returncode}")
        _h5ads = sorted(out_dir.glob("*.h5ad"))
        run_name = _h5ads[0] if _h5ads else None
    return run_name, train_cmd


@app.cell
def _(SATURN_DRY_RUN, out_dir, plt, run_name, sc):
    integrated = None
    if not SATURN_DRY_RUN and run_name is not None and run_name.exists():
        integrated = sc.read_h5ad(run_name)
        if "X_umap" not in integrated.obsm:
            sc.pp.neighbors(integrated, use_rep="X", n_neighbors=15)
            sc.tl.umap(integrated)
        fig, ax = plt.subplots(figsize=(6, 5))
        sc.pl.umap(integrated, color="species", show=False, ax=ax)
        fig.savefig(out_dir / "umap_species.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        if "labels2" in integrated.obs.columns:
            fig, ax = plt.subplots(figsize=(6, 5))
            sc.pl.umap(integrated, color="labels2", show=False, ax=ax)
            fig.savefig(out_dir / "umap_labels.png", dpi=120, bbox_inches="tight")
            plt.close(fig)
    return (integrated,)


@app.cell
def _(
    HARMONIZED_DIR,
    SATURN_DRY_RUN,
    cache_result,
    expr_sources,
    integrated,
    json,
    label_cols,
    label_sources,
    manifest,
    n_genes_union,
    out_dir,
    pd,
    shutil,
    species_order,
    train_cmd,
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
    summary = {
        "species_order": species_order,
        "n_genes_union": n_genes_union,
        "label_cols": label_cols,
        "in_data_label_col": "saturn_label",
        "label_sources": label_sources,
        "expr_sources": expr_sources,
        "saturn_dry_run": SATURN_DRY_RUN,
        "train_cmd": train_cmd,
        "integrated_path": str(integrated.filename) if integrated is not None else None,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(out_dir / "run_summary.csv", index=False)
    print(f"SATURN_IMPACTB: artifacts in {out_dir}", flush=True)
    return


if __name__ == "__main__":
    app.run()
