"""ImpacTB preprocessing helpers for SATURN (downsample, HVG union, cache)."""

from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy import sparse

_SUBJECT_PREFERRED = (
    "subject",
    "donor_id",
    "donor",
    "patient_id",
    "patient",
    "sample_id",
    "sample",
)
_SUBJECT_PATTERN = re.compile(r"(?i)(subject|donor|patient|sample)")


def detect_subject_column(adata: AnnData, explicit_col: str | None = None) -> str:
    """Auto-detect subject/donor column in adata.obs."""
    if explicit_col:
        if explicit_col not in adata.obs.columns:
            raise RuntimeError(
                f"TRAINING_SUBJECT_COL='{explicit_col}' not in obs columns: "
                f"{list(adata.obs.columns)}"
            )
        return explicit_col
    for col in _SUBJECT_PREFERRED:
        if col in adata.obs.columns:
            return col
    matches = [c for c in adata.obs.columns if _SUBJECT_PATTERN.search(str(c))]
    if matches:
        return matches[0]
    raise RuntimeError(
        "Could not detect a subject/donor column in adata.obs. "
        f"Available columns: {list(adata.obs.columns)}. "
        "Set TRAINING_SUBJECT_COL to override."
    )


def balanced_downsample_obs_names(
    adata: AnnData, subject_col: str, max_cells: int, rng: np.random.Generator
) -> list[str]:
    """Balanced per-subject downsample; returns list of obs_names (len <= max_cells)."""
    if adata.n_obs <= max_cells:
        return adata.obs_names.tolist()
    subjects = adata.obs[subject_col].astype(str).unique().tolist()
    n_subjects = len(subjects)
    base_quota = max_cells // n_subjects
    remainder = max_cells % n_subjects
    subj_order = subjects.copy()
    rng.shuffle(subj_order)
    extra_subjects = set(subj_order[:remainder])
    quotas = {s: base_quota + (1 if s in extra_subjects else 0) for s in subjects}
    selected: list[str] = []
    unused_quota = 0
    for subject in subjects:
        mask = adata.obs[subject_col].astype(str) == subject
        idx = np.where(mask)[0]
        quota = quotas[subject]
        n_take = min(len(idx), quota)
        if n_take > 0:
            chosen = rng.choice(idx, size=n_take, replace=False)
            selected.extend(adata.obs_names[chosen].tolist())
        unused_quota += quota - n_take
    if unused_quota > 0:
        already = set(selected)
        candidates = [name for name in adata.obs_names if name not in already]
        if candidates:
            n_extra = min(unused_quota, len(candidates))
            extra = rng.choice(candidates, size=n_extra, replace=False)
            selected.extend(list(extra))
    return selected[:max_cells]


def downsample_adatas_balanced(
    adatas: list[AnnData],
    species_order: list[str],
    max_cells_per_species: int,
    rng: np.random.Generator,
    explicit_subject_col: str | None = None,
) -> tuple[list[AnnData], pd.DataFrame, dict[str, str]]:
    """Per-species subject-balanced downsample; returns (adatas, summary_df, subject_cols)."""
    downsampled: list[AnnData] = []
    summary_rows: list[dict[str, Any]] = []
    subject_cols: dict[str, str] = {}
    for adata, species in zip(adatas, species_order):
        subject_col = detect_subject_column(adata, explicit_subject_col)
        subject_cols[species] = subject_col
        counts_before = adata.obs[subject_col].astype(str).value_counts()
        selected = balanced_downsample_obs_names(adata, subject_col, max_cells_per_species, rng)
        adata_ds = adata[selected].copy()
        downsampled.append(adata_ds)
        counts_after = adata_ds.obs[subject_col].astype(str).value_counts()
        for subject, n_before in counts_before.items():
            summary_rows.append(
                {
                    "species": species,
                    "subject": subject,
                    "subject_col_used": subject_col,
                    "n_cells_before": int(n_before),
                    "n_cells_after": int(counts_after.get(subject, 0)),
                }
            )
    return downsampled, pd.DataFrame(summary_rows), subject_cols


def select_hvg_union(
    adatas: list[AnnData],
    species_order: list[str],
    n_top_genes: int,
    flavor: str,
) -> tuple[list[AnnData], list[str], dict[str, set[str]]]:
    """Per-species HVG selection then union; returns (adatas_subset, hvg_union, per_species_hvgs)."""
    per_species_hvgs: dict[str, set[str]] = {}
    for adata, species in zip(adatas, species_order):
        sc.pp.highly_variable_genes(
            adata, n_top_genes=n_top_genes, flavor=flavor, subset=False
        )
        per_species_hvgs[species] = set(adata.var_names[adata.var.highly_variable])
    hvg_union = sorted(set().union(*per_species_hvgs.values()))
    adatas_subset = [a[:, hvg_union].copy() for a in adatas]
    return adatas_subset, hvg_union, per_species_hvgs


def _looks_like_counts(x: np.ndarray) -> bool:
    """Heuristic: non-negative and mostly integer-valued."""
    if x.size == 0:
        return True
    if np.any(x < 0):
        return False
    frac = np.mean(np.isclose(x, np.round(x), rtol=0, atol=1e-3))
    return frac > 0.9


def resolve_expression_matrix(adata: AnnData) -> tuple[AnnData, str]:
    """Pick count matrix for SATURN; prefer layers/raw when X looks normalized."""
    source = "X"
    if "counts" in adata.layers and not _looks_like_counts(np.asarray(adata.X[:1])):
        adata = adata.copy()
        adata.X = adata.layers["counts"]
        source = "layers['counts']"
    elif adata.raw is not None and not _looks_like_counts(np.asarray(adata.X[:1])):
        adata = adata.copy()
        adata.X = adata.raw.X
        source = "raw.X"
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray()
    x = np.asarray(adata.X)
    if not _looks_like_counts(x):
        warnings.warn(
            f"Expression matrix from {source} may not be raw counts "
            "(SATURN ZINB + seurat_v3 expect counts).",
            stacklevel=2,
        )
    adata.X = x.astype(np.float32)
    return adata, source


def densify_adata(adata: AnnData) -> AnnData:
    """Convert X to dense float32 (scMODAL notebook convention)."""
    if sparse.issparse(adata.X):
        adata.X = adata.X.toarray().astype(np.float32)
    else:
        adata.X = np.asarray(adata.X, dtype=np.float32)
    return adata


def load_manifest_adatas(
    harmonized_dir: Path,
) -> tuple[list[AnnData], list[str], pd.DataFrame]:
    """Load species AnnData objects from integration_manifest.csv."""
    manifest_path = harmonized_dir / "integration_manifest.csv"
    if not manifest_path.exists():
        raise RuntimeError(
            f"integration_manifest.csv not found in {harmonized_dir}. "
            "Ensure GENE_HARMONIZE completed successfully."
        )
    manifest = pd.read_csv(manifest_path)
    manifest = manifest.sort_values("order_index").reset_index(drop=True)
    if manifest.empty:
        raise RuntimeError("integration_manifest.csv is empty.")
    adatas: list[AnnData] = []
    for row in manifest.itertuples(index=False):
        h5ad_path = harmonized_dir / row.h5ad_file
        if not h5ad_path.exists():
            raise RuntimeError(f"Expected h5ad file not found: {h5ad_path}")
        adata = sc.read_h5ad(h5ad_path)
        adatas.append(densify_adata(adata))
    species_order = manifest["species"].tolist()
    return adatas, species_order, manifest


def _build_cache_manifest(
    max_cells_per_species: int,
    training_random_seed: int,
    training_subject_col: str | None,
    species_order: list[str],
    n_cells_before: dict[str, int],
    n_top_genes_per_species: int,
    hvg_flavor: str,
) -> dict[str, Any]:
    return {
        "MAX_CELLS_PER_SPECIES": max_cells_per_species,
        "TRAINING_RANDOM_SEED": training_random_seed,
        "TRAINING_SUBJECT_COL": training_subject_col,
        "species_order": species_order,
        "n_cells_before": n_cells_before,
        "N_TOP_GENES_PER_SPECIES": n_top_genes_per_species,
        "HVG_FLAVOR": hvg_flavor,
    }


def _cache_matches(
    cache_manifest_path: Path,
    cache_variant_dir: Path,
    species_order: list[str],
    manifest_dict: dict[str, Any],
) -> bool:
    if not cache_manifest_path.exists():
        return False
    try:
        cached = json.loads(cache_manifest_path.read_text())
    except Exception:
        return False
    manifest_keys = [
        "MAX_CELLS_PER_SPECIES",
        "TRAINING_RANDOM_SEED",
        "TRAINING_SUBJECT_COL",
        "species_order",
        "N_TOP_GENES_PER_SPECIES",
        "HVG_FLAVOR",
    ]
    for key in manifest_keys:
        if cached.get(key) != manifest_dict.get(key):
            return False
    for species in species_order:
        if not (cache_variant_dir / f"{species}.h5ad").exists():
            return False
    if not (cache_variant_dir / "hvg_genes_union.csv").exists():
        return False
    return True


def build_or_load_cache(
    harmonized_dir: Path,
    cache_dir: Path,
    *,
    max_cells_per_species: int,
    training_random_seed: int,
    training_subject_col: str | None,
    n_top_genes_per_species: int,
    hvg_flavor: str,
) -> dict[str, Any]:
    """Downsample + HVG union with disk cache; returns result bundle."""
    adatas, species_order, manifest = load_manifest_adatas(harmonized_dir)
    pre_downsample_obs = [a.n_obs for a in adatas]
    cache_variant_dir = cache_dir / "downsampled_hvg"
    cache_variant_dir.mkdir(parents=True, exist_ok=True)
    cache_manifest_path = cache_variant_dir / "cache_manifest.json"
    manifest_dict = _build_cache_manifest(
        max_cells_per_species,
        training_random_seed,
        training_subject_col,
        species_order,
        dict(zip(species_order, pre_downsample_obs)),
        n_top_genes_per_species,
        hvg_flavor,
    )

    if _cache_matches(cache_manifest_path, cache_variant_dir, species_order, manifest_dict):
        adatas = []
        for species in species_order:
            adata_cached = sc.read_h5ad(cache_variant_dir / f"{species}.h5ad")
            adatas.append(densify_adata(adata_cached))
        training_downsample_summary = pd.read_csv(
            cache_variant_dir / "training_downsample_summary.csv"
        )
        training_subject_cols = json.loads(
            (cache_variant_dir / "training_subject_cols.json").read_text()
        )
        hvg_genes_df = pd.read_csv(cache_variant_dir / "hvg_genes_union.csv")
        hvg_union = hvg_genes_df["gene"].tolist()
        hvg_per_species_hvgs = {
            s: set(hvg_genes_df.loc[hvg_genes_df[f"in_{s}"], "gene"].tolist())
            for s in species_order
            if f"in_{s}" in hvg_genes_df.columns
        }
    else:
        rng = np.random.default_rng(training_random_seed)
        adatas, training_downsample_summary, training_subject_cols = downsample_adatas_balanced(
            adatas,
            species_order,
            max_cells_per_species,
            rng,
            explicit_subject_col=training_subject_col,
        )
        training_downsample_summary.to_csv(
            cache_variant_dir / "training_downsample_summary.csv", index=False
        )
        (cache_variant_dir / "training_subject_cols.json").write_text(
            json.dumps(training_subject_cols, indent=2)
        )
        adatas, hvg_union, hvg_per_species_hvgs = select_hvg_union(
            adatas, species_order, n_top_genes_per_species, hvg_flavor
        )
        hvg_genes_rows = []
        for gene in hvg_union:
            row: dict[str, Any] = {"gene": gene}
            for species in species_order:
                row[f"in_{species}"] = gene in hvg_per_species_hvgs[species]
            hvg_genes_rows.append(row)
        hvg_genes_df = pd.DataFrame(hvg_genes_rows)
        hvg_genes_df.to_csv(cache_variant_dir / "hvg_genes_union.csv", index=False)
        for species, adata in zip(species_order, adatas):
            adata.write_h5ad(cache_variant_dir / f"{species}.h5ad")
        cache_write = {
            **manifest_dict,
            "n_cells_after": {s: a.n_obs for s, a in zip(species_order, adatas)},
            "n_genes": int(adatas[0].n_vars) if adatas else 0,
            "subject_cols_used": training_subject_cols,
            "n_genes_union": len(hvg_union),
            "hvg_per_species_counts": {
                s: len(hvg_per_species_hvgs[s]) for s in species_order
            },
        }
        cache_manifest_path.write_text(json.dumps(cache_write, indent=2))

    return {
        "adatas": adatas,
        "species_order": species_order,
        "manifest": manifest,
        "cache_variant_dir": cache_variant_dir,
        "training_downsample_summary": training_downsample_summary,
        "training_subject_cols": training_subject_cols,
        "hvg_union": hvg_union,
        "hvg_per_species_hvgs": hvg_per_species_hvgs,
        "n_genes_union": len(hvg_union),
        "n_cells_per_species_before_downsample": dict(zip(species_order, pre_downsample_obs)),
        "max_cells_per_species": max_cells_per_species,
    }
