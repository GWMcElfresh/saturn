"""Post-training export helpers for SATURN ImpacTB workflow."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd

_INTEGRATED_H5AD_EXCLUDE = ("_pretrain", "_ep_")


def _is_integrated_h5ad(path: Path) -> bool:
    name = path.name
    return not any(part in name for part in _INTEGRATED_H5AD_EXCLUDE)


def find_integrated_h5ad(work_dir: Path) -> Path | None:
    """Return final integrated h5ad under work_dir/saturn_results/, or None."""
    results_dir = work_dir / "saturn_results"
    if not results_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in results_dir.glob("*.h5ad") if _is_integrated_h5ad(p)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def find_genes_to_macrogenes_pkl(work_dir: Path) -> Path | None:
    """Return genes_to_macrogenes PKL matching the integrated run, or None."""
    integrated = find_integrated_h5ad(work_dir)
    if integrated is None:
        return None
    pkl_path = integrated.with_name(f"{integrated.stem}_genes_to_macrogenes.pkl")
    return pkl_path if pkl_path.exists() else None


def export_macrogene_weights_tsv(pkl_path: Path, out_path: Path) -> None:
    """Write long TSV: gene, species, macrogene, weight from SATURN PKL dict."""
    with open(pkl_path, "rb") as f:
        scores: dict[str, np.ndarray] = pickle.load(f)

    rows: list[dict[str, object]] = []
    for key, weights in scores.items():
        species, gene = key.split("_", 1)
        for macrogene, weight in enumerate(weights):
            rows.append(
                {
                    "gene": gene,
                    "species": species,
                    "macrogene": macrogene,
                    "weight": float(weight),
                }
            )

    pd.DataFrame(rows).to_csv(out_path, sep="\t", index=False)


def export_cell_clusters_tsv(adata, out_path: Path, leiden_col: str = "leiden") -> None:
    """Write barcode, species, leiden cluster assignments."""
    df = adata.obs[["species", leiden_col]].copy()
    df.index.name = "barcode"
    df.to_csv(out_path, sep="\t")
