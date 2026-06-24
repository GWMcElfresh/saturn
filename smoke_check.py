#!/usr/bin/env python3
"""Runnable self-check for saturn/ modules (ponytail: smallest thing that fails if logic breaks)."""

from __future__ import annotations

import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

SATURN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SATURN_ROOT))

from label_resolve import ResolveLabelColumn, _find_cluster_column  # noqa: E402
from species_map import (  # noqa: E402
    BuildInDataCsv,
    ProteinEmbeddingsDownloadCommand,
    ResolveEmbeddingKey,
    ResolveEmbeddingPath,
)


def _synthetic_adata(n_obs: int = 80, n_vars: int = 100) -> ad.AnnData:
    rng = np.random.default_rng(0)
    x = rng.poisson(3, size=(n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame({"donor_id": [f"d{i % 4}" for i in range(n_obs)]})
    return ad.AnnData(X=x, obs=obs)


def test_cell_type_detection() -> None:
    adata = _synthetic_adata()
    adata.obs["cell_type"] = [f"T{i % 3}" for i in range(adata.n_obs)]
    col, source = ResolveLabelColumn(adata)
    assert col == "cell_type" and source == "cell_type"


def test_cluster_resolution_picker() -> None:
    adata = _synthetic_adata()
    adata.obs["leiden_0.3"] = "0"
    adata.obs["leiden_0.5"] = "1"
    adata.obs["leiden_0.8"] = "2"
    col = _find_cluster_column(adata, preferred_resolution=0.5)
    assert col == "leiden_0.5"


def test_compute_fallback() -> None:
    adata = _synthetic_adata()
    col, source = ResolveLabelColumn(adata, preferred_resolution=0.5)
    assert source == "computed"
    assert col == "saturn_leiden_proxy"
    assert "saturn_leiden_proxy" in adata.obs.columns


def test_species_map() -> None:
    assert ResolveEmbeddingKey("macaque") == "macaca_mulatta"
    emb_dir = SATURN_ROOT / "data" / "protein_embeddings"
    path = ResolveEmbeddingPath("human", emb_dir)
    assert path.name.endswith("ESM1b.pt")
    cmd = ProteinEmbeddingsDownloadCommand(emb_dir)
    assert "protein_embeddings.tar.gz" in cmd


def test_build_in_data_csv(tmp: Path | None = None) -> None:
    tmp = tmp or SATURN_ROOT / "work" / "smoke"
    tmp.mkdir(parents=True, exist_ok=True)
    species = ["human", "mouse"]
    h5ads = {s: tmp / f"{s}.h5ad" for s in species}
    embs = {s: tmp / f"{s}.pt" for s in species}
    labels = {s: "cell_type" for s in species}
    for p in h5ads.values():
        _synthetic_adata().write_h5ad(p)
    for p in embs.values():
        p.write_text("stub")
    out = tmp / "in_data.csv"
    BuildInDataCsv(species, h5ads, embs, labels, out)
    df = pd.read_csv(out, index_col="species")
    assert list(df.index) == species


if __name__ == "__main__":
    test_cell_type_detection()
    test_cluster_resolution_picker()
    test_compute_fallback()
    test_species_map()
    test_build_in_data_csv()
    print("saturn smoke_check: OK")
