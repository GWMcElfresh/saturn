#!/usr/bin/env python3
"""Runnable self-check for saturn/ modules (ponytail: smallest thing that fails if logic breaks)."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pickle

SATURN_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SATURN_ROOT))

from label_resolve import ResolveLabelColumn, _find_cluster_column  # noqa: E402
from saturn_exports import (  # noqa: E402
    export_macrogene_weights_tsv,
    find_genes_to_macrogenes_pkl,
    find_integrated_h5ad,
)
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
    emb_dir = SATURN_ROOT / "data" / "protein_embeddings_export" / "ESM2"
    path = ResolveEmbeddingPath("human", emb_dir, embedding_model="ESM2")
    assert path.name.endswith("ESM2.pt")
    path_esm1b = ResolveEmbeddingPath("human", emb_dir, embedding_model="ESM1b")
    assert path_esm1b.name.endswith("ESM1b.pt")
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


def test_find_integrated_h5ad(tmp: Path | None = None) -> None:
    tmp = tmp or SATURN_ROOT / "work" / "smoke_exports"
    results = tmp / "saturn_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "run_pretrain.h5ad").write_text("stub")
    (results / "run_ep_5.h5ad").write_text("stub")
    final = results / "run.h5ad"
    final.write_text("stub")
    assert find_integrated_h5ad(tmp) == final


def test_find_integrated_h5ad_prefers_newest(tmp: Path | None = None) -> None:
    tmp = tmp or SATURN_ROOT / "work" / "smoke_exports_newest"
    results = tmp / "saturn_results"
    results.mkdir(parents=True, exist_ok=True)
    older = results / "aaa_run.h5ad"
    newer = results / "zzz_run.h5ad"
    older.write_text("stub")
    newer.write_text("stub")
    time.sleep(0.05)
    newer.write_text("stub2")
    assert find_integrated_h5ad(tmp) == newer


def test_export_macrogene_weights_tsv(tmp: Path | None = None) -> None:
    tmp = tmp or SATURN_ROOT / "work" / "smoke_exports"
    tmp.mkdir(parents=True, exist_ok=True)
    pkl_path = tmp / "weights.pkl"
    out_path = tmp / "macrogene_weights.tsv"
    with open(pkl_path, "wb") as f:
        pickle.dump({"human_CD4": np.array([1.0, 2.0]), "mouse_Gapdh": np.array([3.0])}, f)
    export_macrogene_weights_tsv(pkl_path, out_path)
    df = pd.read_csv(out_path, sep="\t")
    assert list(df.columns) == ["gene", "species", "macrogene", "weight"]
    assert len(df) == 3
    assert set(df["species"]) == {"human", "mouse"}


def test_find_genes_to_macrogenes_pkl(tmp: Path | None = None) -> None:
    tmp = tmp or SATURN_ROOT / "work" / "smoke_exports_pkl"
    results = tmp / "saturn_results"
    results.mkdir(parents=True, exist_ok=True)
    (results / "run.h5ad").write_text("stub")
    pkl = results / "run_genes_to_macrogenes.pkl"
    pkl.write_text("stub")
    assert find_genes_to_macrogenes_pkl(tmp) == pkl


if __name__ == "__main__":
    test_cell_type_detection()
    test_cluster_resolution_picker()
    test_compute_fallback()
    test_species_map()
    test_build_in_data_csv()
    test_find_integrated_h5ad()
    test_find_integrated_h5ad_prefers_newest()
    test_export_macrogene_weights_tsv()
    test_find_genes_to_macrogenes_pkl()
    print("saturn smoke_check: OK")
