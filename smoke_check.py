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

from gene_id_remap import (  # noqa: E402
    LooksLikeEntrezIds,
    RemapAnnDataVarNamesToSymbols,
    ResolveSymbolColumn,
)
from label_resolve import ResolveLabelColumn, _find_cluster_column  # noqa: E402
from saturn_exports import (  # noqa: E402
    export_macrogene_weights_tsv,
    find_genes_to_macrogenes_pkl,
    find_integrated_h5ad,
)
from species_map import (  # noqa: E402
    BuildInDataCsv,
    PreflightGeneEmbeddingOverlap,
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
    assert path.name in {
        "human_embedding.torch",
        "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    }
    path_esm1b = ResolveEmbeddingPath("human", emb_dir, embedding_model="ESM1b")
    assert path_esm1b.name in {
        "human_embedding.torch",
        "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt",
    }
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


def test_preflight_gene_embedding_overlap(tmp: Path | None = None) -> None:
    import torch

    tmp = tmp or SATURN_ROOT / "work" / "smoke_overlap"
    tmp.mkdir(parents=True, exist_ok=True)

    genes = [f"Gene{i}" for i in range(10)]
    adata = _synthetic_adata(n_obs=20, n_vars=10)
    adata.var_names = genes
    h5ad_path = tmp / "human_saturn.h5ad"
    adata.write_h5ad(h5ad_path)

    ok_emb = {g: torch.zeros(4) for g in genes[:6]}
    ok_path = tmp / "human_ok.torch"
    torch.save(ok_emb, ok_path)
    stats = PreflightGeneEmbeddingOverlap(
        {"human": h5ad_path}, {"human": ok_path}
    )
    assert stats["human"]["n_matched"] == 6

    bad_emb = {f"Other{i}": torch.zeros(4) for i in range(5)}
    bad_path = tmp / "human_bad.torch"
    torch.save(bad_emb, bad_path)
    try:
        PreflightGeneEmbeddingOverlap({"human": h5ad_path}, {"human": bad_path})
    except ValueError as exc:
        msg = str(exc)
        assert "Gene–embedding overlap is empty or too low" in msg
        assert "human" in msg
        assert "build_entrez_symbol_maps.py" in msg
        assert "gene_remap" in msg
    else:
        raise AssertionError("expected ValueError for zero gene overlap")

    # Near-zero overlap (1/100) must also fail under the 5% floor
    genes100 = [f"Gene{i}" for i in range(100)]
    adata100 = _synthetic_adata(n_obs=10, n_vars=100)
    adata100.var_names = genes100
    h5ad100 = tmp / "human_100.h5ad"
    adata100.write_h5ad(h5ad100)
    low_emb = {genes100[0]: torch.zeros(4), "OtherA": torch.zeros(4)}
    low_path = tmp / "human_low.torch"
    torch.save(low_emb, low_path)
    try:
        PreflightGeneEmbeddingOverlap({"human": h5ad100}, {"human": low_path})
    except ValueError as exc:
        assert "too low" in str(exc) or "0.010" in str(exc) or "min 0.050" in str(exc)
    else:
        raise AssertionError("expected ValueError for near-zero gene overlap")


def test_gene_id_remap_from_var_column() -> None:
    adata = _synthetic_adata(n_obs=10, n_vars=5)
    adata.var_names = ["100", "101", "102", "103", "104"]
    adata.var["gene_symbol"] = ["ADA", "CDK", "GAPDH", "CD3D", "MissingDrop"]
    # Drop last via incomplete map by clearing one later — use mapping that drops none
    out, stats = RemapAnnDataVarNamesToSymbols(adata, "human", force=True)
    assert stats["remapped"] is True
    assert list(out.var_names) == ["ADA", "CDK", "GAPDH", "CD3D", "MissingDrop"]
    assert "entrez_id" in out.var.columns


def test_gene_id_remap_from_shared_genes_and_maps(tmp: Path | None = None) -> None:
    import torch

    tmp = tmp or SATURN_ROOT / "work" / "smoke_remap"
    tmp.mkdir(parents=True, exist_ok=True)

    assert LooksLikeEntrezIds(["100", "1000", "A1BG"]) is False
    assert LooksLikeEntrezIds(["100", "1000", "10000"]) is True

    shared = tmp / "shared_genes.csv"
    pd.DataFrame(
        {
            "human_entrez": [100, 2597, 915],
            "human_symbol": ["ADA", "GAPDH", "CD3D"],
            "mouse_symbol": ["Ada", "Gapdh", "Cd3d"],
            "macaque_symbol": ["ADA", "GAPDH", "CD3D"],
        }
    ).to_csv(shared, index=False)

    adata = _synthetic_adata(n_obs=8, n_vars=3)
    adata.var_names = ["100", "2597", "915"]
    out_h, stats_h = RemapAnnDataVarNamesToSymbols(
        adata, "human", shared_genes_path=shared
    )
    assert list(out_h.var_names) == ["ADA", "GAPDH", "CD3D"]
    assert stats_h["source"].startswith("shared_genes")

    out_m, _ = RemapAnnDataVarNamesToSymbols(
        adata.copy(), "mouse", shared_genes_path=shared
    )
    assert list(out_m.var_names) == ["Ada", "Gapdh", "Cd3d"]

    maps = tmp / "gene_maps"
    maps.mkdir(exist_ok=True)
    pd.DataFrame({"entrez": ["100", "2597"], "symbol": ["ADA", "GAPDH"]}).to_csv(
        maps / "human_entrez_to_human_symbol.tsv", sep="\t", index=False
    )
    adata2 = _synthetic_adata(n_obs=5, n_vars=3)
    adata2.var_names = ["100", "2597", "999999"]
    out2, stats2 = RemapAnnDataVarNamesToSymbols(
        adata2, "human", gene_maps_dir=maps
    )
    assert list(out2.var_names) == ["ADA", "GAPDH"]
    assert stats2["n_dropped"] == 1

    # End-to-end: remapped h5ad overlaps embedding keys
    h5ad_path = tmp / "human_remapped.h5ad"
    out_h.write_h5ad(h5ad_path)
    emb = {g: torch.zeros(2) for g in ["ADA", "GAPDH", "CD3D", "EXTRA"]}
    emb_path = tmp / "human.torch"
    torch.save(emb, emb_path)
    ov = PreflightGeneEmbeddingOverlap({"human": h5ad_path}, {"human": emb_path})
    assert ov["human"]["n_matched"] == 3


def test_gene_id_remap_skips_entrez_like_symbol_column(tmp: Path | None = None) -> None:
    """adata.var['symbol'] holding Entrez IDs must not win over gene_maps."""
    tmp = tmp or SATURN_ROOT / "work" / "smoke_remap_bad_symbol"
    tmp.mkdir(parents=True, exist_ok=True)

    maps = tmp / "gene_maps"
    maps.mkdir(exist_ok=True)
    pd.DataFrame(
        {"entrez": ["100", "2597", "915"], "symbol": ["ADA", "GAPDH", "CD3D"]}
    ).to_csv(maps / "human_entrez_to_human_symbol.tsv", sep="\t", index=False)

    adata = _synthetic_adata(n_obs=5, n_vars=3)
    adata.var_names = ["100", "2597", "915"]
    # Looks like a symbol column but values are still Entrez IDs
    adata.var["symbol"] = ["100", "2597", "915"]

    out, stats = RemapAnnDataVarNamesToSymbols(adata, "human", gene_maps_dir=maps)
    assert stats["remapped"] is True
    assert stats["source"].startswith("gene_maps:")
    assert list(out.var_names) == ["ADA", "GAPDH", "CD3D"]

    # Identity map (shared_genes mapping Entrez→Entrez) must raise after remap
    shared_bad = tmp / "shared_genes_identity.csv"
    pd.DataFrame(
        {
            "human_entrez": [100, 2597, 915],
            "human_symbol": ["100", "2597", "915"],
        }
    ).to_csv(shared_bad, index=False)
    adata_id = _synthetic_adata(n_obs=5, n_vars=3)
    adata_id.var_names = ["100", "2597", "915"]
    try:
        RemapAnnDataVarNamesToSymbols(
            adata_id, "human", shared_genes_path=shared_bad
        )
    except ValueError as exc:
        assert "still produced Entrez-like var_names" in str(exc)
    else:
        raise AssertionError("expected ValueError for Entrez-like remapped names")


def test_gene_id_remap_mixed_ids_uses_maps_not_silent_noop(
    tmp: Path | None = None,
) -> None:
    """70% Entrez + ENSG must remap via gene_maps (not silent noop)."""
    tmp = tmp or SATURN_ROOT / "work" / "smoke_remap_mixed"
    tmp.mkdir(parents=True, exist_ok=True)

    maps = tmp / "gene_maps"
    maps.mkdir(exist_ok=True)
    entrez_ids = [str(100 + i) for i in range(70)]
    symbols = [f"SYM{i}" for i in range(70)]
    pd.DataFrame({"entrez": entrez_ids, "symbol": symbols}).to_csv(
        maps / "human_entrez_to_human_symbol.tsv", sep="\t", index=False
    )

    n = 100
    names = entrez_ids + [f"ENSG{i}" for i in range(30)]
    assert LooksLikeEntrezIds(names, min_frac=0.8) is False
    assert LooksLikeEntrezIds(names, min_frac=0.5) is True

    adata = _synthetic_adata(n_obs=3, n_vars=n)
    adata.var_names = names
    out, stats = RemapAnnDataVarNamesToSymbols(adata, "human", gene_maps_dir=maps)
    assert stats["remapped"] is True
    assert stats["source"].startswith("gene_maps:")
    assert stats["n_mapped"] == 70
    assert stats["n_dropped"] == 30
    assert list(out.var_names[:3]) == ["SYM0", "SYM1", "SYM2"]
    assert not LooksLikeEntrezIds(out.var_names, min_frac=0.5)


def test_gene_id_remap_float_symbol_column_uses_maps(
    tmp: Path | None = None,
) -> None:
    """Float-string Entrez in a symbol column must not win over gene_maps."""
    tmp = tmp or SATURN_ROOT / "work" / "smoke_remap_float_symbol"
    tmp.mkdir(parents=True, exist_ok=True)

    maps = tmp / "gene_maps"
    maps.mkdir(exist_ok=True)
    pd.DataFrame(
        {"entrez": ["100", "1000", "10000"], "symbol": ["ADA", "CDH2", "AKT3"]}
    ).to_csv(maps / "human_entrez_to_human_symbol.tsv", sep="\t", index=False)

    # Float-form var_names alone should also remap (not silent noop)
    adata_float_names = _synthetic_adata(n_obs=5, n_vars=3)
    adata_float_names.var_names = ["100.0", "1000.0", "10000.0"]
    assert LooksLikeEntrezIds(list(adata_float_names.var_names)) is True
    out_f, stats_f = RemapAnnDataVarNamesToSymbols(
        adata_float_names, "human", gene_maps_dir=maps
    )
    assert stats_f["remapped"] is True
    assert list(out_f.var_names) == ["ADA", "CDH2", "AKT3"]

    adata = _synthetic_adata(n_obs=5, n_vars=3)
    adata.var_names = ["100", "1000", "10000"]
    adata.var["symbol"] = ["100.0", "1000.0", "10000.0"]
    assert ResolveSymbolColumn(adata.var) is None

    out, stats = RemapAnnDataVarNamesToSymbols(adata, "human", gene_maps_dir=maps)
    assert stats["remapped"] is True
    assert stats["source"].startswith("gene_maps:")
    assert list(out.var_names) == ["ADA", "CDH2", "AKT3"]


def test_gene_id_remap_feature_name_index_column_write_h5ad(
    tmp: Path | None = None,
) -> None:
    """Index name feature_name + column feature_name must still write_h5ad."""
    tmp = tmp or SATURN_ROOT / "work" / "smoke_remap_feature_name_write"
    tmp.mkdir(parents=True, exist_ok=True)

    maps = tmp / "gene_maps"
    maps.mkdir(exist_ok=True)
    pd.DataFrame(
        {"entrez": ["100", "1000", "10000"], "symbol": ["ADA", "CDH2", "AKT3"]}
    ).to_csv(maps / "human_entrez_to_human_symbol.tsv", sep="\t", index=False)

    adata = _synthetic_adata(n_obs=5, n_vars=3)
    adata.var_names = ["100", "1000", "10000"]
    adata.var_names.name = "feature_name"
    adata.var["feature_name"] = ["100", "1000", "10000"]

    out, stats = RemapAnnDataVarNamesToSymbols(adata, "human", gene_maps_dir=maps)
    assert stats["remapped"] is True
    assert list(out.var_names) == ["ADA", "CDH2", "AKT3"]
    assert out.var_names.name is None
    assert list(out.var["feature_name"]) == ["ADA", "CDH2", "AKT3"]
    assert "entrez_id" in out.var.columns

    out_path = tmp / "human_saturn.h5ad"
    out.write_h5ad(out_path)
    reloaded = ad.read_h5ad(out_path)
    assert list(reloaded.var_names) == ["ADA", "CDH2", "AKT3"]


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
    test_preflight_gene_embedding_overlap()
    test_gene_id_remap_from_var_column()
    test_gene_id_remap_from_shared_genes_and_maps()
    test_gene_id_remap_skips_entrez_like_symbol_column()
    test_gene_id_remap_mixed_ids_uses_maps_not_silent_noop()
    test_gene_id_remap_float_symbol_column_uses_maps()
    test_gene_id_remap_feature_name_index_column_write_h5ad()
    print("saturn smoke_check: OK")
