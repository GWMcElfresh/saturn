"""Remap Entrez-like AnnData var_names to species-native gene symbols for SATURN."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData

# ImpacTB GENE_HARMONIZE stores shared human Entrez IDs in var_names across species.
# SATURN embeddings are keyed by species-native symbols, so we remap before training.
_ENTREZ_RE = re.compile(r"^\d+$")

_SYMBOL_COL_CANDIDATES = (
    "gene_symbol",
    "feature_name",
    "symbol",
    "gene_name",
    "GeneSymbol",
    "Gene_Symbol",
    "SYMBOL",
)

# shared_genes.csv / gene map: human Entrez → per-species symbol
_SPECIES_SYMBOL_COL_CANDIDATES: dict[str, tuple[str, ...]] = {
    "human": (
        "human_symbol",
        "symbol_human",
        "human_gene",
        "gene_symbol_human",
        "hsapiens_symbol",
    ),
    "mouse": (
        "mouse_symbol",
        "symbol_mouse",
        "mouse_gene",
        "gene_symbol_mouse",
        "mmusculus_symbol",
    ),
    "macaque": (
        "macaque_symbol",
        "macaca_mulatta_symbol",
        "symbol_macaque",
        "macaque_gene",
        "gene_symbol_macaque",
        "mmulatta_symbol",
    ),
}

_ENTREZ_COL_CANDIDATES = (
    "entrez",
    "entrez_id",
    "EntrezID",
    "GeneID",
    "gene_id",
    "human_entrez",
    "human_gene_id",
    "ncbi_gene_id",
)


def LooksLikeEntrezIds(var_names: list[str] | pd.Index, min_frac: float = 0.8) -> bool:
    """True when most var_names are purely numeric (Entrez-like)."""
    names = [str(g) for g in var_names]
    if not names:
        return False
    n_entrez = sum(1 for g in names if _ENTREZ_RE.match(g))
    return (n_entrez / len(names)) >= min_frac


def ResolveSymbolColumn(var: pd.DataFrame) -> str | None:
    """Return adata.var column that looks like gene symbols, if any."""
    for col in _SYMBOL_COL_CANDIDATES:
        if col in var.columns:
            return col
    return None


def _PickColumn(columns: list[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand.lower() in lower:
            return lower[cand.lower()]
    return None


def LoadSharedGenesMap(path: Path, species: str) -> dict[str, str] | None:
    """Load human-Entrez → species-symbol from shared_genes.csv-like table."""
    if path is None or not Path(path).exists():
        return None
    df = pd.read_csv(path)
    cols = list(df.columns)
    entrez_col = _PickColumn(cols, _ENTREZ_COL_CANDIDATES)
    symbol_cols = _SPECIES_SYMBOL_COL_CANDIDATES.get(species, ())
    symbol_col = _PickColumn(cols, symbol_cols)
    # Fallback: generic symbol column when table is single-species
    if symbol_col is None and species == "human":
        symbol_col = _PickColumn(cols, _SYMBOL_COL_CANDIDATES)
    if entrez_col is None or symbol_col is None:
        return None
    mapping: dict[str, str] = {}
    for entrez, symbol in zip(df[entrez_col], df[symbol_col]):
        if pd.isna(entrez) or pd.isna(symbol):
            continue
        e = str(int(entrez)) if isinstance(entrez, (int, float, np.integer, np.floating)) else str(entrez).strip()
        s = str(symbol).strip()
        if _ENTREZ_RE.match(e) and s and s.lower() not in {"nan", "none", "na"}:
            mapping[e] = s
    return mapping or None


def GeneMapTsvPath(gene_maps_dir: Path, species: str) -> Path:
    """Path to static human-Entrez → species-symbol TSV."""
    return Path(gene_maps_dir) / f"human_entrez_to_{species}_symbol.tsv"


def LoadEntrezSymbolMap(gene_maps_dir: Path, species: str) -> dict[str, str] | None:
    """Load TSV with columns entrez,symbol (keys = human Entrez IDs)."""
    path = GeneMapTsvPath(gene_maps_dir, species)
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", dtype=str)
    cols = {c.lower(): c for c in df.columns}
    entrez_col = cols.get("entrez") or cols.get("entrez_id") or cols.get("geneid")
    symbol_col = cols.get("symbol") or cols.get("gene_symbol")
    if entrez_col is None or symbol_col is None:
        raise ValueError(
            f"Gene map {path} must have columns 'entrez' and 'symbol'; "
            f"got {list(df.columns)}"
        )
    mapping: dict[str, str] = {}
    for entrez, symbol in zip(df[entrez_col], df[symbol_col]):
        if pd.isna(entrez) or pd.isna(symbol):
            continue
        e = str(entrez).strip()
        s = str(symbol).strip()
        if e and s and s.lower() not in {"nan", "none", "na"}:
            mapping[e] = s
    return mapping or None


def ResolveEntrezToSymbolMap(
    species: str,
    adata: AnnData,
    *,
    shared_genes_path: Path | None = None,
    gene_maps_dir: Path | None = None,
) -> tuple[dict[str, str], str]:
    """Resolve mapping source: adata.var column → shared_genes.csv → gene_maps TSV."""
    symbol_col = ResolveSymbolColumn(adata.var)
    if symbol_col is not None:
        mapping = {
            str(eid): str(sym).strip()
            for eid, sym in zip(adata.var_names, adata.var[symbol_col])
            if pd.notna(sym) and str(sym).strip()
        }
        return mapping, f"adata.var['{symbol_col}']"

    if shared_genes_path is not None:
        shared = LoadSharedGenesMap(Path(shared_genes_path), species)
        if shared:
            return shared, f"shared_genes:{shared_genes_path}"

    if gene_maps_dir is not None:
        static = LoadEntrezSymbolMap(Path(gene_maps_dir), species)
        if static:
            return static, f"gene_maps:{GeneMapTsvPath(gene_maps_dir, species)}"

    raise FileNotFoundError(
        f"No Entrez→symbol map for species={species}. Tried adata.var symbol columns, "
        f"shared_genes_path={shared_genes_path!s}, and "
        f"{GeneMapTsvPath(gene_maps_dir or Path('data/gene_maps'), species)}. "
        "Run: python scripts/build_entrez_symbol_maps.py"
    )


def RemapAnnDataVarNamesToSymbols(
    adata: AnnData,
    species: str,
    *,
    shared_genes_path: Path | None = None,
    gene_maps_dir: Path | None = None,
    force: bool = False,
) -> tuple[AnnData, dict[str, Any]]:
    """Remap Entrez-like var_names to species symbols; no-op if already symbolic.

    Drops unmapped genes and collapses duplicate symbols (keeps first). Stores
    original IDs in ``adata.var['entrez_id']``.
    """
    var_names = [str(g) for g in adata.var_names]
    stats: dict[str, Any] = {
        "species": species,
        "n_input": len(var_names),
        "remapped": False,
        "source": "noop",
        "n_mapped": len(var_names),
        "n_dropped": 0,
        "n_dup_collapsed": 0,
        "n_output": len(var_names),
    }
    if not force and not LooksLikeEntrezIds(var_names):
        return adata, stats

    mapping, source = ResolveEntrezToSymbolMap(
        species,
        adata,
        shared_genes_path=shared_genes_path,
        gene_maps_dir=gene_maps_dir,
    )
    symbols: list[str] = []
    keep_idx: list[int] = []
    for i, eid in enumerate(var_names):
        sym = mapping.get(eid)
        if sym is None:
            continue
        symbols.append(sym)
        keep_idx.append(i)

    n_mapped = len(keep_idx)
    n_dropped = len(var_names) - n_mapped
    if n_mapped == 0:
        raise ValueError(
            f"Entrez→symbol remap produced 0 mapped genes for species={species} "
            f"(source={source}, n_input={len(var_names)}, map_size={len(mapping)}). "
            f"sample_var_names={var_names[:5]}; sample_map_keys={list(mapping)[:5]}"
        )

    out = adata[:, keep_idx].copy()
    out.var["entrez_id"] = var_names_kept = [var_names[i] for i in keep_idx]
    # Collapse duplicate symbols (keep first occurrence)
    seen: set[str] = set()
    uniq_idx: list[int] = []
    uniq_symbols: list[str] = []
    uniq_entrez: list[str] = []
    for j, sym in enumerate(symbols):
        key = sym.lower()
        if key in seen:
            continue
        seen.add(key)
        uniq_idx.append(j)
        uniq_symbols.append(sym)
        uniq_entrez.append(var_names_kept[j])
    n_dup = n_mapped - len(uniq_idx)
    if n_dup:
        out = out[:, uniq_idx].copy()
        out.var["entrez_id"] = uniq_entrez
    out.var_names = pd.Index(uniq_symbols, name=out.var_names.name)
    out.var_names_make_unique()

    stats.update(
        {
            "remapped": True,
            "source": source,
            "n_mapped": n_mapped,
            "n_dropped": n_dropped,
            "n_dup_collapsed": n_dup,
            "n_output": out.n_vars,
        }
    )
    print(
        f"SATURN_IMPACTB: gene_remap species={species} source={source} "
        f"mapped={n_mapped}/{len(var_names)} dropped={n_dropped} "
        f"dup_collapsed={n_dup} n_out={out.n_vars}",
        flush=True,
    )
    return out, stats
