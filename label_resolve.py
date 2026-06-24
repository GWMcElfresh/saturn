"""Label column resolution for SATURN metric learning."""

from __future__ import annotations

import re
from typing import Literal

import scanpy as sc
from anndata import AnnData

LabelSource = Literal["cell_type", "cluster", "computed"]

_CELL_TYPE_NAMES = frozenset(
    {
        "cell_type",
        "celltype",
        "cell_type_ontology",
        "annotation",
        "annot",
        "cell_annotation",
        "celltypes",
        "author_cell_type",
        "celltype_annotation",
    }
)

_CLUSTER_EXACT = frozenset({"leiden", "louvain", "cluster", "clusters"})
_CLUSTER_PREFIXES = ("leiden", "louvain", "cluster")
_RESOLUTION_RE = re.compile(r"(?:res[_-]?)?(\d+(?:[._]\d+)?)", re.IGNORECASE)


def _check_clustering_deps() -> None:
    """Fail fast with install hint if clustering backends are missing."""
    try:
        import igraph  # noqa: F401
        import leidenalg  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Clustering dependencies missing (igraph, leidenalg). "
            "Install saturn env: pip install -e '.[clustering]' or uv sync in saturn/"
        ) from exc


def _obs_col_map(adata: AnnData) -> dict[str, str]:
    return {str(c).lower(): str(c) for c in adata.obs.columns}


def _find_cell_type_column(adata: AnnData) -> str | None:
    col_map = _obs_col_map(adata)
    for name in _CELL_TYPE_NAMES:
        if name in col_map:
            return col_map[name]
    return None


def _parse_resolution(col_name: str) -> float | None:
    match = _RESOLUTION_RE.search(col_name)
    if not match:
        return None
    return float(match.group(1).replace("_", "."))


def _cluster_kind(col_name: str) -> str:
    lower = col_name.lower()
    for prefix in _CLUSTER_PREFIXES:
        if lower == prefix or lower.startswith(f"{prefix}_") or lower.startswith(f"{prefix}-"):
            return prefix
    return "cluster"


def _find_cluster_column(adata: AnnData, preferred_resolution: float) -> str | None:
    col_map = _obs_col_map(adata)
    candidates: list[tuple[float, int, str]] = []
    for lower, original in col_map.items():
        if lower in _CLUSTER_EXACT:
            kind_rank = 0 if lower in {"leiden", "cluster", "clusters"} else 1
            candidates.append((abs(preferred_resolution - 0.5), kind_rank, original))
            continue
        kind = _cluster_kind(lower)
        if kind in _CLUSTER_PREFIXES and lower != kind:
            res = _parse_resolution(lower)
            if res is not None:
                kind_rank = 0 if kind == "leiden" else 1
                candidates.append((abs(res - preferred_resolution), kind_rank, original))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (t[0], t[1], t[2]))
    return candidates[0][2]


def ComputeClusterLabels(
    adata: AnnData,
    resolution: float = 0.5,
    n_neighbors: int = 30,
    random_state: int = 42,
) -> str:
    """Run scanpy neighbors + leiden; write obs['saturn_leiden_proxy']; return column name."""
    _check_clustering_deps()
    work = adata.copy()
    sc.pp.normalize_total(work, target_sum=1e4)
    sc.pp.log1p(work)
    n_hvg = min(2000, work.n_vars)
    if n_hvg >= 2:
        sc.pp.highly_variable_genes(work, n_top_genes=n_hvg, flavor="seurat_v3")
    n_comps = min(50, work.n_obs - 1, work.n_vars - 1)
    if n_comps >= 1:
        sc.pp.pca(work, n_comps=n_comps)
    n_nb = min(n_neighbors, work.n_obs - 1)
    sc.pp.neighbors(work, n_neighbors=n_nb, random_state=random_state)
    sc.tl.leiden(
        work,
        resolution=resolution,
        random_state=random_state,
        key_added="saturn_leiden_proxy",
    )
    adata.obs["saturn_leiden_proxy"] = work.obs["saturn_leiden_proxy"].astype(str)
    return "saturn_leiden_proxy"


def ResolveLabelColumn(
    adata: AnnData,
    preferred_resolution: float = 0.5,
    *,
    explicit_col: str | None = None,
    compute_if_missing: bool = True,
    n_neighbors: int = 30,
    random_state: int = 42,
) -> tuple[str, LabelSource]:
    """Resolve label column: cell type → cluster → on-the-fly Leiden."""
    if explicit_col:
        if explicit_col not in adata.obs.columns:
            raise RuntimeError(
                f"IN_LABEL_COL='{explicit_col}' not in obs columns: {list(adata.obs.columns)}"
            )
        return explicit_col, "cell_type"

    cell_type_col = _find_cell_type_column(adata)
    if cell_type_col is not None:
        return cell_type_col, "cell_type"

    cluster_col = _find_cluster_column(adata, preferred_resolution)
    if cluster_col is not None:
        return cluster_col, "cluster"

    if not compute_if_missing:
        raise RuntimeError(
            "No cell-type or clustering column found and compute_if_missing=False."
        )

    col = ComputeClusterLabels(
        adata,
        resolution=preferred_resolution,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )
    return col, "computed"
