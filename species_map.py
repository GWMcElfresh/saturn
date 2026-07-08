"""Manifest species → SATURN protein embedding paths."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# Candidate filenames per species/model. First entry is the Stanford bundle layout
# (protein_embeddings.tar.gz → protein_embeddings_export/{ESM1b,ESM2}/<key>_embedding.torch).
# Later entries support legacy lab-internal .pt names.
_EMBEDDING_CANDIDATES: dict[str, dict[str, list[str]]] = {
    "ESM1b": {
        "human": [
            "human_embedding.torch",
            "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt",
        ],
        "mouse": [
            "mouse_embedding.torch",
            "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt",
        ],
        "macaca_mulatta": [
            "macaca_mulatta_embedding.torch",
            "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM1b.pt",
        ],
        "macaca_fascicularis": [
            "macaca_fascicularis_embedding.torch",
            "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM1b.pt",
        ],
    },
    "ESM2": {
        "human": [
            "human_embedding.torch",
            "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
        ],
        "mouse": [
            "mouse_embedding.torch",
            "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
        ],
        "macaca_mulatta": [
            "macaca_mulatta_embedding.torch",
            "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
        ],
        "macaca_fascicularis": [
            "macaca_fascicularis_embedding.torch",
            "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
        ],
    },
}

_MANIFEST_TO_EMBEDDING_KEY: dict[str, str] = {
    "human": "human",
    "mouse": "mouse",
    "macaque": "macaca_mulatta",
}


def ResolveEmbeddingKey(manifest_species: str) -> str:
    """Map integration manifest species name to SATURN embedding key."""
    if manifest_species in _MANIFEST_TO_EMBEDDING_KEY:
        if manifest_species == "macaque":
            override = os.environ.get("MACAQUE_EMBEDDING_SPECIES", "macaca_mulatta").strip()
            return override or "macaca_mulatta"
        return _MANIFEST_TO_EMBEDDING_KEY[manifest_species]
    return manifest_species


def _EmbeddingCandidates(embedding_model: str, key: str) -> list[str]:
    candidates = _EMBEDDING_CANDIDATES.get(embedding_model, {}).get(key)
    if not candidates:
        raise RuntimeError(
            f"No {embedding_model} embedding file mapped for species key '{key}'. "
            "Set MACAQUE_EMBEDDING_SPECIES or embedding_path in in_data.csv."
        )
    return candidates


def ResolveEmbeddingPath(
    manifest_species: str,
    embeddings_dir: Path,
    embedding_model: str = "ESM2",
) -> Path:
    """Return path to protein embedding file for a manifest species."""
    key = ResolveEmbeddingKey(manifest_species)
    try:
        candidates = _EmbeddingCandidates(embedding_model, key)
    except RuntimeError as exc:
        raise RuntimeError(
            f"No {embedding_model} embedding file mapped for species '{manifest_species}' "
            f"(key={key}). Set MACAQUE_EMBEDDING_SPECIES or embedding_path in in_data.csv."
        ) from exc
    for filename in candidates:
        path = embeddings_dir / filename
        if path.exists():
            return path
    return embeddings_dir / candidates[0]


def RequiredEmbeddingPaths(
    species_order: list[str],
    embeddings_dir: Path,
    embedding_model: str = "ESM2",
) -> dict[str, Path]:
    """Return manifest_species → embedding path for all species."""
    return {
        species: ResolveEmbeddingPath(species, embeddings_dir, embedding_model)
        for species in species_order
    }


def BuildInDataCsv(
    species_order: list[str],
    h5ad_paths: dict[str, Path],
    embedding_paths: dict[str, Path],
    label_cols: dict[str, str],
    out_path: Path,
) -> Path:
    """Write SATURN in_data CSV (index_col=species)."""
    rows = []
    for species in species_order:
        rows.append(
            {
                "species": species,
                "path": str(h5ad_paths[species].resolve()),
                "embedding_path": str(embedding_paths[species].resolve()),
                "in_label_col": label_cols[species],
            }
        )
    df = pd.DataFrame(rows).set_index("species")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path)
    return out_path


def ProteinEmbeddingsDownloadCommand(embeddings_dir: Path) -> str:
    """Shell command to fetch SATURN protein embeddings."""
    extract_root = embeddings_dir.parent.parent
    return (
        f"mkdir -p {extract_root} && "
        "curl -L http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz | "
        f"tar -xz -C {extract_root}"
    )
