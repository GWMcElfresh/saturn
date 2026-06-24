"""Manifest species → SATURN protein embedding paths."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

# ESM1b filenames from snap.stanford.edu/saturn protein_embeddings bundle
_EMBEDDING_FILES: dict[str, dict[str, str]] = {
    "ESM1b": {
        "human": "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM1b.pt",
        "mouse": "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM1b.pt",
        "macaca_mulatta": "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM1b.pt",
        "macaca_fascicularis": (
            "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM1b.pt"
        ),
    }
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


def ResolveEmbeddingPath(
    manifest_species: str,
    embeddings_dir: Path,
    embedding_model: str = "ESM1b",
) -> Path:
    """Return path to protein embedding .pt for a manifest species."""
    key = ResolveEmbeddingKey(manifest_species)
    files = _EMBEDDING_FILES.get(embedding_model, {})
    if key not in files:
        raise RuntimeError(
            f"No {embedding_model} embedding file mapped for species '{manifest_species}' "
            f"(key={key}). Set MACAQUE_EMBEDDING_SPECIES or embedding_path in in_data.csv."
        )
    return embeddings_dir / files[key]


def RequiredEmbeddingPaths(
    species_order: list[str],
    embeddings_dir: Path,
    embedding_model: str = "ESM1b",
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
    parent = embeddings_dir.parent
    return (
        f"mkdir -p {parent} && "
        "curl -L http://snap.stanford.edu/saturn/data/protein_embeddings.tar.gz | "
        f"tar -xz -C {parent} && "
        f"mv {parent}/protein_embeddings/* {embeddings_dir}/ 2>/dev/null || true"
    )
