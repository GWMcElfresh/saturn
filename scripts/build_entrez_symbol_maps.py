#!/usr/bin/env python3
"""Build human-Entrez → species-symbol TSVs from NCBI gene_info + gene_orthologs.

ImpacTB GENE_HARMONIZE uses shared human Entrez IDs as var_names. SATURN needs
species-native gene symbols. This script writes:

  data/gene_maps/human_entrez_to_human_symbol.tsv
  data/gene_maps/human_entrez_to_mouse_symbol.tsv
  data/gene_maps/human_entrez_to_macaque_symbol.tsv

Usage:
  python scripts/build_entrez_symbol_maps.py
  python scripts/build_entrez_symbol_maps.py --out-dir /path/to/gene_maps
"""

from __future__ import annotations

import argparse
import gzip
import io
import urllib.request
from pathlib import Path

NCBI_GENE_INFO = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_info.gz"
NCBI_ORTHOLOGS = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene_orthologs.gz"

TAX_HUMAN = 9606
TAX_MOUSE = 10090
TAX_MACAQUE = 9544  # Macaca mulatta

SPECIES_TAX: dict[str, int] = {
    "human": TAX_HUMAN,
    "mouse": TAX_MOUSE,
    "macaque": TAX_MACAQUE,
}


def _Download(url: str) -> bytes:
    """Fetch URL; prefer curl (handles corporate TLS MITM), else urllib."""
    import shutil
    import subprocess

    print(f"Downloading {url} ...", flush=True)
    curl = shutil.which("curl")
    if curl:
        # ponytail: curl works when urllib hits CERTIFICATE_VERIFY_FAILED on lab Macs
        return subprocess.check_output(
            [curl, "-fsSL", "--retry", "3", url],
            timeout=3600,
        )
    try:
        with urllib.request.urlopen(url, timeout=600) as resp:
            return resp.read()
    except Exception:
        import ssl

        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, timeout=600, context=ctx) as resp:
            return resp.read()


def _LoadGeneInfoSymbols(raw: bytes, tax_ids: set[int]) -> dict[int, dict[int, str]]:
    """tax_id → {GeneID → Symbol} for requested taxa."""
    out: dict[int, dict[int, str]] = {t: {} for t in tax_ids}
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        header = fh.readline().decode("utf-8", errors="replace")
        # #tax_id GeneID Symbol ...
        for line in fh:
            parts = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            try:
                tax = int(parts[0])
            except ValueError:
                continue
            if tax not in out:
                continue
            try:
                gene_id = int(parts[1])
            except ValueError:
                continue
            symbol = parts[2].strip()
            if symbol and symbol != "-":
                out[tax][gene_id] = symbol
    for tax, mapping in out.items():
        print(f"  gene_info tax={tax}: {len(mapping):,} symbols", flush=True)
    return out


def _LoadOrthologsToHuman(
    raw: bytes, other_tax_ids: set[int]
) -> dict[int, dict[int, int]]:
    """other_tax → {human_GeneID → other_GeneID} using NCBI ortholog table."""
    # Columns: #tax_id GeneID relationship Other_tax_id Other_GeneID
    out: dict[int, dict[int, int]] = {t: {} for t in other_tax_ids}
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as fh:
        fh.readline()  # header
        for line in fh:
            parts = line.decode("utf-8", errors="replace").rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            try:
                tax_a = int(parts[0])
                gene_a = int(parts[1])
                tax_b = int(parts[3])
                gene_b = int(parts[4])
            except ValueError:
                continue
            # Prefer rows anchored on human
            if tax_a == TAX_HUMAN and tax_b in out:
                out[tax_b].setdefault(gene_a, gene_b)
            elif tax_b == TAX_HUMAN and tax_a in out:
                out[tax_a].setdefault(gene_b, gene_a)
    for tax, mapping in out.items():
        print(f"  orthologs human→tax={tax}: {len(mapping):,} pairs", flush=True)
    return out


def _WriteMap(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("entrez\tsymbol\n")
        for entrez, symbol in pairs:
            f.write(f"{entrez}\t{symbol}\n")
    print(f"Wrote {path} ({len(pairs):,} rows)", flush=True)


def BuildMaps(out_dir: Path) -> None:
    info_raw = _Download(NCBI_GENE_INFO)
    ortho_raw = _Download(NCBI_ORTHOLOGS)
    symbols = _LoadGeneInfoSymbols(
        info_raw, {TAX_HUMAN, TAX_MOUSE, TAX_MACAQUE}
    )
    orthologs = _LoadOrthologsToHuman(ortho_raw, {TAX_MOUSE, TAX_MACAQUE})

    human_syms = symbols[TAX_HUMAN]
    _WriteMap(
        out_dir / "human_entrez_to_human_symbol.tsv",
        sorted((str(gid), sym) for gid, sym in human_syms.items()),
    )

    for species, tax in (("mouse", TAX_MOUSE), ("macaque", TAX_MACAQUE)):
        other_syms = symbols[tax]
        human_to_other = orthologs[tax]
        pairs: list[tuple[str, str]] = []
        for human_gid, other_gid in human_to_other.items():
            sym = other_syms.get(other_gid)
            if sym:
                pairs.append((str(human_gid), sym))
        _WriteMap(
            out_dir / f"human_entrez_to_{species}_symbol.tsv",
            sorted(pairs, key=lambda x: int(x[0])),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_out = Path(__file__).resolve().parent.parent / "data" / "gene_maps"
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=default_out,
        help=f"Output directory (default: {default_out})",
    )
    args = parser.parse_args()
    BuildMaps(args.out_dir)


if __name__ == "__main__":
    main()
