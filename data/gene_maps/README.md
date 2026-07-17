# Gene ID maps for SATURN

ImpacTB GENE_HARMONIZE stores **human Entrez IDs** in `var_names` across species.
SATURN protein embeddings are keyed by **species-native gene symbols**.

These TSVs are **gitignored** — generate once on each machine (HPC or local):

```bash
python scripts/build_entrez_symbol_maps.py
```

Produces:

- `human_entrez_to_human_symbol.tsv`
- `human_entrez_to_mouse_symbol.tsv`
- `human_entrez_to_macaque_symbol.tsv`

Override location with `GENE_MAPS_DIR`. Prefer `shared_genes.csv` or a real `adata.var` symbol column when available (Entrez-like symbol columns are ignored).

## Empty gene–embedding overlap on HPC

If preflight reports `0/... genes match embedding` with numeric `sample_var_names` (e.g. `100`, `1000`) vs symbol `sample_embedding_keys` (e.g. `A1BG`):

1. Build maps (above) and confirm the three TSVs exist under this directory (or `GENE_MAPS_DIR`).
2. Clear stale SATURN inputs: `rm -f cache/saturn_inputs/*_saturn.h5ad`
3. Re-run (`SATURN_DRY_RUN=1` is enough to validate). Logs should show
   `SATURN_IMPACTB: gene_remap ... mapped=.../...` with a `gene_maps:` or `shared_genes:` source — not `noop`, and not still-numeric `var_names`.

```bash
rm -f cache/saturn_inputs/*_saturn.h5ad
SATURN_DRY_RUN=1 python impac_tb_saturn.py
# expect: gene_remap source=gene_maps:... and gene_overlap matched>0
```
