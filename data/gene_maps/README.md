# Gene ID maps for SATURN

ImpacTB GENE_HARMONIZE stores **human Entrez IDs** in `var_names` across species.
SATURN protein embeddings are keyed by **species-native gene symbols**.

Generate the TSVs (not committed; download once on HPC or locally):

```bash
python scripts/build_entrez_symbol_maps.py
```

Produces:

- `human_entrez_to_human_symbol.tsv`
- `human_entrez_to_mouse_symbol.tsv`
- `human_entrez_to_macaque_symbol.tsv`

Override location with `GENE_MAPS_DIR`. Prefer `shared_genes.csv` or an `adata.var` symbol column when available.
