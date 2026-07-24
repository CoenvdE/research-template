# Datasets

Living overview, kept current by the `dataset-overview` skill and the
anti-drift hook. One section per dataset.

## synthetic (shipped with template)

- **What**: deterministic random-linear regression, `y = x @ W / sqrt(d)`, targets ~N(0, 1). Exists so tests and smoke runs work anywhere, CPU-only, no download.
- **Location & format**: generated in-memory by `src/data/synthetic.py` from a seed.
- **Size**: configurable; defaults n_train=512, n_val=128, in_dim=8, out_dim=4.
- **Splits**: index-disjoint, pseudo-temporal ordering with a configurable gap (demonstrates the leakage-test pattern).
- **Preprocessing**: none; inputs and targets are already ~N(0, 1).

<!-- Real datasets: add name/source/license, path + format (zarr/netCDF/parquet),
sizes and splits, variables with dims/dtype/units and coordinate ranges, and
normalization stats + where they were computed. Inspect the real data; never guess. -->
