# `data/raw/` — real wearable data (not included)

Drop-in location for **real** per-participant records. Empty and git-ignored: the
TemPredict / Oura Ring data used in the paper are **not redistributable** (Oura's
data-use policy; see the manuscript's Data Availability statement). Self-report
data can be requested from the study authors.

**Do not commit participant data here.**

## Expected schema

One **Parquet** file per participant, `<pid>.parquet`, minute-resolution, with:

| column      | dtype | description                              |
|-------------|-------|------------------------------------------|
| `temp_skin` | float | distal skin temperature (°C)             |
| `is_awake`  | bool  | sleep/wake label                         |
| `met`       | float | metabolic equivalents (optional; ANCOVA) |
| `day`       | int   | day index, to group segments             |

Plus `demographics.parquet` (or `.csv`) indexed by `pid` with `age`, `sex`
(`male`/`female`), and `group` (`DM`/`No-DM`).

Then point a stage script at it:

```bash
python scripts/02_diabetes_analysis.py --data-dir data/raw
```

The schema is validated on load (`msaad.io.load_participant`). Only `temp_skin`
and `is_awake` are strictly required.
