# Demantiq Simulator — Milestone Tracker

## Dependency Graph

```
M1 (Core DGP)
├── M2 (Pricing + Distribution)    ─┐
├── M3 (Endogeneity + Competition) ─┼── M4 (Interactions)
│                                    │     └── M5 (Scenarios + Monte Carlo)
│                                    │           ├── M6 (Evaluation)
│                                    │           ├── M7 (Calibration)
│                                    │           └── M8 (Training Data)
└── Streamlit UI (incremental)
```

## Status

| Milestone | Status | Tests | Commit |
|-----------|--------|-------|--------|
| M1: Core DGP | **done** | 104 | `07b933e` |
| M2: Pricing + Distribution | **done** | +53 | `452229f` |
| M3: Endogeneity + Competition | **done** | (with M2) | `452229f` |
| M4: Interactions | **done** | +29 | `5f07884` |
| M5: Scenarios + Monte Carlo | **done** | +60 | `270301c` |
| M6: Evaluation | **done** | +68 | `5c4e3be` |
| M7: Calibration | **done** | +11 | `5c4e3be` |
| M8: Training Data | **done** | +22 | `15c136a` |
| UI: Streamlit pages | **done** | — | `5aaac59` |

**Total tests: 416**
