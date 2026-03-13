# Demantiq Simulator — Milestone Tracker

## Dependency Graph

```
M1 (Core DGP)
├── M2 (Pricing + Distribution)    ─┐
├── M3 (Endogeneity + Competition) ─┼── M4 (Interactions)
│                                    │     └── M5 (Scenarios + Monte Carlo)
│                                    │           └── M6 (Evaluation)
│                                    │                 └── M7 (Calibration)
│                                    │                       └── M8 (Training Data)
└── Streamlit UI (incremental)
```

## Status

| Milestone | Status | Dependencies | Acceptance |
|-----------|--------|--------------|------------|
| M1: Core DGP | pending | none | SCN-001 valid, OLS recovers betas within 5% |
| M2: Pricing + Distribution | pending | M1 | SCN-004, SCN-011 valid |
| M3: Endogeneity + Competition | pending | M1 | SCN-003 bias matches analytical within 10% |
| M4: Interactions | pending | M2, M3 | SCN-005 valid, interaction magnitudes correct |
| M5: Scenarios + Monte Carlo | pending | M4 | 15 scenarios valid, 10k sims < 60 min |
| M6: Evaluation | pending | M5 | Head-to-head comparison produces structured results |
| M7: Calibration | pending | M5 | Realism validator passes on all scenarios |
| M8: Training Data | pending | M5 | 1M+ tuples generated |
| UI: Streamlit pages | pending | incremental | All pages render without errors |
