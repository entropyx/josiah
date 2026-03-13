# Josiah / Demantiq Simulator — Documentation

## Requirements

- [Product Requirements Document (PRD)](prd.md) — Full specification for the Demantiq Synthetic Dataset Generator.

## Design

- [Design Spec](specs/2026-03-13-demantiq-simulator-design.md) — Architecture decisions, milestone plan, demand kernel pipeline, agent orchestration.

## Project Management

- [Milestones](project/milestones.md) — Milestone tracker with dependency graph and status.
- [Stage Status](project/stage-status.md) — Current active milestone and work items.

## Workflows (Claude Flow)

Per-milestone workflow files for autonomous agent-driven development:

- [Coordinator](workflows/coordinator.yml) — Top-level orchestrator
- [M1: Core DGP](workflows/m1-core-dgp.yml) — Demand kernel foundation
- [M2: Pricing + Distribution](workflows/m2-pricing-dist.yml) — Pricing engine, distribution module
- [M3: Endogeneity + Competition](workflows/m3-endog-comp.yml) — Endogeneity, competition, macro
- [M4: Interactions](workflows/m4-interactions.yml) — Cross-variable interactions
- [M5: Scenarios + Monte Carlo](workflows/m5-scenarios-mc.yml) — 15 named scenarios, batch runner
- [M6: Evaluation](workflows/m6-evaluation.yml) — Parameter recovery, model comparison (API-only)
- [M7: Calibration](workflows/m7-calibration.yml) — Empirical distributions, realism validator (API-only)
- [M8: Training Data](workflows/m8-training-data.yml) — Large-scale training data pipeline (API-only)
