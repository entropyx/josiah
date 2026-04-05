# Project Overview

## What is Demantiq?

Demantiq is a neural inference engine for Marketing Mix Modeling (MMM) that replaces traditional MCMC-based fitting with instant/fast inference. It sits alongside a synthetic demand simulator that generates data with known ground truth.

## The Goal

Given a client's observable data (weekly sales, ad spend per channel, prices, promotions, distribution, etc.), decompose demand into per-component contributions:
- How much of each week's sales came from Facebook? Google? Email?
- How much is organic baseline? Price effect? Competition?

## Why Not Traditional MMM?

- PyMC/Meridian: 2-8 hours per client (MCMC sampling)
- Robyn: Faster but still parametric fitting per client
- Our approach: Neural network trained on 50K simulated businesses → instant category-level inference + fast per-scenario optimization for channels

## Tech Stack

- **Simulator**: `demantiq/core/demand_kernel.py` — 15-step demand generation pipeline
- **Training data**: `demantiq/orchestration/training_pipeline.py` — generates .npz batches
- **Neural engine**: `demantiq/neural/` — encoder-decoder architecture
- **Evaluation**: `scripts/train_neural.py` — training + evaluation with metrics
- **GPU training**: `scripts/vast_train.py` — vast.ai integration
- **Scenarios**: `demantiq/scenarios/` — ScenarioSampler (random) + ScenarioLibrary (fixed benchmarks)

## Current State (April 2026)

- Category decomposition (baseline vs media) works at 2-9pp accuracy
- Channel differentiation does NOT work with any neural approach tried
- Next step: hybrid approach (neural for category + differential evolution for channels)
- Vast.ai GPU workflow is set up and tested ($0.04/hr)
