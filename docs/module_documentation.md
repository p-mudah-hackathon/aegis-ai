# Module Documentation

## File: core\config.py
AegisNode Shared Configuration
================================
Single source of truth for directory paths and default hyperparameters.
All modules import paths and defaults from here.

## File: core\data\constants.py
AegisNode Data Constants
=========================
Shared constants for dataset generation, preprocessing, and demo simulations.
Includes issuers, MCC codes, locations, temporal weights, and utility functions.

## File: core\data\generator.py
AegisNode Synthetic QRIS Cross-Border Transaction Dataset Generator
===================================================================
Generates PaySim-grade synthetic data for HTGNN fraud detection training.

Graph Definition for HTGNN:
  Node Types:
    - payer   (payer_token_id)  : Foreign tourist, features: source_issuer, source_country
    - merchant (merchant_nmid)  : Indonesian merchant, features: mcc, location, qris_type

  Edge Type:
    - TRANSACTS (payer -> merchant) : Temporal edge per transaction
      Features: amount_idr, hour_of_day, day_of_week, time_since_last_txn_sec

  Fraud Scenarios Embedded:
    1. velocity_attack    : Rapid transactions across multiple merchants (<20 min)
    2. card_testing       : Many small amounts then one large purchase
    3. collusion_ring     : Coordinated payers hitting same merchant(s)
    4. geo_anomaly        : Same payer in distant cities within impossible time
    5. amount_anomaly     : Extreme amounts at off-hours

Output: aegis_qris_transactions.csv, aegis_graph_meta.json

## File: core\data\preprocessing.py
AegisNode Data Preprocessing — CSV to PyG HeteroData
=====================================================
Converts aegis_qris_transactions.csv into PyTorch Geometric HeteroData
objects with temporal train/val/test splits for HTGNN training.

Graph Structure:
  Node Types:  payer, merchant
  Edge Type:   (payer, TRANSACTS, merchant)
  Edge Label:  is_fraud (binary)

## File: core\data\__init__.py
AegisNode Data — Dataset generation and preprocessing.

## File: core\export\exporter.py
AegisNode Model Export — TorchScript for PAI-EAS
==================================================
Exports the trained model to TorchScript format (.pt) for deployment.
Also saves inference configuration (label encoders, normalization params).

Usage:
  python scripts/export_model.py

## File: core\export\__init__.py
AegisNode Export — Model export for PAI-EAS deployment.

## File: core\model\htgnn.py
AegisNode HTGNN Model — Heterogeneous Temporal Graph Neural Network
====================================================================
Architecture:
  - Node encoders (embedding + linear for payer & merchant)
  - Temporal encoder (Bochner time encoding)
  - 3x HGTConv layers with residual connections
  - Edge-level MLP classifier for fraud detection

Graph structure:
  Nodes: payer, merchant
  Edges: (payer, TRANSACTS, merchant) — temporal edges

## File: core\model\losses.py
AegisNode Losses — Focal Loss and Weighted BCE for class imbalance
====================================================================
Focal Loss (Lin et al., 2017) down-weights easy negatives and focuses
learning on hard-to-classify fraud samples.
WeightedBCELoss uses pos_weight for stable training with extreme imbalance.

## File: core\model\__init__.py
AegisNode Model — HTGNN architecture and loss functions.

## File: core\training\evaluator.py
AegisNode Evaluation — Post-training test set evaluation
=========================================================
Loads the best checkpoint and evaluates on the test set with:
  - Overall metrics (AUPRC, AUROC, F1)
  - Per-fraud-type recall breakdown
  - Confusion matrix at optimal threshold
  - Classification report

Usage:
  python scripts/evaluate.py

## File: core\training\explainer.py
AegisNode XAI — GNN Explainability for Paylabs Dashboard
=========================================================
Generates per-transaction explanations for flagged fraud transactions:
  - Feature importance rankings
  - Related transaction subgraph
  - JSON output format for frontend consumption

Usage:
  python scripts/explain.py                     # Explain all flagged transactions
  python scripts/explain.py --top-k 10          # Explain top 10 highest-scoring
  python scripts/explain.py --txn-id TX-000104  # Explain specific transaction

## File: core\training\trainer.py
AegisNode Training Pipeline
============================
Trains the AegisHTGNN model on the preprocessed HeteroData.

Usage:
  python scripts/train.py                  # Full training (200 epochs)
  python scripts/train.py --epochs 3       # Smoke test
  python scripts/train.py --no-save        # Don't save checkpoints

## File: scripts\demo_attack.py
AegisNode — Live Attack Demo Simulator
========================================
Simulates 1,000 foreign tourist QRIS transactions hitting the Paylabs API.
Mix of legitimate and fraudulent transactions across all 5 attack vectors.

Usage:
  python scripts/demo_attack.py                # Full 1,000 txn demo
  python scripts/demo_attack.py --speed fast   # Faster output (no delay)
  python scripts/demo_attack.py --total 100    # Smaller demo

Demo Flow for Hackathon:
  1. Script simulates inbound cross-border QRIS transactions
  2. Each transaction is scored by the HTGNN model in real-time
  3. Normal transactions → APPROVED (green)
  4. Fraud transactions → FLAGGED (red) with XAI explanation
  5. Summary statistics at the end

## File: scripts\eda_validation.py
AegisNode Dataset — Comprehensive EDA & PaySim Benchmark Comparison
===================================================================
Validates dataset quality against PaySim and real-world fraud detection standards.

Usage:
  python scripts/eda_validation.py

## File: scripts\evaluate.py
Evaluate the trained model on val/test sets.

## File: scripts\explain.py
Generate XAI explanations for flagged transactions.

## File: scripts\export_model.py
Export trained model for PAI-EAS deployment.

## File: scripts\generate_dataset.py
Generate the AegisNode synthetic QRIS transaction dataset.

## File: scripts\preprocess.py
Preprocess raw CSV into PyG HeteroData for HTGNN training.

## File: scripts\train.py
Train the AegisHTGNN model.

