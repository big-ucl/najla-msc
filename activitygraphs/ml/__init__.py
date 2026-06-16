"""
Machine-learning sub-package for activity graph prediction.

This package contains everything needed to train and evaluate graph neural
networks (GNNs) that predict which locations (nodes) a person will visit
during a day, given a travel-survey dataset.

Sub-modules
-----------
models         -- GNN architectures (GATSkip, GraphTransformer, NodeMLP, etc.)
dataset        -- PyTorch-Geometric Dataset classes and data-loading helpers
datamodule     -- LightningDataModule that wraps the dataset pipeline
lightning_module -- LightningModule (training loop, loss, metrics)
baselines      -- Non-learning frequency baselines (Uniform, Global, Node, Conditional)
metrics        -- Custom torchmetrics classes and hop-band distance utilities
experiment     -- High-level ``run_experiment`` / ``evaluate_baseline`` orchestration
callbacks      -- Lightning callbacks for metric collection and debug output
sampling       -- Stochastic location-sampling strategies (Poisson, PPS)
"""
