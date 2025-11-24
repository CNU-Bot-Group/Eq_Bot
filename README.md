# Eq.Bot: A Universal SE(2) Equivariant Framework for Robotic Manipulation

Eq.Bot is a model-agnostic solution that can be integrated with existing multi-modal architectures (both CNN-based and Transformer-based) without requiring architectural modifications. It works by transforming observations into a canonical space, applying an existing policy, and mapping the resulting actions back to the original space.

## Core Code Structure

The core logic of the Eq.Bot framework is organized as follows:

- `canonicalization/`: This directory contains the foundational modules for performing the canonicalization and inverse transformation operations.

  - `basecanonicalization.py`: Defines the abstract base class for the canonicalization process.
  - `discrete_group.py`: Implements the discrete SE(2) group operations (e.g., rotations and translations) used to transform observations and actions.
  - `utils.py`: Contains utility functions supporting the transformation and canonicalization process.
- `canonicalization_networks/`: This directory provides the network architectures used to estimate the canonical transformation from input observations. As discussed in our paper, we support different network designs.

  - `equivariant_networks.py`: Implements group equivariant network architectures for the canonicalization module. These networks are designed with built-in symmetries to improve sample efficiency and generalization.
  - `nonequivariant_networks.py`: Implements standard (non-equivariant) CNN architectures that can be used within the canonicalization framework to estimate the transformation parameters.

## How It Works

The Eq.Bot framework operates in three main stages:

1. **Canonicalization**: An input observation (e.g., an image) is passed through a **Group Equivariant Canonicalization Network** (from `canonicalization_networks/`) to estimate a transformation $g$. The observation is then transformed into a canonical orientation.
2. **Action Prediction**: The canonicalized observation is fed into a pre-existing, unmodified base policy (e.g., CLIPort, OpenVLA-OFT) to predict an action in the canonical coordinate system.
3. **Inverse Transformation**: The predicted action is mapped back from the canonical space to the original observation's coordinate system using the inverse of the estimated transformation $g^{-1}$. The modules in `canonicalization/` are used for this step.

This "wrap-around" approach allows us to add a strong inductive bias for spatial equivariance to a wide range of models without needing to alter their internal architecture.

## Usage

This code provides the core building blocks for the Eq.Bot framework. To integrate it with a robotic manipulation pipeline, you would:

1. Instantiate a canonicalization network from `canonicalization_networks/`.
2. Wrap your existing policy with the canonicalization and inverse transformation logic provided in `canonicalization/`.
3. Train or finetune the canonicalization network alongside the base policy.

For detailed experimental setup and results, please refer to our main paper.
