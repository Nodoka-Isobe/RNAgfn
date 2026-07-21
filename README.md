# RNAgfn

A Python Package for Sampling RNA Secondary Structures with GFlowNets.

RNAgfn is a research-oriented extension of **torchgfn** for RNA secondary structure sampling, including pseudoknotted structures. Please cite our paper if you use RNAgfn in your research. Since this project is built upon and extends the **torchgfn** library, please also cite the original torchgfn paper.

## Installing the package

RNAgfn requires **Python >= 3.10**.

To install the package together with the required dependencies:

```bash
git clone https://github.com/<username>/RNAgfn.git
cd RNAgfn

conda create -n rnagfn python=3.10
conda activate rnagfn

pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

RNAgfn additionally requires **ViennaRNA** for thermodynamic energy calculations.

```bash
conda install -c bioconda viennarna
```

## About this repository

RNAgfn is built upon the **torchgfn** framework and extends it for RNA secondary structure generation.

This repository was developed by cloning the torchgfn codebase and extending its implementation to support RNA-specific applications. In particular, we introduce:

- RNA secondary structure environments
- RNA-specific state and action representations
- Structure validity constraints for RNA folding
- Pseudoknot-compatible structure generation
- Thermodynamic energy-based reward functions
- Temperature annealing strategies
- Customized exploration methods for long RNA sequences
- Training procedures specialized for RNA structure sampling

The goal of RNAgfn is to provide an open and reproducible framework for developing GFlowNet-based RNA secondary structure sampling algorithms. It is intended both as the implementation accompanying our research and as a starting point for future work on RNA generative modeling.

## Getting Started

RNAgfn provides example scripts for training, sampling, and evaluation.

Typical workflow:

1. Prepare an RNA sequence.
2. Train a GFlowNet.
3. Sample RNA secondary structures.
4. Evaluate sampled structures and their distributions.

Example:

```bash
python scripts/train.py \
    --config configs/default.yaml
```

After training:

```bash
python scripts/sample.py \
    --model checkpoints/model.pt \
    --sequence GGGAAACCC
```

The repository also includes example configuration files for reproducing experiments reported in the accompanying paper.

## Components of the Library

### States, Actions, & Containers

RNAgfn represents RNA folding as a sequential decision process.

A state corresponds to a partially constructed RNA secondary structure, while actions modify the current structure by adding valid structural elements according to RNA folding constraints.

The library provides efficient state containers together with forward and backward action masks required for GFlowNet training.

### RNA Environments

RNAgfn introduces RNA-specific environments built on top of torchgfn.

These environments define:

- valid RNA folding states
- pairing and stacking actions
- pseudoknot constraints
- terminal conditions
- reward computation

Users can easily implement new RNA environments by extending the provided interfaces.

### Modules, Estimators, & Samplers

Policy networks are implemented using torchgfn estimators.

RNAgfn provides sampling pipelines for generating RNA secondary structures from trained GFlowNets while supporting customizable neural architectures and sampling strategies.

### Reward Functions

RNAgfn supports thermodynamic reward functions based on RNA free energy.

Current implementations include:

- Boltzmann-based rewards
- Energy scaling
- Reward clipping
- Temperature-dependent rewards

Additional reward functions can be implemented by extending the reward interface.

### Training Objectives

RNAgfn currently supports GFlowNet objectives implemented in torchgfn, including:

- Trajectory Balance (TB)
- Detailed Balance (DB)
- SubTB
- Flow Matching (FM)

RNA-specific training strategies such as delayed logZ optimization and temperature annealing are also provided.

### Exploration Strategies

To improve sampling diversity, RNAgfn includes several exploration techniques, including:

- temperature annealing
- local search
- replay buffers
- exploration noise

These methods are particularly useful for sampling diverse structures from long RNA sequences.

### Defining RNA Environments

One of the primary goals of RNAgfn is to simplify the implementation of new RNA folding environments.

Researchers can define custom environments by specifying:

- state representation
- action space
- transition rules
- reward function
- validity constraints

without modifying the underlying GFlowNet implementation.

### Advanced Usage

RNAgfn is designed as a research platform.

Researchers are encouraged to extend the framework by implementing:

- new RNA reward models
- alternative sampling algorithms
- novel exploration methods
- different GFlowNet objectives
- new RNA structure representations

### Full API Reference

Documentation for all major modules, including environments, samplers, reward functions, training procedures, and utility functions, is provided in the API reference.

## Acknowledgements

RNAgfn is built upon the excellent **torchgfn** library developed by the GFNOrg team.

This repository extends the original framework for RNA secondary structure sampling and would not have been possible without their open-source implementation.

Please cite both the original **torchgfn** work and the accompanying RNAgfn publication when using this repository in your research.