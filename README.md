# Ensembling Language Models with Sequential Monte Carlo (NeurIPS 2026)

[![arXiv](https://img.shields.io/badge/arXiv-2603.05432-b31b1b.svg)](https://arxiv.org/abs/2603.05432)

This repository contains the code for the experiments in the paper *Ensembling Language Models with Sequential Monte Carlo* (NeurIPS 2026).

The reference implementation of the ensembling algorithm can be found in the actively maintained [genlm/genlm-control](https://github.com/genlm/genlm-control) library.

This repo contains the prompts, scorers and scripts to replicate the paper experiments based on the [genlm/genlm-control](https://github.com/genlm/genlm-control) and [genlm/genlm-eval](https://github.com/genlm/genlm-eval) libraries.

## Setup

### Requirements

- Python >=3.11
- A GPU (two for cross-model ensembles)
- The dependencies in `pyproject.toml`

### Installation

Clone this repository, including the Spider evaluator submodule:
```bash
git clone --recurse-submodules https://github.com/genlm/ensemble-neurips-2026.git
cd ensemble-neurips-2026
```

Install the dependencies:
```bash
pip install -e ".[vllm]"
```

Note: It is recommended to use a virtual environment to manage the dependencies. For example, with conda:

```bash
conda create -n ensemble python=3.11
conda activate ensemble
pip install -e ".[vllm]"
```

### Download required datasets

All datasets are downloaded from the Hugging Face Hub when first used, except for Spider.

#### Text to SQL

Download and unzip the Spider dataset in the `data/spider` directory with the following commands:
```bash
mkdir -p data/spider
cd data/spider
gdown 'https://drive.google.com/u/0/uc?id=1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J&export=download'
unzip spider_data.zip
cd ../..
export SPIDER_DIR=$PWD/data/spider/spider_data
```

## Running the experiments

All ensembles are run through `ensemble_bench.smc`. The `--task` argument is one of `bbh`, `spider`, `json_schema`, `gsm8k` or `mmlu`, and the `--level` argument is `token` or `byte`.

To run a within-model ensemble of the two prompts used in the paper for a task:
```bash
python -m ensemble_bench.smc --task bbh --level token --model-1 meta-llama/Llama-3.1-8B-Instruct
```

To run a cross-model ensemble with the same prompt for both models:
```bash
python -m ensemble_bench.smc --task spider --level byte \
    --model-1 meta-llama/Llama-3.1-8B-Instruct --model-2 Qwen/Qwen2.5-7B-Instruct \
    --template-1 mid_1 --template-2 mid_1
```

The defaults are the settings used in the paper: 10 particles, a resampling threshold of 0.9, equal expert weights and 100 examples. See `python -m ensemble_bench.smc --help` for all arguments.

The base models are run through `ensemble_bench.base`:
```bash
python -m ensemble_bench.base --task bbh --model meta-llama/Llama-3.1-8B-Instruct
```

See the `scripts` directory for the Slurm scripts which run the experiments from the paper, and for the scripts which aggregate their output into the paper's tables.

### Output saving

The output is saved in the directory provided through the `--out-dir` argument. If not provided, the default is the current directory. The files are named as follows:

- `*-smc.jsonl`: One file per operator, with one line per example recording the sampled output, its correctness and all particles with their weights.
- `*-metrics.jsonl`: One file per run, with the expected accuracy and its bootstrap confidence interval for each operator.

## Repository structure

```
ensemble-neurips-2026/
├── ensemble_bench/           # Core experimental code
│   ├── smc.py                # Main CLI entry point for running ensembles
│   ├── base.py               # CLI entry point for running the base models
│   ├── baselines.py          # Implementation of the UniTE and GaC baselines
│   ├── experts.py            # Ensembles of more than two experts
│   ├── models.py             # Model loading for the vLLM and Hugging Face backends
│   ├── tasks/                # Task definitions, prompts and scorers
│   └── spider/               # Spider prompt formatting and evaluation
├── scripts/                  # Experiment execution and result aggregation scripts
├── notebooks/                # Prompt intersection example (Figure 1)
├── tests/                    # Test suite
└── third_party/spider/       # Official Spider evaluator (git submodule)
```

## Citation

```bibtex
@inproceedings{chan2026ensembling,
    title = {Ensembling Language Models with Sequential Monte Carlo},
    author = {Chan, Robin S. M. and Liu, Tianyu and Kiegeland, Samuel and Pasti, Clemente and Vigly, Jacob Hoover and O'Donnell, Timothy J. and Cotterell, Ryan and Vieira, Tim},
    booktitle = {Advances in Neural Information Processing Systems},
    year = {2026}
}
```
