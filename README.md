# CS336-2026

Personal self-study repository for **Stanford CS336: Language Modeling from Scratch, Spring 2026**.

This repository contains course materials, assignment starter code, my implementations, experiment records, and study notes.

## Repository Structure

```text
CS336-2026/
├── assignments/
│   ├── a1-basics/
│   ├── a2-systems/
│   ├── a3-scaling/
│   ├── a4-data/
│   └── a5-alignment/
├── lectures/
├── notes/
├── .gitignore
└── README.md
```

- `assignments/`: assignment starter code and personal implementations
- `lectures/`: lecture slides, webpages, images, and related materials
- `notes/`: study notes and implementation records

## Progress

### Assignment 1: Basics

- [x] Development environment setup
- [x] Unicode and UTF-8 exercises
- [x] GPT-2 regex pre-tokenization
- [x] BPE training implementation
- [x] BPE training correctness tests
- [x] BPE special-token tests
- [x] BPE training speed test
- [x] Tokenizer implementation
- [x] Transformer components
- [x] Language-model training
- [x] Profiling and benchmarking

### Assignment 2: Systems

- [x] Profiling and benchmarking
- [x] Single-GPU memory
- [x] GPU-Kernels
- [x] Distributed data parallel Training
- [x] Optimization State Sharding
- [x] Fully shared Data Parallel
- [x] Analyzing Parallelism Strategies

### Assignment 3: Scaling

- [ ] Not started

### Assignment 4: Data

- [ ] Not started

### Assignment 5: Alignment

- [ ] Not started

## Environment

Each assignment is an independent Python project with its own dependencies and configuration.

The assignments use [`uv`](https://docs.astral.sh/uv/) for environment and dependency management.

Example:

```bash
cd assignments/a1-basics
uv sync
uv run pytest
```

Run an individual test file:

```bash
uv run pytest tests/test_train_bpe.py -vv
```

## Assignment 1: BPE Training

The current BPE trainer implements:

- byte-level vocabulary initialization
- UTF-8 byte encoding
- GPT-2-style regex pre-tokenization
- special-token boundary handling
- weighted adjacent-pair frequency counting
- deterministic tie-breaking
- iterative non-overlapping pair merging
- vocabulary and merge-list generation

Current public test result:

```text
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED
```

## Notes

Detailed learning and implementation records are stored under `notes/`.

These notes focus on:

- problem decomposition
- relevant data structures
- implementation decisions
- debugging processes
- test interpretation
- conceptual understanding

## Repository Policy

This repository is maintained for personal study and progress tracking.

Original course materials and assignment specifications belong to their respective authors and Stanford University. This repository is not an official Stanford course repository.

The implementation code represents my own coursework and should not be copied or submitted as another student's work.