# BloomGoal for ALFworld

This directory contains the implementation and training scripts for BloomGoal based on ALFWorld.

## Installation Guide

### 1. Install ALFWorld and its dependencies

```bash
git clone https://github.com/alfworld/alfworld.git
cd alfworld
pip install setuptools==63.2.0
pip install --no-build-isolation
pip install -e .[full]
```

### 2. Install dependencies for this project

```bash
pip install -r requirement.txt
```


## Project Structure

- bloom_dist.py: Main implementation of the BloomGoal algorithm.
- train.py: Basic training pipeline script.
- train_max.py: Long-horizon training script that continues training and logs statistics at intervals.
- base_config.yaml: Configuration for the ALFWorld environment and training.
- prompts/alfworld_3prompts.json: Example prompt data.

## Usage Instructions

Run the following command to execute the BloomGoal method:
```bash
python bloom_dist.py
```
This will generate bloom_dist_progress.json to record runtime results and preserve progress if the program is interrupted.It will also generate bloom_memory.json, which serves as the persistent memory for the Bloom method.
Run the following command to train BloomGoal:
```bash
python train.py
```
This will generate the bloom_memory.json memory file. You can run bloom_dist.py again for testing after training.
