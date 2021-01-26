C-LEARNING: HORIZON-AWARE CUMULATIVE ACCESSIBILITY ESTIMATION
========================================================

This repository provides the implementation of [C-learning](https://arxiv.org/pdf/2011.12363.pdf). 


# Example Script

In order to train the models run the scripts in the experiment folder:
```
python experiment/env_name -s 21
```
The trained models will be saved in runs/env_name directory.


To train the model for other environments you should write similar scripts.

# Simulate the results

For visualizing and saving the simulations:

```
python evaluate/sim.py FetchPickAndPlace-v1
python evaluate/sim.py HandManipulatePen-v0
```

# Learning curves and evaluation

To plot the learning curves:
```
python evaluate/training_curve.py env_name
```
For final evaluations on the given metrics:
```
python evaluate/eval.py env_name
```
