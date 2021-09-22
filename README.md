<p align="center">
<a href="https://layer6.ai/"><img src="https://github.com/layer6ai-labs/DropoutNet/blob/master/logs/logobox.jpg" width="180"></a>
</p>

## C-Learning: Horizon-Aware Cumulative Accessibility Estimation

Authors: Panteha Naderian, Gabriel Loaiza-Ganem, Harry J. Braviner, Anthony L. Caterini, Jesse C. Cresswell, Tong Li, Animesh Garg
[[paper](https://openreview.net/pdf?id=W3Wf_wKmqm9)]


<a name="Environment"/>

## Environment:

The code was developed and tested on the following python environment:
```
python 3.7.4
pytorch 1.4.0
numpy 1.17.2
gym 0.15.7
mujoco 1.5
```
<a name="instructions"/>

## Instructions:

1. In order to train the models run the scripts in the experiment folder:
```
python experiment/env_name -s 21
```
The trained models will be saved in runs/env_name directory.


To train the model for other environments you should write similar scripts.

2. For visualizing and saving the simulations:

```
python evaluate/sim.py FetchPickAndPlace-v1
python evaluate/sim.py HandManipulatePen-v0
```

3. To plot the learning curves:
```
python evaluate/training_curve.py env_name
```

4. For final evaluations on the given metrics:
```
python evaluate/eval.py env_name
```

<a name="citation"/>

## Citation

If you find this code useful in your research, please cite the following paper:

    @inproceedings{naderian2020c,
  title={C-Learning: Horizon-Aware Cumulative Accessibility Estimation},
  author={Naderian, Panteha and Loaiza-Ganem, Gabriel and Braviner, Harry J and Caterini, Anthony L and Cresswell, Jesse C and Li, Tong and Garg, Animesh},
  booktitle={International Conference on Learning Representations},
  year={2020}
}

    
