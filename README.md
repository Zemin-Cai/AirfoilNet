# AirfoilNet
This repository contains the source code for our paper:
[A Neural Network Model for Predicting Aerodynamic
Parameters of Airfoils](https://arxiv.org/html/2403.14979v1).
# Requirement
$\cdot$ python 3.10.11
$\cdot$ torch 1.13.1
$\cdot$ torchvision 0.14.1
# Run AirfoilNet
## Training
You can run ``AirfoilNet/train.py`` to train AirfoilNet. We prepare the datasets in `AirfoilNet/data`. For the config of predictor of the AirfoilNet, you can revise the parameter of prediction_head.

## Evaluate
Run `evaluate.py`. 

# Run DDAM
## Training
Run `train.py` to train a DDAM model. The datasets is prepared in the `AifoilNet/data` for training DDAM.

## Generation
Run `generate_data.py` to generate data based on original training data.

# Acknowledgement
Thanks to AlinaNeh for the [public data](https://splinecloud.com/repository/AlinaNeh/NACA_airfoil_database/) .  Our DDAM code references this [code](https://github.com/Alokia/diffusion-DDIM-pytorch). 
