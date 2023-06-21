# Neural system identification and control for Formula Student Driverless cars

This repository contains the [report](OANCEA_TUDOR_DPC_project_report.pdf), 
[final presentation](MATH_591_final_presentation.pdf) and code for my first semester 
project in my Computational Science and Engieering master's degree at EPFL.

The code contains the model definitions, training and evaluation scripts for the
neural networks used in the project. Everything is written in Python 3.10 and is 
based on Pytorch, Lightning Fabric and Wandb. It was designed to be very clear (although 
poorly documented), modular and extensible, and can be used for other projects
as well. The data is available in the latest release of the repository.

<!-- ## Project description

## Project results -->

## Codebase

### Workspace setup

To setup the workspace you can run the following commands:
```bash
git clone https://github.com/tudoroancea/math_591_project
cd math_591_project
mamba env create --file env.yml # or conda env create --file env.yml if you don't use mamba, but it's a shame not to use it since it's simply much MUCH faster
conda activate math_591_project
wget https://github.com/tudoroancea/math_591_project/releases/download/untagged-d3fb9058dc258922e9bc/dataset_v2.0.0.zip
unzip dataset_v2.0.0.zip -d .
```
Then you can use the training scripts for the [system identification](train_sysid.py) and [control](train_control.py) tasks,
and finally obtain again all the plots in my report using the evaluation scripts for the [system identification](sysid_experiments.py) and [control](control_experiments.py).