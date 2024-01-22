# A kinematic model for HSA robots

This repository contains code for verifying the Selective Piecewise Constant Strain (SPCS) kinematic model for representing
the shape of the rods making up HSA robots.
This model allows us to keep the twist and axial strains constant throughout the robot, while the bending and shear strains 
vary piecewise. We refer to the [publication](##Citation) for more details.

A JAX implementation of SPCS can be found [here](https://github.com/tud-cor-sr/jax-spcs-kinematics) and is used in the scripts of this repository.

## Citation

This repository is part of the publication **Modelling Handed Shearing Auxetics:
Selective Piecewise Constant Strain Kinematics and Dynamic Simulation** presented at the 
_6th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2023)_. 
You can find the publication online on [IEEE Xplore](https://ieeexplore.ieee.org/abstract/document/10121989/).

Please use the following citation if you use our method in your (scientific) work:

```bibtex
@inproceedings{stolzle2023modelling,
  title={Modelling Handed Shearing Auxetics: Selective Piecewise Constant Strain Kinematics and Dynamic Simulation},
  author={St{\"o}lzle, Maximilian and Chin, Lillian and Truby, Ryan L. and Rus, Daniela and Della Santina, Cosimo},
  booktitle={2023 IEEE 6th International Conference on Soft Robotics (RoboSoft)},
  year={2023},
  organization={IEEE}
}
```

## Installation

All necessary dependencies can be installed using `pip`:

```bash
pip install -r requirements.txt
```

## See also

You might also be interested in the following repositories:

 - The [`jax-spcs-kinematics`](https://github.com/tud-phi/jax-spcs-kinematics) repository contains an implementation
 of the Selective Piecewise Constant Strain (SPCS) kinematics in JAX. We have shown in our paper that this kinematic 
model is suitable for representing the shape of HSA rods.
 - The [`HSA-PyElastica`](https://github.com/tud-phi/HSA-PyElastica) repository contains a plugin for PyElastica
for the simulation of HSA robots.
