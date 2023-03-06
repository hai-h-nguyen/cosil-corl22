# Leveraging Fully Observable Policies for Learning under Partial Observability

This is the repo stored the code for our paper [Leveraging Fully Observable Policies for Learning under Partial Observability](https://openreview.net/pdf?id=pn-HOPBioUE) accepted at CoRL 2022.

```
@article{nguyen2022leveraging,
  title={Leveraging Fully Observable Policies for Learning under Partial Observability},
  author={Nguyen, Hai and Baisero, Andrea and Wang, Dian and Amato, Christopher and Platt, Robert},
  journal={arXiv preprint arXiv:2211.01991},
  year={2022}
}
```
---
## Contents

[Setup](#setup)

[Domains](#domain)

[Train](#train)

[License, Acknowledgments](#license)

---

## Setup
1. Install [anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)
1. Create and activate environment, install required packages
```
conda create --env cosil python=3.8.5
conda activate cosil
```
1. Clone this repository
```
git clone --recursive https://github.com/hai-h-nguyen/cosil-corl22.git
pip install -r requirements.txt
```
1. Install domains
```
cd pomdp_robot_domains
pip install -r requirements.txt
pip install -e .
cd ..
cd pomdp-domains
pip install -e .
cd ..
```
1. Install [Pytorch](https://pytorch.org/get-started/previous-versions/) (I used 1.12.0 for cuda 10.2 but other versions should work)
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=10.2 -c pytorch
```
---

## Train

### Before Training
```export PYTHONPATH=${PWD}:$PYTHONPATH```

### Bumps-1D (Discrete Action)
* COSIL (**sacde**) / Behavior-Cloning (**bcd**) / Recurrent SAC (**sacd**) / Offpolicy-Advisor (**sacda**)

```
python3 policies/main.py --cfg configs/pomdp/bumps_1d/rnn.yml --algo sacde --target_entropy 1.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/bumps_1d/rnn.yml --algo sacda --target_entropy 0.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/bumps_1d/rnn.yml --algo bcd --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/bumps_1d/rnn.yml --algo sacd --target_entropy 0.7 --seed 0 --cuda 0
```

### Bumps-2D (Discrete Action)
* COSIL (**sacde**) / Behavior-Cloning (**bcd**) / Recurrent SAC (**sacd**) / Offpolicy-Advisor (**sacda**)

```
python3 policies/main.py --cfg configs/pomdp/bumps_2d/rnn.yml --algo sacde --target_entropy 1.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/bumps_2d/rnn.yml --algo sacda --target_entropy 0.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/bumps_2d/rnn.yml --algo bcd --seed 0 --cuda 0
python3 policies/main.py --cfg configs/mdp/bumps_2d/rnn.yml --algo sacd --target_entropy 0.7 --seed 0 --cuda 0
```

### LunarLander-P, -V (Continuous Action)

* COSIL (**sace**) / Behavior-Cloning (**bc**) / Recurrent SAC (**sac**) / Offpolicy-Advisor (**saca**)

```
python3 policies/main.py --cfg configs/pomdp/lunarlander/rnn_p(rnn_v).yml --algo sace --target_entropy -1.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/lunarlander/rnn_p(rnn_v).yml --algo bc/saca/sac --seed 0 --cuda 0
```

### CarFlag (Continuous Action)
* COSIL (**sace**) / Behavior-Cloning (**bc**) / Recurrent SAC (**sac**) / Offpolicy-Advisor (**saca**)
```
python3 policies/main.py --cfg configs/pomdp/car_flag_continuous/rnn.yml --algo sace --target_entropy -1.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/car_flag_continuous/rnn.yml --algo sac/saca/bc --seed 0 --cuda 0
```

### Block-Picking (Continuous Action)
* COSIL (**sace**) / Behavior-Cloning (**bc**) / Recurrent SAC (**sac**) / Offpolicy-Advisor (**saca**)
```
python3 policies/main.py --cfg configs/pomdp/blockpicking/rnn.yml --algo sace --target_entropy 0.0 --seed 0 --cuda 0
python3 policies/main.py --cfg configs/pomdp/blockpicking/rnn.yml --algo sac/saca/bc --seed 0 --cuda 0
```

---
## Visualization using Tensorboard
```
tensorboard --logdir logs/folder_to_plot
```

---

## License

This code is released under the MIT License.

---

## Acknowledgments

This codebase evolved from the [pomdp-baselines](https://github.com/twni2016/pomdp-baselines).
