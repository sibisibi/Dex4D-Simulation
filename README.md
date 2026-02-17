# Dex4D-Simulation

This is the codebase for the simulation (AP2AP) part of the Dex4D project.


https://github.com/user-attachments/assets/0c43135a-5d40-4ba5-908a-4de3bdafa851


## Installation

The code is tested on Python 3.8.20 with cuda toolkit 11.8.

Please follow the steps below to perform the installation:


### 1. Create virtual environment

```bash
conda create -n dex4d-sim python==3.8.20
conda activate dex4d-sim
```

### 2. Install Isaac Gym

First install PyTorch:
```bash
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
```

Then install pytorch3d:
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable" 
```

Then download Isaac Gym from the [official source](https://developer.nvidia.com/isaac-gym). After downloading and unzipping, install Isaac Gym in editable mode:

```bash
cd <PATH_TO_ISAACGYM_INSTALL_DIR>/python
pip install -e .
```

### 3. Install repo from source code

Once Isaac Gym is installed and samples work within your current python environment, install this repo from source code:

```bash
cd dex4d_policy/
pip install -e .
```

### 4. Install pointnet2_ops

```bash
git clone git@github.com:erikwijmans/Pointnet2_PyTorch.git && cd Pointnet2_PyTorch/pointnet2_ops_lib/
python setup.py install
```


## Dataset

Please follow [UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp) to set up the object dataset in `dex4d_policy/assets` folder.

The `dex4d_policy/assets` folder should look like:
```
dex4d_policy/assets
├── datasetv4.1
├── meshdatav3_pc_feat
├── meshdatav3_scaled
├── mjcf
├── textures
└── urdf
```

## Training

All the training commands are executed from the `dex4d_policy/dex4d` folder.

### Teacher PPO Policy Stage 1 & 2

First fill in the `<YOUR_LOG_DIR>` in `script/run_train_ppo_stage_1_2.sh`.

```bash
# override config for stage 1 & 2 training
cp cfg/xarm6_leap_hand_ap2ap_stage_1_2.yaml cfg/xarm6_leap_hand_ap2ap.yaml
cp cfg/ppo/config_stage_1_2.yaml cfg/ppo/config.yaml
# launch training
bash script/run_train_ppo_stage_1_2.sh
```

### Teacher PPO Policy Stage 3

First fill in the `<YOUR_LOG_DIR>` and `<CKPT_NAME_FROM_STAGE_1_2>` in `script/run_train_ppo_stage_3.sh`.

```bash
# override config for stage 3 training
cp cfg/xarm6_leap_hand_ap2ap_stage_3.yaml cfg/xarm6_leap_hand_ap2ap.yaml
cp cfg/ppo/config_stage_3.yaml cfg/ppo/config.yaml
# launch training
bash script/run_train_ppo_stage_3.sh
```

### Student DAgger Policy

First fill in the `<YOUR_LOG_DIR>` and `<CKPT_NAME_FROM_STAGE_3>` in `script/run_train_dagger.sh`.

```bash
bash script/run_train_dagger.sh
```

For more provided args, please check these scripts and `utils/config.py`.

## Playing with trained models

All the playing commands are executed from the `dex4d_policy/dex4d` folder, and pretrained models are provided in `dex4d_policy/dex4d/example_models`.

```bash
# First fill in <CKPT_NAME> in script/play.sh and script/play_dagger.sh, then run:
bash script/play.sh # Teacher PPO policy
bash script/play_dagger.sh # Student DAgger policy
```


## Acknowledgement

This project is built upon the following open-source projects:

- [PKU-EPIC/UniDexGrasp](https://github.com/PKU-EPIC/UniDexGrasp)
- [NVIDIA-Omniverse/IsaacGymEnvs](https://github.com/NVIDIA-Omniverse/IsaacGymEnvs)
- [PKU-MARL/DexterousHands](https://github.com/PKU-MARL/DexterousHands)

We thank the authors for their open-source contributions.
