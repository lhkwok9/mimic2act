# Installation

### Requirements
- Linux machine
- python 3.8.0
- conda
-------

### Create and activate conda environemnt
```
conda create -n mimic2act python=3.8.0
conda activate mimic2act
cd ~
git clone https://github.com/lhkwok9/mimic2act.git
```

### Install robomimic from source
```
cd ~/mimic2act
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic
pip install -e .
```

### Install robosuite from source (simulator)
Note: git checkout is for reproducing experiments
```
cd ~/mimic2act
git clone https://github.com/ARISE-Initiative/robosuite.git
cd robosuite
git checkout v1.4.1
pip install -r requirements.txt
```

### Install ACT
modified from [ACT](https://github.com/tonyzhaozh/act) library by Tony Z. Zhao
```
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco
pip install dm_control
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd ~/mimic2act/act/detr && pip install -e .
```
-------
## Training and Testing

### Extracting Observations from MuJoCo states
[reference link](https://robomimic.github.io/docs/datasets/robosuite.html)

PickPlaceCan data is in ~/mimic2act/data_collection/PickPlaceCan_Aug5_quat_hdf5/demo.hdf5
```
cd ~/mimic2act/robomimic/robomimic/scripts
python dataset_states_to_obs.py --dataset ~/mimic2act/data_collection/PickPlaceCan_Aug5_quat_hdf5/demo.hdf5 --output_name image.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 --exclude-next-obs
```
change camera_height to 480 and camera_width to 640 if u want to follow the paper

### Training
```
cd ~/mimic2act/scripts
python trainACT.py --task_name sim_pick_place_can --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 1 --ckpt_dir ~/mimic2act/act_trained_models/PickPlaceCan_Aug5
```
change dataset_dir, camera_names in sim_pick_place_can in act/constants.py if u want to use another dataset or cameras

### Validation
Download [model](https://drive.google.com/file/d/1sb9ir9Hwjlaw7lBqv5lcpiei_2rMKzWB/view?usp=drive_link) to ~/mimic2act
```
cd ~/mimic2act/scripts
python trainACT.py --task_name sim_pick_place_can --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 1 --ckpt_dir ~/mimic2act/act_trained_models/PickPlaceCan_Aug5 --eval 
```
Add --temporal_agg for temporal aggregation of actions

If u change the task, Find a trained robomimic model in the same environment (e.g. PickPlaceCan with image obs) and add its path to line 226 in trainACT.py(TODO: make this step more convinient)
check line 109 ckpt_names for the models' name to be tested