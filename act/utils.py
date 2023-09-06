import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
        super(EpisodicDataset).__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = True
        # self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False # hardcode

        episode_id = self.episode_ids[index] + 1
        # dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        dataset_path = self.dataset_dir
        with h5py.File(dataset_path, 'r') as root:
            # is_sim = root.attrs['sim']
            # is_sim = True
            original_action_shape = root['/data/demo_'+str(episode_id)+'/actions'].shape
            episode_len = original_action_shape[0]
            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)
            # get observation at start_ts only
            eefpos = root['/data/demo_'+str(episode_id)+'/obs/robot0_eef_pos'][start_ts]
            eefquat = root['/data/demo_'+str(episode_id)+'/obs/robot0_eef_quat'][start_ts]
            qpos = np.append(eefpos, eefquat, axis=0)
            # print(f"{np.append(eefpos, eefquat, axis=1).shape=}")
            gripper_qpos = root['/data/demo_'+str(episode_id)+'/obs/robot0_gripper_qpos'][start_ts]
            qpos = np.append(qpos, gripper_qpos, axis=0)
            # print(f"{qpos}")
            # qpos = root['/observations/qpos'][start_ts]
            # qvel = root['/observations/qvel'][start_ts]
            image_dict = dict()
            for cam_name in self.camera_names:
                # image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]
                image_dict[cam_name] = root['/data/demo_'+str(episode_id)+f'/obs/{cam_name}'][start_ts]
            # get all actions after and including start_ts
            if self.is_sim:
                action = root['/data/demo_'+str(episode_id)+'/actions'][start_ts:]
                # eefpos = root['/data/demo_'+str(episode_id)+'/obs/robot0_eef_pos'][start_ts:]
                # print(f"{eefpos.shape=}")
                # eefquat = root['/data/demo_'+str(episode_id)+'/obs/robot0_eef_quat'][start_ts:]
                # print(f"{eefquat.shape=}")
                # action = np.append(eefpos, eefquat, axis=1)
                # print(f"{action.shape=}")
                # gripper_qpos = root['/data/demo_'+str(episode_id)+'/obs/robot0_gripper_qpos'][start_ts:]
                # action = np.append(action, gripper_qpos, axis=1)
                # print(f"{action.shape=}")
                action_len = episode_len - start_ts
            else:
                action = root['/data/demo_'+str(episode_id)+'/actions'][max(0, start_ts - 1):] # hack, to make timesteps more aligned
                action_len = episode_len - max(0, start_ts - 1) # hack, to make timesteps more aligned

        # self.is_sim = is_sim
        # padded_action = np.zeros(original_action_shape, dtype=np.float32)
        padded_action = np.zeros((200, action.shape[1]), dtype=np.float32)
        padded_action[:action_len] = action
        is_pad = np.zeros(200)
        is_pad[action_len:] = 1

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = torch.from_numpy(all_cam_images)
        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        # action_data = torch.from_numpy(padded_action).double()
        # print(f"{action_data.dtype=}")
        is_pad = torch.from_numpy(is_pad).bool()

        # channel last
        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # normalize image and change dtype to float
        image_data = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        # action_data = (action_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

        # print(f"{image_data.shape}")
        # print(f"{action_data.shape}")
        # print(f"{qpos_data.shape}")
        # print(f"{is_pad.shape}")

        return image_data, qpos_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes):
    # all_qpos_data = []
    # all_action_data = []
    with h5py.File(dataset_dir, 'r') as root:
        for episode_idx in range(num_episodes):
            eefpos = root['/data/demo_'+str(episode_idx+1)+'/obs/robot0_eef_pos'][()]
            eefquat = root['/data/demo_'+str(episode_idx+1)+'/obs/robot0_eef_quat'][()]
            qpos = np.append(eefpos, eefquat, axis=1)
            # print(f"{np.append(eefpos, eefquat, axis=1).shape=}")
            gripper_qpos = root['/data/demo_'+str(episode_idx+1)+'/obs/robot0_gripper_qpos'][()]
            qpos = np.append(qpos, gripper_qpos, axis=1)
            # print(f"{qpos.shape=}")
            action = root['/data/demo_'+str(episode_idx+1)+'/actions'][()]
            # print(f"{action.shape=}")
            # all_qpos_data.append(torch.from_numpy(qpos))
            # all_action_data.append(torch.from_numpy(action))
            if episode_idx == 0:
                all_qpos_data = qpos
                all_action_data = action
            else:
                all_qpos_data = np.append(all_qpos_data, qpos, axis=0)
                all_action_data = np.append(all_action_data, action, axis=0)
            # print(f"{all_qpos_data.shape=}")
    # print(f"{all_qpos_data.shape=}")
    # all_qpos_data = torch.stack(all_qpos_data)
    # print(f"{all_qpos_data.shape=}")
    # all_action_data = torch.stack(all_action_data)
    all_qpos_data = torch.asarray(all_qpos_data)
    all_action_data = torch.asarray(all_action_data)
    # print(f"{all_qpos_data=}")

    # normalize qpos data
    # qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
    # qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
    qpos_mean = all_qpos_data.mean(0, keepdim=True)
    qpos_std = all_qpos_data.std(0, keepdim=True)
    # print(f"{qpos_std=}")
    qpos_std = torch.clip(qpos_std, 1e-4, 1) # clipping

    # normalize action data
    # action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
    # action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_mean = all_action_data.mean(0, keepdim=True)
    action_std = all_action_data.std(0, keepdim=True)
    # print(f"{action_std=}")
    action_std = torch.clip(action_std, 1e-4, 1) # clipping

    stats = {
             "action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             "qpos_mean": qpos_mean.numpy().squeeze(), "qpos_std": qpos_std.numpy().squeeze(),
             "example_qpos": qpos}

    return stats


def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.9
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
