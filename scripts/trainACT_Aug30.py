"""
The main entry point for training policies.

Args:
    config (str): path to a config json that will be used to override the default settings.
        If omitted, default settings are used. This is the preferred way to run experiments.

    algo (str): name of the algorithm to run. Only needs to be provided if @config is not
        provided.

    name (str): if provided, override the experiment name defined in the config

    dataset (str): if provided, override the dataset path defined in the config

    debug (bool): set this flag to run a quick training run for debugging purposes    
"""

# python trainACT.py --config /home/jk/mimic2aloha_Aug30/config/custom/bc_rnn.json --task_name sim_transfer_cube_scripted --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000  --lr 1e-5 --seed 1 --ckpt_dir /home/jk/mimic2aloha_Aug30/act_trained_models

import argparse
import json
import numpy as np
import time
import os
import shutil
import psutil
import sys
import socket
import traceback

from collections import OrderedDict

import torch
from torch.utils.data import DataLoader

import robomimic
import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.torch_utils as TorchUtils
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory, RolloutPolicy
from robomimic.utils.log_utils import PrintLogger, DataLogger, flush_warnings

# ACT imports
import pickle
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

sys.path.insert(1, '/home/jk/mimic2aloha_Aug30/act')
from constants import DT
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data # data functions
from utils import sample_box_pose, sample_insertion_pose # robot functions
from utils import compute_dict_mean, set_seed, detach_dict # helper functions
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

from imitate_episodes import forward_pass, make_optimizer, make_policy, plot_history

import IPython
e = IPython.embed


def train(config, act_args, device):
    """
    Train a model using the algorithm.
    """

    # first set seeds
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)

    torch.set_num_threads(2)

    # ACT config
    # command line parameters
    is_eval = act_args['eval']
    ckpt_dir = act_args['ckpt_dir']
    policy_class = act_args['policy_class']
    onscreen_render = act_args['onscreen_render']
    task_name = act_args['task_name']
    batch_size_train = act_args['batch_size']
    batch_size_val = act_args['batch_size']
    num_epochs = act_args['num_epochs']

    # get task parameters
    is_sim = task_name[:4] == 'sim_'
    if is_sim:
        from constants import SIM_TASK_CONFIGS
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    # fixed parameters
    state_dim = 14
    lr_backbone = 1e-5
    backbone = 'resnet18'
    enc_layers = 4
    dec_layers = 7
    nheads = 8
    act_policy_config = {'lr': 1e-5,
                        'num_queries': 100,
                        'kl_weight': 10,
                        'hidden_dim': 512,
                        'dim_feedforward': 3200,
                        'lr_backbone': lr_backbone,
                        'backbone': backbone,
                        'enc_layers': enc_layers,
                        'dec_layers': dec_layers,
                        'nheads': nheads,
                        'camera_names': camera_names
                        }

    act_config = {
        'num_epochs': num_epochs,
        'ckpt_dir': "/home/jk/mimic2aloha_Aug30/act_trained_models",
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': 1e-5,
        'policy_class': policy_class,
        # 'onscreen_render': onscreen_render,
        'policy_config': act_policy_config,
        'task_name': 'sim_transfer_cube_scripted',
        'seed': 1,
        # 'temporal_agg': args['temporal_agg'],
        'camera_names': camera_names,
        'real_robot': True
    }
    # ACT config end

    print("\n============= New Training Run with Config =============")
    print(config)
    print("")
    log_dir, ckpt_dir, video_dir = TrainUtils.get_exp_dir(config)

    if config.experiment.logging.terminal_output_to_txt:
        # log stdout and stderr to a text file
        logger = PrintLogger(os.path.join(log_dir, 'log.txt'))
        sys.stdout = logger
        sys.stderr = logger

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # make sure the dataset exists
    dataset_path = os.path.expanduser(config.train.data)
    if not os.path.exists(dataset_path):
        raise Exception("Dataset at provided path {} not found!".format(dataset_path))

    # load basic metadata from training file
    print("\n============= Loaded Environment Metadata =============")
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=config.train.data)
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=config.train.data,
        all_obs_keys=config.all_obs_keys,
        verbose=True
    )

    if config.experiment.env is not None:
        env_meta["env_name"] = config.experiment.env
        print("=" * 30 + "\n" + "Replacing Env to {}\n".format(env_meta["env_name"]) + "=" * 30)

    # create environment
    envs = OrderedDict()
    if config.experiment.rollout.enabled:
        # create environments for validation runs
        env_names = [env_meta["env_name"]]

        if config.experiment.additional_envs is not None:
            for name in config.experiment.additional_envs:
                env_names.append(name)

        for env_name in env_names:
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                env_name=env_name, 
                render=False, 
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"], 
            )
            env = EnvUtils.wrap_env_from_config(env, config=config) # apply environment warpper, if applicable
            envs[env.name] = env
            print(envs[env.name])

    print("")

    # setup for a new training run
    # data_logger = DataLogger(
    #     log_dir,
    #     config,
    #     log_tb=config.experiment.logging.log_tb,
    #     log_wandb=config.experiment.logging.log_wandb,
    # )

    # model = algo_factory(
    #     algo_name=config.algo_name,
    #     config=config,
    #     obs_key_shapes=shape_meta["all_shapes"],
    #     ac_dim=shape_meta["ac_dim"],
    #     device=device,
    # )

    # act policy and optimizer
    act_policy = make_policy(act_config['policy_class'], act_policy_config)
    act_policy.cuda()
    optimizer = make_optimizer(act_config['policy_class'], act_policy)
    # act policy and optimizer end
    
    # save the config as a json file
    with open(os.path.join(log_dir, '..', 'config.json'), 'w') as outfile:
        json.dump(config, outfile, indent=4)

    # print("\n============= Model Summary =============")
    # print(model)  # print model summary
    # print("")

    # load training data
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"])
    train_sampler = trainset.get_dataset_sampler()
    print("\n============= Training Dataset =============")
    print(trainset)
    print("")
    if validset is not None:
        print("\n============= Validation Dataset =============")
        print(validset)
        print("")

    # maybe retreve statistics for normalizing observations
    obs_normalization_stats = None
    if config.train.hdf5_normalize_obs:
        obs_normalization_stats = trainset.get_obs_normalization_stats()

    # initialize data loaders
    train_loader = DataLoader(
        dataset=trainset,
        sampler=train_sampler,
        batch_size=config.train.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.train.num_data_workers,
        drop_last=True
    )

    if config.experiment.validate:
        # cap num workers for validation dataset at 1
        num_workers = min(config.train.num_data_workers, 1)
        valid_sampler = validset.get_dataset_sampler()
        valid_loader = DataLoader(
            dataset=validset,
            sampler=valid_sampler,
            batch_size=config.train.batch_size,
            shuffle=(valid_sampler is None),
            num_workers=num_workers,
            drop_last=True
        )
    else:
        valid_loader = None

    # print all warnings before training begins
    print("*" * 50)
    print("Warnings generated by robomimic have been duplicated here (from above) for convenience. Please check them carefully.")
    flush_warnings()
    print("*" * 50)
    print("")

    # # main training loop
    # best_valid_loss = None
    # best_return = {k: -np.inf for k in envs} if config.experiment.rollout.enabled else None
    # best_success_rate = {k: -1. for k in envs} if config.experiment.rollout.enabled else None
    # last_ckpt_time = time.time()

    # # number of learning steps per epoch (defaults to a full dataset pass)
    # train_num_steps = config.experiment.epoch_every_n_steps
    # valid_num_steps = config.experiment.validation_epoch_every_n_steps

    # for epoch in range(1, config.train.num_epochs + 1): # epoch numbers start at 1
    #     step_log = TrainUtils.run_epoch(
    #         model=model,
    #         data_loader=train_loader,
    #         epoch=epoch,
    #         num_steps=train_num_steps,
    #         obs_normalization_stats=obs_normalization_stats,
    #     )
    #     model.on_epoch_end(epoch)

    #     # setup checkpoint path
    #     epoch_ckpt_name = "model_epoch_{}".format(epoch)

    #     # check for recurring checkpoint saving conditions
    #     should_save_ckpt = False
    #     if config.experiment.save.enabled:
    #         time_check = (config.experiment.save.every_n_seconds is not None) and \
    #             (time.time() - last_ckpt_time > config.experiment.save.every_n_seconds)
    #         epoch_check = (config.experiment.save.every_n_epochs is not None) and \
    #             (epoch > 0) and (epoch % config.experiment.save.every_n_epochs == 0)
    #         epoch_list_check = (epoch in config.experiment.save.epochs)
    #         should_save_ckpt = (time_check or epoch_check or epoch_list_check)
    #     ckpt_reason = None
    #     if should_save_ckpt:
    #         last_ckpt_time = time.time()
    #         ckpt_reason = "time"

    #     print("Train Epoch {}".format(epoch))
    #     print(json.dumps(step_log, sort_keys=True, indent=4))
    #     for k, v in step_log.items():
    #         if k.startswith("Time_"):
    #             data_logger.record("Timing_Stats/Train_{}".format(k[5:]), v, epoch)
    #         else:
    #             data_logger.record("Train/{}".format(k), v, epoch)

    #     # Evaluate the model on validation set
    #     if config.experiment.validate:
    #         with torch.no_grad():
    #             step_log = TrainUtils.run_epoch(model=model, data_loader=valid_loader, epoch=epoch, validate=True, num_steps=valid_num_steps)
    #         for k, v in step_log.items():
    #             if k.startswith("Time_"):
    #                 data_logger.record("Timing_Stats/Valid_{}".format(k[5:]), v, epoch)
    #             else:
    #                 data_logger.record("Valid/{}".format(k), v, epoch)

    #         print("Validation Epoch {}".format(epoch))
    #         print(json.dumps(step_log, sort_keys=True, indent=4))

    #         # save checkpoint if achieve new best validation loss
    #         valid_check = "Loss" in step_log
    #         if valid_check and (best_valid_loss is None or (step_log["Loss"] <= best_valid_loss)):
    #             best_valid_loss = step_log["Loss"]
    #             if config.experiment.save.enabled and config.experiment.save.on_best_validation:
    #                 epoch_ckpt_name += "_best_validation_{}".format(best_valid_loss)
    #                 should_save_ckpt = True
    #                 ckpt_reason = "valid" if ckpt_reason is None else ckpt_reason

    #     # Evaluate the model by by running rollouts

    #     # do rollouts at fixed rate or if it's time to save a new ckpt
    #     video_paths = None
    #     rollout_check = (epoch % config.experiment.rollout.rate == 0) or (should_save_ckpt and ckpt_reason == "time")
    #     if config.experiment.rollout.enabled and (epoch > config.experiment.rollout.warmstart) and rollout_check:

    #         # wrap model as a RolloutPolicy to prepare for rollouts
    #         rollout_model = RolloutPolicy(model, obs_normalization_stats=obs_normalization_stats)

    #         num_episodes = config.experiment.rollout.n
    #         all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
    #             policy=rollout_model,
    #             envs=envs,
    #             horizon=config.experiment.rollout.horizon,
    #             use_goals=config.use_goals,
    #             num_episodes=num_episodes,
    #             render=False,
    #             video_dir=video_dir if config.experiment.render_video else None,
    #             epoch=epoch,
    #             video_skip=config.experiment.get("video_skip", 5),
    #             terminate_on_success=config.experiment.rollout.terminate_on_success,
    #         )

    #         # summarize results from rollouts to tensorboard and terminal
    #         for env_name in all_rollout_logs:
    #             rollout_logs = all_rollout_logs[env_name]
    #             for k, v in rollout_logs.items():
    #                 if k.startswith("Time_"):
    #                     data_logger.record("Timing_Stats/Rollout_{}_{}".format(env_name, k[5:]), v, epoch)
    #                 else:
    #                     data_logger.record("Rollout/{}/{}".format(k, env_name), v, epoch, log_stats=True)

    #             print("\nEpoch {} Rollouts took {}s (avg) with results:".format(epoch, rollout_logs["time"]))
    #             print('Env: {}'.format(env_name))
    #             print(json.dumps(rollout_logs, sort_keys=True, indent=4))

    #         # checkpoint and video saving logic
    #         updated_stats = TrainUtils.should_save_from_rollout_logs(
    #             all_rollout_logs=all_rollout_logs,
    #             best_return=best_return,
    #             best_success_rate=best_success_rate,
    #             epoch_ckpt_name=epoch_ckpt_name,
    #             save_on_best_rollout_return=config.experiment.save.on_best_rollout_return,
    #             save_on_best_rollout_success_rate=config.experiment.save.on_best_rollout_success_rate,
    #         )
    #         best_return = updated_stats["best_return"]
    #         best_success_rate = updated_stats["best_success_rate"]
    #         epoch_ckpt_name = updated_stats["epoch_ckpt_name"]
    #         should_save_ckpt = (config.experiment.save.enabled and updated_stats["should_save_ckpt"]) or should_save_ckpt
    #         if updated_stats["ckpt_reason"] is not None:
    #             ckpt_reason = updated_stats["ckpt_reason"]

    #     # Only keep saved videos if the ckpt should be saved (but not because of validation score)
    #     should_save_video = (should_save_ckpt and (ckpt_reason != "valid")) or config.experiment.keep_all_videos
    #     if video_paths is not None and not should_save_video:
    #         for env_name in video_paths:
    #             os.remove(video_paths[env_name])

    #     # Save model checkpoints based on conditions (success rate, validation loss, etc)
    #     if should_save_ckpt:
    #         TrainUtils.save_model(
    #             model=model,
    #             config=config,
    #             env_meta=env_meta,
    #             shape_meta=shape_meta,
    #             ckpt_path=os.path.join(ckpt_dir, epoch_ckpt_name + ".pth"),
    #             obs_normalization_stats=obs_normalization_stats,
    #         )

    #     # Finally, log memory usage in MB
    #     process = psutil.Process(os.getpid())
    #     mem_usage = int(process.memory_info().rss / 1000000)
    #     data_logger.record("System/RAM Usage (MB)", mem_usage, epoch)
    #     print("\nEpoch {} Memory Usage: {} MB\n".format(epoch, mem_usage))

    # # terminate logging
    # data_logger.close()

    # act main training loop
    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(act_config['num_epochs'])):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            act_policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(valid_loader):
                # print(f"{data.keys()}")
                # print(f"{data['obs']['agentview_image'].shape}")
                # print(f"{data['obs']['robot0_eef_pos'].shape}")
                # print(f"{data['obs']['robot0_eef_quat'].shape}")
                # print(f"{data['obs']['robot0_gripper_qpos'].shape}")
                # print(f"{data['actions'].shape}")
                # print(f"{data['obs']['agentview_image'][0][0]/256}")
                
                # print(f"{data['obs'].keys()}")
                forward_dict = forward_pass(data, act_policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(act_policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        act_policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_loader):
            forward_dict = forward_pass(data, act_policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{config.train.seed}.ckpt')
            torch.save(act_policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, config.train.seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(act_policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{config.train.seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {config.train.config.train.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, act_config['num_epochs'], ckpt_dir, config.train.seed)

    # act main training loop end


def main(args, act_args):

    if args.config is not None:
        ext_cfg = json.load(open(args.config, 'r'))
        config = config_factory(ext_cfg["algo_name"])
        # update config with external json - this will throw errors if
        # the external config has keys not present in the base algo config
        with config.values_unlocked():
            config.update(ext_cfg)
    else:
        config = config_factory(args.algo)

    if args.dataset is not None:
        config.train.data = args.dataset

    if args.name is not None:
        config.experiment.name = args.name

    # get torch device
    device = TorchUtils.get_torch_device(try_to_use_cuda=config.train.cuda)

    # maybe modify config for debugging purposes
    if args.debug:
        # shrink length of training to test whether this run is likely to crash
        config.unlock()
        config.lock_keys()

        # train and validate (if enabled) for 3 gradient steps, for 2 epochs
        config.experiment.epoch_every_n_steps = 3
        config.experiment.validation_epoch_every_n_steps = 3
        config.train.num_epochs = 2

        # if rollouts are enabled, try 2 rollouts at end of each epoch, with 10 environment steps
        config.experiment.rollout.rate = 1
        config.experiment.rollout.n = 2
        config.experiment.rollout.horizon = 10

        # send output to a temporary directory
        config.train.output_dir = "/tmp/tmp_trained_models"

    # lock config to prevent further modifications and ensure missing keys raise errors
    config.lock()

    # catch error during training and print it
    res_str = "finished run successfully!"
    try:
        train(config, act_args, device=device)
    except Exception as e:
        res_str = "run failed with error:\n{}\n\n{}".format(e, traceback.format_exc())
    print(res_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # External config file that overwrites default config
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="(optional) path to a config json that will be used to override the default settings. \
            If omitted, default settings are used. This is the preferred way to run experiments.",
    )

    # Algorithm Name
    parser.add_argument(
        "--algo",
        type=str,
        help="(optional) name of algorithm to run. Only needs to be provided if --config is not provided",
    )

    # Experiment Name (for tensorboard, saving models, etc.)
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="(optional) if provided, override the experiment name defined in the config",
    )

    # Dataset path, to override the one in the config
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="(optional) if provided, override the dataset path defined in the config",
    )

    # debug mode
    parser.add_argument(
        "--debug",
        action='store_true',
        help="set this flag to run a quick training run for debugging purposes"
    )

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

    # for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    args = parser.parse_args()
    
    act_args = vars(parser.parse_args())

    main(args, act_args)

