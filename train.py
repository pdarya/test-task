from copy import deepcopy
from datetime import datetime
import itertools
import json
import os
from tqdm import tqdm

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
import wandb

from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.bigym_env import CONTROL_FREQUENCY_MAX  # 500 Hz
from bigym.envs.reach_target import ReachTarget
from bigym.envs.move_plates import MovePlate
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

import utils
from dataset import DemoDataset
from policy import ACTPolicy

torch.backends.cudnn.benchmark = True


def load_data(cfg: DictConfig, env, resume: bool = False):
    env = MovePlate(  # TODO from hydra config and check
        action_mode=JointPositionActionMode(
            absolute=True,
            floating_base=True,  # only absolute = false ?
            # floating_dofs=[PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ],
        ),
        observation_config=ObservationConfig(
            cameras=[
                CameraConfig(
                    name=camera,
                    rgb=True,
                    depth=False,  # TODO
                    resolution=(84, 84),  # default is (128, 128)
                ) for camera in ['head', 'right_wrist', 'left_wrist']
            ],
            proprioception=True,
        ),
        render_mode='rgb_array',
        control_frequency=CONTROL_FREQUENCY_MAX / 10,  # as in ACT 50 Hz
        start_seed=0,
    )

    datasets = {}
    for dataset_type in ('train', 'val'):
        datasets[dataset_type] = DemoDataset(
            dataset_path=os.path.join(cfg.data_dir, dataset_type),
            actions_num=cfg.actions_num,
            full_demo=cfg.full_demo,
            resume=True,  # resume,  # TODO resume
        )

    if False:  # not resume:  # TODO
        metadata = Metadata.from_env(env)
        demo_store = DemoStore()
        print(f'going to load #{cfg.demos_cnt} demos')
        demos = demo_store.get_demos(metadata, amount=cfg.demos_cnt, frequency=50)
        print(f'loaded demos #{len(demos)}')

        shuffled_indices = np.random.permutation(len(demos))
        train_indices = shuffled_indices[:int(cfg.train_ratio * len(demos))]
        val_indices = shuffled_indices[int(cfg.train_ratio * len(demos)):]
        print(f'train episodes #{len(train_indices)}, val episodes #{len(val_indices)}')

        for dataset_type, indices, save_stats in (
            ('train', train_indices, True),
            ('val', val_indices, False),
        ):
            for idx in indices:
                datasets[dataset_type].add_episode(demos[idx])
            datasets[dataset_type].compute_stats(save_stats=save_stats)

    train_loader = DataLoader(datasets['train'], batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(datasets['val'], batch_size=cfg.batch_size, num_workers=0, pin_memory=True)

    return train_loader, val_loader, {
        'coefs': datasets['train'].normalization_coefs,
        'train_size': datasets['train'].size(),
        'val_size': datasets['val'].size(),
    }


def train(policy: ACTPolicy, train_dataloader, val_dataloader, cfg: DictConfig, resume: bool = False):
    if resume:
        # load last checkpoint
        checkpoint = torch.load(os.path.join(cfg.ckpt_dir, f'last.ckpt'))
        policy.load_state_dict(checkpoint['model_state_dict'])
        policy.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        policy.cuda()
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0
    print(f'will run {cfg.num_epochs} epochs in total starting from epoch {start_epoch}, {cfg.train_steps} train steps and {cfg.val_steps} validation steps per epoch')

    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in range(start_epoch, cfg.num_epochs):
        print(f'epoch #{epoch}')
        if epoch != 0 and epoch % 50 == 0:
            for g in policy.optimizer.param_groups:
                g['lr'] *= 0.5
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(itertools.islice(val_dataloader, cfg.val_steps)):
                cameras, qpos, actions, mask = data['cameras'], data['qpos'], data['actions'], data['mask']
                cameras, qpos, actions, mask = cameras.cuda(), qpos.cuda(), actions.cuda(), mask.cuda()
                forward_dict = policy(qpos, cameras, actions, mask)  # l1, kl, loss
                epoch_dicts.append(utils.detach_dict(forward_dict))
                if batch_idx % cfg.log_interval == 0:
                    wandb.log(utils.add_prefix(forward_dict, 'batch/val/', {'batch': epoch * cfg.val_steps + batch_idx}))
            epoch_summary = utils.compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # save best model
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                utils.save_checkpoint(policy=policy, epoch=epoch, path=os.path.join(cfg.ckpt_dir, f'best.ckpt'))
                print('new best model saved')

        # log validation metrics
        wandb.log(utils.add_prefix(epoch_summary, 'epoch/val/', {'epoch': epoch}))
        loss_summary_string = ' '.join([f'{loss_type}: {loss_value:.5f}' for loss_type, loss_value in epoch_summary.items()])
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'validation:', loss_summary_string)

        # training
        train_history = []
        policy.train()
        policy.optimizer.zero_grad()
        for batch_idx, data in enumerate(itertools.islice(train_dataloader, cfg.train_steps)):
            cameras, qpos, actions, mask = data['cameras'], data['qpos'], data['actions'], data['mask']
            cameras, qpos, actions, mask = cameras.cuda(), qpos.cuda(), actions.cuda(), mask.cuda()
            forward_dict = policy(qpos, cameras, actions, mask)  # l1, kl, loss
            # backward
            loss = forward_dict['loss']
            loss.backward()
            policy.optimizer.step()
            policy.optimizer.zero_grad()
            # stats
            stats = utils.detach_dict(forward_dict)
            train_history.append(stats)
            if batch_idx % cfg.log_interval == 0:
                wandb.log(utils.add_prefix(stats, 'batch/train/', {'batch': epoch * cfg.train_steps + batch_idx}))

        # log training metrics by epoch
        epoch_summary = utils.compute_dict_mean(train_history)
        epoch_summary['lr'] = policy.optimizer.param_groups[-1]['lr']
        wandb.log(utils.add_prefix(epoch_summary, 'epoch/train/', {'epoch': epoch}))
        loss_summary_string = ' '.join([f'{loss_type}: {loss_value:.5f}' for loss_type, loss_value in epoch_summary.items()])
        print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 'training:', loss_summary_string)

        if epoch % cfg.save_ckpt_frequency == 0:
            utils.save_checkpoint(policy=policy, epoch=epoch, path=os.path.join(cfg.ckpt_dir, f'epoch_{epoch}_seed_{cfg.seed}.ckpt'))

        # saving last ckpt
        utils.save_checkpoint(policy=policy, epoch=epoch, path=os.path.join(cfg.ckpt_dir, f'last.ckpt'))

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f'training finished: seed {cfg.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    return best_ckpt_info


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)
    utils.prepare_dirs(cfg)

    # prepare data
    resume = cfg.training.wandb_run_id != ''
    print(f'if resume training: {resume}')
    # env = hydra.utils.instantiate(cfg.env)
    train_dataloader, val_dataloader, info = load_data(cfg.data, None, resume=resume)
    cfg.training.train_steps, cfg.training.val_steps = 300, min(info['val_size'], 1)

    if resume:
        wandb.init(
            entity='dapoli',
            project='bigym_act',
            id=cfg.training.wandb_run_id,
            resume='must',
        )
    else:
        wandb.init(
            config=OmegaConf.to_object(cfg),
            project='bigym_act',
            name=f"run_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}",
        )

    policy = ACTPolicy(cfg, mean=info['coefs']['cameras_mean'], std=info['coefs']['cameras_std'])
    wandb.watch(policy)

    train(policy, train_dataloader, val_dataloader, cfg.training, resume)
    wandb.finish()


if __name__ == '__main__':
    main()
