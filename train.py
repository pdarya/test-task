from copy import deepcopy
import itertools
from datetime import datetime
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
from bigym.utils.observation_config import ObservationConfig, CameraConfig
from demonstrations.demo_player import DemoPlayer
from demonstrations.demo_store import DemoStore
from demonstrations.utils import Metadata

import utils
from dataset import DemoDataset
from policy import ACTPolicy

torch.backends.cudnn.benchmark = True


def load_data(cfg: DictConfig):
    env = ReachTarget(  # TODO from hydra config and check
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

    metadata = Metadata.from_env(env)
    demo_store = DemoStore()
    print(f'going to load #{cfg.demos_cnt} demos')
    demos = demo_store.get_demos(metadata, amount=cfg.demos_cnt, frequency=50)
    print(f'loaded demos #{len(demos)}')

    shuffled_indices = np.random.permutation(len(demos))
    train_indices = shuffled_indices[:int(cfg.train_ratio * len(demos))]
    val_indices = shuffled_indices[int(cfg.train_ratio * len(demos)):]
    print(f'train episodes #{len(train_indices)}, val episodes #{len(val_indices)}')

    train_dataset = DemoDataset(actions_num=cfg.actions_num)
    for idx in train_indices:
        train_dataset.add_episode(demos[idx])
    train_dataset.compute_stats(save_stats=True)

    val_dataset = DemoDataset(actions_num=cfg.actions_num)
    for idx in val_indices:
        val_dataset.add_episode(demos[idx])
    val_dataset.compute_stats(save_stats=False)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=True)

    return train_loader, val_loader, {
        'coefs': train_dataset.normalization_coefs,
        'train_size': train_dataset.size(),
        'val_size': val_dataset.size(),
    }


def train(policy: ACTPolicy, train_dataloader, val_dataloader, cfg: DictConfig):
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(cfg.num_epochs)):
        print(f'epoch #{epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(itertools.islice(val_dataloader, cfg.val_steps)):
                cameras, qpos, actions, mask = data['cameras'], data['qpos'], data['actions'], data['mask']
                cameras, qpos, actions, mask = cameras.cuda(), qpos.cuda(), actions.cuda(), mask.cuda()
                forward_dict = policy(qpos, cameras, actions, mask)  # l1, kl, loss
                epoch_dicts.append(forward_dict)
                if batch_idx % cfg.log_interval == 0:
                    wandb.log(utils.add_prefix(forward_dict, 'batch/val/'))
            epoch_summary = utils.compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            # save best model
            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
                ckpt_path = os.path.join(cfg.ckpt_dir, f'policy_epoch_{best_epoch}_seed_{cfg.seed}_best.ckpt')
                torch.save(best_state_dict, ckpt_path)

        # log validation metrics
        wandb.log(utils.add_prefix(epoch_summary, 'epoch/val/'))
        loss_summary_string = ' '.join([f'{loss_type}: {loss_value.item():.5f}' for loss_type, loss_value in epoch_summary.items()])
        print('validation:', loss_summary_string)

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
                wandb.log(utils.add_prefix(stats, 'batch/train/'))

        # log training metrics by epoch
        epoch_summary = utils.compute_dict_mean(train_history)
        epoch_summary['lr'] = policy.optimizer.param_groups[-1]['lr']
        wandb.log(utils.add_prefix(epoch_summary, 'epoch/train/'))
        loss_summary_string = ' '.join([f'{loss_type}: {loss_value.item():.5f}' for loss_type, loss_value in epoch_summary.items()])
        print('training:', loss_summary_string)

        if epoch % cfg.save_ckpt_frequency == 0:
            ckpt_path = os.path.join(cfg.ckpt_dir, f'policy_epoch_{epoch}_seed_{cfg.seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)

    # saving last ckpt
    ckpt_path = os.path.join(cfg.ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    print(f'training finished: seed {cfg.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    return best_ckpt_info


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg: DictConfig):
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    train_dataloader, val_dataloader, info = load_data(cfg.data)
    cfg.training.train_steps, cfg.training.val_steps = info['train_size'], info['val_size']
    print(cfg)
    return
    # wandb.init(config=OmegaConf.to_object(cfg), project='bigym_act', name=f"run_{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}")
    policy = ACTPolicy(cfg.model, mean=info['coefs']['cameras_mean'], std=info['coefs']['cameras_std'])
    # wandb.watch(policy)

    train(policy, train_dataloader, val_dataloader, cfg.training)
    # wandb.finish()


if __name__ == '__main__':
    main()
