from collections import defaultdict, OrderedDict
import io
import json
import shutil
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset


class DemoDataset(IterableDataset):
    OBSERVATION_PROPRIO_KEYS = (
        'proprioception',
        'proprioception_floating_base',
        # 'proprioception_floating_base_actions',  # accumulated actions
        'proprioception_grippers',
    )
    OBSERVATION_CAM_KEYS = (
        'rgb_head',
        'rgb_left_wrist',
        'rgb_right_wrist',
    )

    def __init__(
        self,
        dataset_path: str = 'data',
        max_episodes_buffer_size: int = 10,
        update_buffer_probability: float = 0.2,
        actions_num: int = 16,
    ):
        # data storage
        self.current_episode = defaultdict(list)  # dict of lists with observations/actions
        self.episodes_info = []  # stored episodes TODO check for different workers
        self.dataset_path = dataset_path
        if os.path.exists(self.dataset_path):  # TODO delete
            shutil.rmtree(self.dataset_path)
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)

        # dataset usage
        self.episodes_buffer = OrderedDict()
        self.max_episodes_buffer_size = max_episodes_buffer_size  # TODO
        self.update_buffer_probability = update_buffer_probability
        self.actions_num = actions_num  # actions to predict

        self.stats = defaultdict(lambda: 0)
        self.normalization_coefs = {}

    def size(self):
        """ number of ts stored """
        return sum([info['size'] for info in self.episodes_info])

    def add(
        self,
        observation: dict,
        action: np.ndarray,
        reward: float,
        terminal: bool,
        truncated: bool,
        episode_id: str = '',
        env_name: str = '',
    ):
        self.current_episode['actions'].append(action)

        # concat proprio
        proprio_observations = []
        for key in self.OBSERVATION_PROPRIO_KEYS:
            proprio_observations.append(observation[key])
        self.current_episode['qpos'].append(np.concatenate(proprio_observations))

        # concat images
        images = []
        for key in self.OBSERVATION_CAM_KEYS:
            images.append(observation[key])
        self.current_episode['cameras'].append(np.array(images, dtype=images[0].dtype))

        if terminal or truncated:  # TODO check condition
            for key, value in self.current_episode.items():
                self.current_episode[key] = np.array(value, dtype=value[0].dtype)
    
            self.track_stats()

            file_path = os.path.join(self.dataset_path, f'{len(self.episodes_info)}_{env_name}_{episode_id}.npz')
            self.episodes_info.append(dict(
                id=episode_id,
                size=len(self.current_episode['actions']),
                file_path=file_path,
            ))
            with io.BytesIO() as bs:
                np.savez_compressed(bs, **self.current_episode)
                with open(file_path, 'wb') as f:
                    bs.seek(0)
                    f.write(bs.read())

    def add_episode(self, demo):
        self.current_episode = defaultdict(list)
        # print(f'adding episode with length {len(demo.timesteps)}')
        for idx, ts in enumerate(demo.timesteps):
            self.add(
                observation=ts.observation,
                action=ts.executed_action,
                reward=ts.reward,
                terminal=ts.termination,
                truncated=ts.truncation,
                episode_id=demo.uuid,
                env_name=demo.metadata.env_name,
            )
            if ts.termination or ts.truncation:
                # print(f'termination happened at {idx}th ts')
                return

    def update_stored_episodes(self):
        worker_info = torch.utils.data.get_worker_info()
        num_workers = int(worker_info.num_workers) if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        assert num_workers <= len(self.episodes_info), f'need to reduce dataloader workers number {num_workers} > {len(self.episodes_info)}'

        # do not update filled buffer with some probability
        if len(self.episodes_buffer) == self.max_episodes_buffer_size and np.random.rand() > self.update_buffer_probability:
            return

        if len(self.episodes_buffer) == self.max_episodes_buffer_size:
            self.episodes_buffer.popitem(last=False)  # pop oldest episode

        # try to load new episode to buffer
        episode_idxs = np.random.permutation(np.arange(worker_id, len(self.episodes_info), num_workers))
        for episode_idx in episode_idxs:  # add while buffer is not full
            if episode_idx in self.episodes_buffer:
                continue

            episode_file_path = self.episodes_info[episode_idx]['file_path']
            with open(episode_file_path, 'rb') as f:
                episode = np.load(f)
                episode = {k: episode[k] for k in episode.keys()}

            self.episodes_buffer[episode_idx] = episode

            if len(self.episodes_buffer) == self.max_episodes_buffer_size:
                break

    def sample_single(self):
        # keep k episodes in memory (for every worker)
        # every n samples fetch new episodes
        # prepare sample from current keeped episode (return next m actions maybe padded)

        self.update_stored_episodes()

        episode_idx = np.random.choice(list(self.episodes_buffer.keys()))  # select episode
        episode_size = self.episodes_info[episode_idx]['size']
        ts_idx = np.random.randint(0, episode_size - 1)  # select ts
        # print('sampled:', episode_idx, ts_idx)

        # prepare sample
        sample = {}
        # add observation
        sample['qpos'] = ((self.episodes_buffer[episode_idx]['qpos'][ts_idx]
                          - self.normalization_coefs['qpos_mean']) / self.normalization_coefs['qpos_std']).astype(np.float32)
        sample['cameras'] = (self.episodes_buffer[episode_idx]['cameras'][ts_idx] / 255.).astype(np.float32)

        # add actions (target)
        actions_end_idx = min(ts_idx + self.actions_num, episode_size)
        actions_idxs = list(range(ts_idx, actions_end_idx))
        actions_seq = ((self.episodes_buffer[episode_idx]['actions'][actions_idxs] -
                       self.normalization_coefs['actions_mean']) / self.normalization_coefs['actions_std']).astype(np.float32)
        if len(actions_seq) < self.actions_num:
            actions_seq = np.concatenate(
                [actions_seq, np.zeros((self.actions_num - len(actions_seq), *actions_seq.shape[1:]), dtype=np.float32)],  # TODO
                axis=0,
            )
        sample['actions'] = actions_seq
        sample['mask'] = np.arange(self.actions_num) >= actions_end_idx

        return sample

    def __iter__(self):
        while True:
            yield self.sample_single()

    def track_stats(self):
        """ after current episode is loaded """
        for key, value in self.current_episode.items():
            if key == 'mask':
                continue
            if key == 'cameras':
                self.stats[f'{key}_sum'] += (value / 255.).sum(axis=(0, 1, 3, 4))
                self.stats[f'{key}_sum2'] += ((value / 255.) ** 2).sum(axis=(0, 1, 3, 4))
                self.stats[f'{key}_cnt'] += (value.shape[0] * value.shape[1] * value.shape[3] * value.shape[4])
            else:
                self.stats[f'{key}_sum'] += value.sum(axis=0)
                self.stats[f'{key}_sum2'] += (value ** 2).sum(axis=0)
                self.stats[f'{key}_cnt'] += value.shape[0]

    def compute_stats(self, save_stats=False):  # optional
        sizes = [info['size'] for info in self.episodes_info]
        total_ts = sum(sizes)
        print(f'demo episodes number: {len(self.episodes_info)};'
              f'total ts number: {total_ts}; '
              f'avg ts in episode: {np.mean(sizes):.3f}; '
              f'min|median|max: {np.min(sizes)}|{np.median(sizes)}|{np.max(sizes)};')

        for key in ('cameras', 'qpos', 'actions'):
            total_mean = self.stats[f'{key}_sum'] / self.stats[f'{key}_cnt']
            total_var = (self.stats[f'{key}_sum2'] / self.stats[f'{key}_cnt']) - (total_mean ** 2)
            total_std = np.sqrt(total_var)
            total_std[total_std == 0] = 1.

            print(key, 'mean:', np.round(total_mean, 3), 'std:', np.round(total_std, 3))

            self.normalization_coefs[f'{key}_mean'] = total_mean
            self.normalization_coefs[f'{key}_std'] = total_std

        if save_stats:
            with open(os.path.join(self.dataset_path, 'stats.json'), 'w+') as stats_file:
                json.dump(
                    {key: list(value) for key, value in self.normalization_coefs.items()},
                    stats_file,
                    indent=4,
                )
