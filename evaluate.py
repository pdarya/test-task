import json
import os
from tqdm import tqdm

import cv2
import hydra
import numpy as np
import torch

from dataset import DemoDataset
from policy import ACTPolicy


def get_model_input(observation, normalize_qpos_fn=None):
    # qpos
    proprio_observations = []
    for key in DemoDataset.OBSERVATION_PROPRIO_KEYS:
        proprio_observations.append(observation[key])
    qpos = np.concatenate(proprio_observations).astype(np.float32)
    if normalize_qpos_fn:
        qpos = normalize_qpos_fn(qpos)
    qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)

    # camera images
    images = []
    for key in DemoDataset.OBSERVATION_CAM_KEYS:
        images.append(observation[key])
    cameras = np.array(images).astype(np.float32) / 255.
    cameras = torch.from_numpy(cameras).float().cuda().unsqueeze(0)

    return qpos, cameras


def eval(cfg):
    torch.manual_seed(cfg.evaluation.seed)
    np.random.seed(cfg.evaluation.seed)

    # load stats
    with open(cfg.evaluation.stats_file_path, 'r') as f:
        stats = json.load(f)
    pre_process_qpos = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process_action = lambda a: a * stats['actions_std'] + stats['actions_mean']

    # load policy
    policy = ACTPolicy(cfg, mean=stats['cameras_mean'], std=stats['cameras_std'])
    checkpoint = torch.load(cfg.evaluation.ckpt_path)
    loading_status = policy.load_state_dict(checkpoint['model_state_dict'])
    print(loading_status)

    policy.cuda()
    policy.eval()
    print(f'loaded model from: {cfg.evaluation.ckpt_path}')

    # make environment
    env = hydra.utils.instantiate(cfg.env)

    query_frequency = cfg.model.num_actions
    if cfg.evaluation.temporal_agg:
        query_frequency = 1

    max_timesteps = int(cfg.evaluation.max_timesteps)  # may increase for real-world tasks

    episode_returs = []
    # evaluation loop
    for rollout_id in range(cfg.evaluation.num_rollouts):
        obs, _ = env.reset(seed=rollout_id)

        if cfg.evaluation.temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps + cfg.model.num_actions, cfg.action_dim]).cuda()
        image_list, qpos_list, applied_actions, rewards = [], [], [], []  # for visualization

        with torch.inference_mode():
            for t in tqdm(range(max_timesteps)):
                # for visualization
                image_list.append({camera_name: obs[camera_name] for camera_name in ['rgb_head', 'rgb_left_wrist', 'rgb_right_wrist']})

                # prepare model input
                qpos, camera_images = get_model_input(obs, pre_process_qpos)

                # query policy
                if t % query_frequency == 0:
                    all_actions = policy(qpos, camera_images)
                if cfg.evaluation.temporal_agg:
                    all_time_actions[[t], t:t + cfg.model.num_actions] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    exp_weights = np.exp(-cfg.evaluation.k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)  # to apply in env
                else:
                    raw_action = all_actions[:, t % query_frequency]

                # post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action_to_apply = post_process_action(raw_action)  # denormalize
                action_to_apply = np.clip(action_to_apply, env.action_space.low, env.action_space.high)

                # step the environment
                obs, reward, terminated, truncated, info = env.step(action_to_apply)

                # for visualizations
                qpos_list.append(qpos.cpu().numpy())
                applied_actions.append(action_to_apply)
                rewards.append(reward)

                if terminated:
                    image_list.append({camera_name: obs[camera_name] for camera_name in ['rgb_head', 'rgb_left_wrist', 'rgb_right_wrist']})
                    break

        print(f'rollout #{rollout_id}, ts: {t}, success={np.sum(rewards) > 0}')
        episode_returs.append(np.sum(rewards))
        if cfg.evaluation.save_video and rollout_id < 5:
            save_videos(image_list, video_path=os.path.join(cfg.evaluation.videos_dir, f'rollout_{rollout_id}.mp4'))

    success_rate = np.mean(episode_returs)
    print(f'evaluation finished:\n rollouts number: {cfg.evaluation.num_rollouts}, success rate: {success_rate}')


def save_videos(video, fps=50, video_path=None):
    if isinstance(video, list):
        cam_names = list(video[0].keys())
        print('in video:', len(video), video[0][cam_names[0]].shape)
        c, h, w = video[0][cam_names[0]].shape
        w = w * len(cam_names)
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for ts, image_dict in enumerate(video):
            images = []
            for cam_name in cam_names:
                image = image_dict[cam_name]
                image = np.transpose(image[[2, 1, 0], :, :], (1, 2, 0))  # swap B and R channel
                images.append(image)
            images = np.concatenate(images, axis=1)
            out.write(images)
        out.release()
        print(f'saved video to: {video_path}')
    elif isinstance(video, dict):
        cam_names = list(video.keys())
        all_cam_videos = []
        for cam_name in cam_names:
            all_cam_videos.append(video[cam_name])
        all_cam_videos = np.concatenate(all_cam_videos, axis=2)  # width dimension

        n_frames, h, w, _ = all_cam_videos.shape
        out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for t in range(n_frames):
            image = all_cam_videos[t]
            image = image[:, :, [2, 1, 0]]  # swap B and R channel
            out.write(image)
        out.release()
        print(f'saved video to: {video_path}')


@hydra.main(version_base=None, config_path='conf', config_name='config')
def main(cfg):
    eval(cfg)


if __name__ == '__main__':
    main()
