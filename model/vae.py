# modified from https://github.com/tonyzhaozh/act/blob/main/detr/models/detr_vae.py

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from .transformer import TransformerEncoder, TransformerEncoderLayer


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class DETRVAE(nn.Module):
    def __init__(
        self,
        backbone,
        transformer,
        encoder,
        num_actions: int,
        action_dim: int,
        observation_dim: int,
        latent_dim: int = 32,
        use_task_emb: bool = False,
    ):
        super().__init__()
        self.transformer = transformer
        self.encoder = encoder
        self.num_actions = num_actions
        self.use_task_emb = use_task_emb

        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_actions, hidden_dim)  # like detr queries

        if backbone is not None:
            self.backbone = backbone
            # to align output channels with hidden_dim
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
            self.input_proj_robot_state = nn.Linear(observation_dim, hidden_dim)
            self.input_task_embed = nn.Embedding(10, hidden_dim) if self.use_task_emb else None
        else:
            raise NotImplementedError('No image backbone')

        # encoder extra parameters
        self.latent_dim = latent_dim  # final size of latent z
        self.cls_embed = nn.Embedding(1, hidden_dim)  # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)  # project action to hidden_dim
        self.encoder_joint_proj = nn.Linear(observation_dim, hidden_dim)  # project qpos to hidden_dim
        self.encoder_task_embed = nn.Embedding(10, hidden_dim) if self.use_task_emb else None

        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)  # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(2 + num_actions if not self.use_task_emb else 3 + num_actions, hidden_dim))  # [CLS], qpos, (task), a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)  # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2 if not self.use_task_emb else 3, hidden_dim)  # learned position embedding for proprio and latent

    def forward(self, qpos, image, env_state=None, actions=None, is_pad=None, task_type=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        is_pad: batch, seq (actions mask)
        """
        is_training = actions is not None  # train or val
        bs, _ = qpos.shape
        if is_training:  # obtain latent z from action sequence
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions)  # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight  # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)  # (bs, 1, hidden_dim)

            if self.use_task_emb:
                task_embed = self.encoder_task_embed(task_type).unsqueeze(1)   # (bs, 1, hidden_dim
                encoder_input = torch.cat([cls_embed, task_embed, qpos_embed, action_embed], axis=1)  # (bs, 1 + 1 + actions_num, hidden_dim)
            else:
                encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1)  # (bs, 1 + 1 + actions_num, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2)  # (seq_len, bs, hidden_dim)

            # prepare mask for encoder, do not mask cls token and qpos token
            cls_joint_is_pad = torch.full((bs, 2 if not self.use_task_emb else 3), False).to(qpos.device)  # cls and qpos (and task emb) are not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq_len)

            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq_len, 1, hidden_dim)

            # apply encoder
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]  # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)

        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbone is not None:
            # image observation features and position embeddings
            all_cam_features = []
            all_cam_pos = []
            cameras_number = image.shape[1]
            for cam_id in range(cameras_number):
                features, pos = self.backbone(image[:, cam_id])  # apply backbone and compute positional embeddings
                features = features[0]  # take the last layer feature
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))  # project to obtain hidden_dim
                all_cam_pos.append(pos)
            # proprioception features
            proprio_input = self.input_proj_robot_state(qpos)
            # task_embed
            task_embed_input = self.input_task_embed(task_type) if self.use_task_emb else None
            # fold camera dimension into width dimension
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, task_embed_input)[0]

        else:
            qpos = self.input_proj_robot_state(qpos)
            env_state = self.input_proj_env_state(env_state)
            transformer_input = torch.cat([qpos, env_state], axis=1)  # seq length = 2
            hs = self.transformer(transformer_input, None, self.query_embed.weight, self.pos.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        return a_hat, is_pad_hat, [mu, logvar]


def build_encoder(cfg):
    d_model = cfg.hidden_dim  # 256
    normalize_before = cfg.pre_norm  # False

    encoder_layer = TransformerEncoderLayer(
        d_model,
        cfg.nheads,
        cfg.dim_feedforward,
        cfg.dropout,
        "relu",
        normalize_before,
    )
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, cfg.enc_layers, encoder_norm)

    return encoder
