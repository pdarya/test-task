import hydra
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from model.backbone import build_backbone
from model.transformer import build_transformer
from model.vae import build_encoder, DETRVAE


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


class ACTPolicy(nn.Module):
    def __init__(self, cfg, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        super().__init__()
        self.model = self.build_model(cfg.model)
        self.model.cuda()

        param_dicts = [
            {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": cfg.optimizer.lr_backbone,
            },
        ]
        self.optimizer = hydra.utils.instantiate(cfg.optimizer.main, params=param_dicts)
        self.kl_weight = cfg.model.kl_weight
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def build_model(self, cfg):
        backbone = build_backbone(cfg.backbone)
        transformer = build_transformer(cfg.transformer)
        encoder = build_encoder(cfg.transformer)

        model = DETRVAE(
            backbone,
            transformer,
            encoder,
            num_actions=cfg.num_actions,
            action_dim=cfg.action_dim,
            observation_dim=cfg.observation_dim,
            latent_dim=cfg.latent_dim,
        )

        n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f'number of parameters: {n_parameters / 1e6:.2f}')

        return model

    def __call__(self, qpos, image, actions=None, is_pad=None):
        image = self.normalize(image)
        print('image after normalization:', torch.min(image), torch.max(image), image)

        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, image, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)
            loss_dict = dict()
            all_l1 = F.l1_loss(actions, a_hat, reduction='none')
            l1 = (all_l1 * ~is_pad.unsqueeze(-1)).mean()
            loss_dict['l1'] = l1
            loss_dict['kl'] = total_kld[0]
            loss_dict['loss'] = loss_dict['l1'] + loss_dict['kl'] * self.kl_weight

            return loss_dict

        else:  # inference time
            a_hat, _, (_, _) = self.model(qpos, image)  # no action, sample from prior
            return a_hat
