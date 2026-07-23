from pathlib import Path
import pickle
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import MetricCollection
import time
import numpy as np

from src.metrics import MR, minADE, minFDE, brier_minFDE, AvgMinADE, ActorMR, AvgMinFDE, AvgBrierMinFDE
from src.utils.optim import WarmupCosLR
from src.model.drifttraj import (
    drifting_loss,
    endpoint_diversity_loss,
    joint_drifting_loss,
    select_winner,
    split_winner_and_others,
)
from src.model.drifttraj.utils import assert_finite_tensor, scheduled_weight

try:
    from pytorch_lightning.utilities.rank_zero import rank_zero_warn
except ImportError:
    def rank_zero_warn(message):
        print(f"WARNING: {message}")


class BaseLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: dict,
        optim: dict = None,
        ma = False
    ) -> None:
        super(BaseLightningModule, self).__init__()
        self.pre_ensemble = True
        self.time_list = []
        self.optim = optim
        # self.save_hyperparameters()

        self.model = model
        self.curr_ep = 0
        self.ma = ma
        self.stream_the_scene = True
        self.ma_loss_each = True
        self.ma_eval_only = False
        self.reset_val_scores(init=True)      

        print( torch.cuda._raw_device_uuid_nvml() )
        return

    def forward(self, data):
        return self.model(data)

    def target_sample_mask(self, data):
        if not getattr(self.model, 'vehicle_only', False):
            return torch.ones(data['x_attr'].shape[0], dtype=torch.bool, device=data['x_attr'].device)
        # AV2 combined type 0 is a motor vehicle (vehicle or bus). Other
        # actors remain in x_attr as scene context, but receive no target loss.
        return data['x_attr'][:, 0, 2].long() == 0
    
    def load_chkpt(self, ckpt_path, multi=False):
        ckpt = torch.load(ckpt_path, map_location="cpu")["state_dict"]
        state_dict = {
            k[len("model.") :]: v for k, v in ckpt.items() if k.startswith("model.")
        }
        incompatible = self.model.load_state_dict(state_dict=state_dict, strict=False)
        if multi:
            state_dict = {
                k[len("consitency_module.") :]: v for k, v in ckpt.items() if k.startswith("consitency_module.")
            }
            self.consitency_module.load_state_dict(state_dict=state_dict, strict=False)
        return incompatible

    """
        Reset the custom metrics before each evaluation run.
    """
    def reset_val_scores(self, init=False):
        if self.ma:
            if init:
                self.metrics = MetricCollection(
                    {
                        "AvgMinADE": AvgMinADE(),
                        "AvgMinFDE": AvgMinFDE(),
                        "ActorMR": ActorMR(),
                        "AvgBrierMinFDE": AvgBrierMinFDE()
                    }
                )
            self.val_scores = {"AvgMinADE": [], "AvgMinFDE": [], "ActorMR": [], "AvgBrierMinFDE": []}
        else:
            if init:
                self.metrics = MetricCollection(
                    {
                        'minADE1': minADE(k=1),
                        'minADE6': minADE(k=10),
                        'minFDE1': minFDE(k=1),
                        'minFDE6': minFDE(k=10),
                        'MR': MR(),
                        'b-minFDE6': brier_minFDE(k=10),
                    }
                )

            self.val_scores = {"MR": [], "minADE1": [], "minADE6": [], "minFDE1": [], "minFDE6": [], "b-minFDE6": []} 
    
    """
        Compute multi-agent loss on scene predictions.
    """
    def ma_cal_loss(self, out, data, tag=''):
        gt_len = data['target'][:, 0].shape[-2] 
        y_hat = out['y_hat'][:, :, :, :gt_len]
        y = data['target'][:, :, :y_hat.shape[3]]
        valid_mask = data["scored_mask"].unsqueeze(1)
        scene_avg_ade = (
            torch.norm(y_hat[..., :2] - y.unsqueeze(1), dim=-1).sum(dim=-1) * valid_mask
        ).sum(dim=-1) / valid_mask.sum(dim=-1)
        best_mode = torch.argmin(scene_avg_ade, dim=-1)

        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]

        pi = out["pi"]
        reg_mask = data["scored_mask"].unsqueeze(-1).unsqueeze(-1).expand_as(y_hat_best)
        agent_reg_loss = F.smooth_l1_loss(y_hat_best[reg_mask], y[reg_mask])
        agent_cls_loss = F.cross_entropy(pi, best_mode.detach())

        loss = agent_reg_loss + agent_cls_loss
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
        }

        return loss, disp_dict

    """
        Compute loss (single-agent or multi-agent)
    """
    def cal_loss(self, out, data, tag='', ma=False, apply_drift_loss=True):
        if ma: return self.ma_cal_loss(out, data, tag=tag)
        gt_len = data['target'][:, 0].shape[-2] # int((11 - data['timestamp'][0]).item() * 10)
        y_hat, pi, y_hat_others = out['y_hat'][:, :, :gt_len], out['pi'], out['y_hat_others'][:, :, :gt_len]
        assert_finite_tensor("y_hat", y_hat)
        assert_finite_tensor("pi", pi)
        new_y_hat = out.get('y_hat_single', None)
        if new_y_hat is not None: new_y_hat = new_y_hat[:, :, :gt_len]
        new_pi = out.get('pi_single', None)
        y, y_others = data['target'][:, 0], data['target'][:, 1:]
        target_sample_mask = self.target_sample_mask(data)

        configured_use_drift_loss = bool(getattr(self.model, 'use_drift_loss', False))
        use_drift_loss = configured_use_drift_loss and apply_drift_loss
        use_mdf = bool(getattr(self.model, 'use_mdf', False))
        use_endpoint_diversity = bool(getattr(self.model, 'use_endpoint_diversity', False))
        extra_training_loss_active = use_drift_loss or use_endpoint_diversity
        if use_mdf and not configured_use_drift_loss:
            rank_zero_warn('use_mdf=True has no effect when use_drift_loss=False')

        best_mode = select_winner(
            y_hat[..., :2],
            y,
            metric=getattr(self.model, 'winner_metric', 'l2_sum'),
            fde_weight=getattr(self.model, 'winner_fde_weight', 1.0),
        )
        y_hat_best = y_hat[torch.arange(y_hat.shape[0]), best_mode]
        zero_loss = y_hat.sum() * 0.0
        trajectory_reg_loss = (
            F.smooth_l1_loss(y_hat_best[target_sample_mask, ..., :2], y[target_sample_mask])
            if target_sample_mask.any()
            else zero_loss
        )
        endpoint_reg_loss = (
            F.smooth_l1_loss(y_hat_best[target_sample_mask, -1, :2], y[target_sample_mask, -1])
            if target_sample_mask.any()
            else zero_loss
        )
        endpoint_reg_weight = getattr(self.model, 'endpoint_reg_weight', 0.0)
        agent_reg_loss = trajectory_reg_loss + endpoint_reg_weight * endpoint_reg_loss
        label_smoothing = (
            getattr(self.model, 'label_smoothing', 0.0)
            if getattr(self.model, 'use_label_smoothing_ce', False) and extra_training_loss_active
            else 0.0
        )
        agent_cls_loss = (
            F.cross_entropy(
                pi[target_sample_mask],
                best_mode[target_sample_mask].detach(),
                label_smoothing=label_smoothing,
            )
            if target_sample_mask.any()
            else zero_loss
        )

        if new_y_hat is not None:
            new_best_mode = select_winner(
                new_y_hat[..., :2],
                y,
                metric=getattr(self.model, 'winner_metric', 'l2_sum'),
                fde_weight=getattr(self.model, 'winner_fde_weight', 1.0),
            )
            new_y_hat_best = new_y_hat[torch.arange(new_y_hat.shape[0]), new_best_mode]
            new_trajectory_reg_loss = (
                F.smooth_l1_loss(
                    new_y_hat_best[target_sample_mask, ..., :2], y[target_sample_mask]
                )
                if target_sample_mask.any()
                else zero_loss
            )
            new_endpoint_reg_loss = (
                F.smooth_l1_loss(
                    new_y_hat_best[target_sample_mask, -1, :2], y[target_sample_mask, -1]
                )
                if target_sample_mask.any()
                else zero_loss
            )
            new_agent_reg_loss = (
                new_trajectory_reg_loss + endpoint_reg_weight * new_endpoint_reg_loss
            )
            new_agent_cls_loss = (
                F.cross_entropy(
                    new_pi[target_sample_mask],
                    new_best_mode[target_sample_mask].detach(),
                    label_smoothing=label_smoothing,
                )
                if target_sample_mask.any()
                else zero_loss
            )
        else:
            new_agent_reg_loss = zero_loss
            new_agent_cls_loss = zero_loss

        others_reg_mask = data['target_mask'][:, 1:]
        if getattr(self.model, 'vehicle_only', False):
            others_reg_mask = others_reg_mask & (data['x_attr'][:, 1:, 2:3].long() == 0)
        others_reg_loss = (
            F.smooth_l1_loss(y_hat_others[others_reg_mask], y_others[others_reg_mask])
            if others_reg_mask.any()
            else zero_loss
        )

        extra_loss = y_hat.new_zeros(())
        extra_disp_dict = {}
        current_epoch = int(getattr(self, 'current_epoch', 0))
        use_loss_weight_schedule = bool(getattr(self.model, 'use_loss_weight_schedule', True))
        if use_drift_loss:
            radii = getattr(self.model, 'mdf_r_list', [0.1]) if use_mdf else [getattr(self.model, 'drift_single_radius', 0.1)]
            drift_loss_type = getattr(self.model, 'drift_loss_type', 'official')
            if drift_loss_type == 'joint_official':
                drift_loss_value, drift_stats = joint_drifting_loss(
                    predictions=y_hat[..., :2],
                    ground_truth=y,
                    scene_feature=out['drift_scene_feature'],
                    valid_scene_mask=target_sample_mask,
                    r_list=radii,
                    num_waypoints=getattr(self.model, 'drift_num_waypoints', 10),
                    endpoint_weight=getattr(self.model, 'drift_endpoint_weight', 2.0),
                    context_alpha=getattr(self.model, 'drift_context_alpha', 1.0),
                )
            else:
                winner, others = split_winner_and_others(y_hat[..., :2], best_mode)
                if target_sample_mask.any():
                    drift_loss_value, drift_stats = drifting_loss(
                        gen=winner[target_sample_mask],
                        fixed_pos=y[target_sample_mask],
                        fixed_neg=others[target_sample_mask],
                        r_list=radii,
                        use_mdf=use_mdf,
                        include_old_gen_as_neg=getattr(self.model, 'mdf_include_old_gen_as_neg', True),
                        normalize_force=getattr(self.model, 'mdf_normalize_force', True),
                        pos_weight=getattr(self.model, 'mdf_positive_weight', 1.0),
                        neg_weight=getattr(self.model, 'mdf_negative_weight', 1.0),
                        force_clip=getattr(self.model, 'drift_force_clip', 1.0),
                        detach_target=getattr(self.model, 'drift_detach_target', True),
                        space=getattr(self.model, 'drift_space', 'traj'),
                        normalize_space=getattr(self.model, 'drift_normalize_space', True),
                        loss_type=drift_loss_type,
                        soft_tau=getattr(self.model, 'drift_soft_tau', 0.05),
                        error_gate=getattr(self.model, 'drift_error_gate', 1.0),
                        protect_gt_direction=getattr(self.model, 'drift_protect_gt_direction', True),
                    )
                else:
                    drift_loss_value = zero_loss
                    drift_stats = {}
            drift_base_w = getattr(self.model, 'drift_weight', 0.0)
            drift_w = (
                scheduled_weight(
                    drift_base_w,
                    current_epoch,
                    getattr(self.model, 'drift_warmup_epochs', 0),
                    getattr(self.model, 'auxiliary_decay_start_epoch', None),
                    getattr(self.model, 'auxiliary_decay_end_epoch', None),
                )
                if use_loss_weight_schedule
                else float(drift_base_w)
            )
            extra_loss = extra_loss + drift_w * drift_loss_value
            extra_disp_dict[f'{tag}drift_loss'] = drift_loss_value.item()
            extra_disp_dict[f'{tag}drift_weight'] = drift_w
            extra_disp_dict[f'{tag}drift_weighted_loss'] = (drift_w * drift_loss_value).item()
            extra_disp_dict.update({
                f'{tag}drift_{name}': value.item()
                for name, value in drift_stats.items()
            })

        if use_endpoint_diversity:
            diversity_loss = endpoint_diversity_loss(
                y_hat[..., :2],
                sigma=getattr(self.model, 'diversity_sigma', 2.0),
                valid_mask=target_sample_mask,
            )
            diversity_base_w = getattr(self.model, 'diversity_weight', 0.0)
            diversity_w = (
                scheduled_weight(
                    diversity_base_w,
                    current_epoch,
                    getattr(self.model, 'diversity_warmup_epochs', 0),
                    getattr(self.model, 'auxiliary_decay_start_epoch', None),
                    getattr(self.model, 'auxiliary_decay_end_epoch', None),
                )
                if use_loss_weight_schedule
                else float(diversity_base_w)
            )
            extra_loss = extra_loss + diversity_w * diversity_loss
            extra_disp_dict.update({
                f'{tag}diversity_loss': diversity_loss.item(),
                f'{tag}diversity_weight': diversity_w,
            })

        dual_loss_weight = getattr(self.model, 'dual_loss_weight', 1.0)
        others_loss_weight = getattr(self.model, 'others_loss_weight', 1.0)
        loss = (
            agent_reg_loss
            + agent_cls_loss
            + others_loss_weight * others_reg_loss
            + dual_loss_weight * (new_agent_reg_loss + new_agent_cls_loss)
            + extra_loss
        )
        assert_finite_tensor("loss", loss)
        disp_dict = {
            f'{tag}loss': loss.item(),
            f'{tag}reg_loss': agent_reg_loss.item(),
            f'{tag}trajectory_reg_loss': trajectory_reg_loss.item(),
            f'{tag}endpoint_reg_loss': endpoint_reg_loss.item(),
            f'{tag}cls_loss': agent_cls_loss.item(),
            f'{tag}others_reg_loss': others_reg_loss.item(),
            f'{tag}others_loss_weight': others_loss_weight,
            f'{tag}dual_reg_loss': new_agent_reg_loss.item(),
            f'{tag}dual_cls_loss': new_agent_cls_loss.item(),
            f'{tag}dual_loss_weight': dual_loss_weight,
        }
        if 'gmp_comp_idx' in out:
            comp_idx = out['gmp_comp_idx']
            counts = torch.bincount(comp_idx.reshape(-1), minlength=int(comp_idx.max().item()) + 1).float()
            probs = counts / counts.sum().clamp_min(1)
            entropy = -(probs * probs.clamp_min(1e-8).log()).sum()
            disp_dict[f'{tag}gmp_component_entropy'] = entropy.item()
        disp_dict.update(extra_disp_dict)

        return loss, disp_dict

    def training_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        loss, loss_dict = self.cal_loss(out, data)

        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(data["scenario_id"])
            )

        return loss
    
    def on_validation_end(self) -> None:
        self.curr_ep += 1

    def on_validation_start(self) -> None:
        self.reset_val_scores()

    def validation_step(self, data, batch_idx):
        if isinstance(data, list):
            data = data[-1]

        out = self(data)

        _, loss_dict = self.cal_loss(out, data, apply_drift_loss=False)
        if self.ma:
            metrics = self.metrics(out, data['target'], data["scored_mask"])
        else:
            metrics = self.metrics(
                out,
                data['target'][:, 0],
                sample_mask=self.target_sample_mask(data),
            )

        self.log(
            'val/reg_loss',
            loss_dict['reg_loss'],
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
            batch_size=len(data["scenario_id"])
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=len(data["scenario_id"])
        )
        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())
        return [out]


    def on_test_start(self) -> None:
        save_dir = Path('./submission')
        save_dir.mkdir(exist_ok=True)
        if self.ma:
            from src.utils.ma_submission_av2 import SubmissionAv2 as SubmissionHandler
        else:
            #from src.utils.submission_av2 import SubmissionAv2 as SubmissionHandler
            from src.utils.submission_av1 import SubmissionAv1 as SubmissionHandler
        self.submission_handler = SubmissionHandler(
            save_dir=save_dir
        )

    def test_step(self, data, batch_idx) -> None:
        if isinstance(data, list):
            data = data[-1]
        out = self(data)
        self.submission_handler.format_data(data, out['y_hat'], out['pi'])

    def on_test_end(self) -> None:
        self.submission_handler.generate_submission_file()

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            nn.Linear,
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.MultiheadAttention,
            nn.LSTM,
            nn.GRU,
        )
        blacklist_weight_modules = (
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.SyncBatchNorm,
            nn.LayerNorm,
            nn.Embedding,
        )
        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                full_param_name = (
                    '%s.%s' % (module_name, param_name) if module_name else param_name
                )
                if 'bias' in param_name:
                    no_decay.add(full_param_name)
                elif 'weight' in param_name:
                    if isinstance(module, whitelist_weight_modules):
                        decay.add(full_param_name)
                    elif isinstance(module, blacklist_weight_modules):
                        no_decay.add(full_param_name)
                elif not ('weight' in param_name or 'bias' in param_name):
                    no_decay.add(full_param_name)
        param_dict = {
            param_name: param for param_name, param in self.named_parameters() if param.requires_grad
        }
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0
        #assert len(param_dict.keys() - union_params) == 0

        optim_groups = [
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(decay))
                ],
                'weight_decay': self.optim.weight_decay,
            },
            {
                'params': [
                    param_dict[param_name] for param_name in sorted(list(no_decay))
                ],
                'weight_decay': 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            optim_groups, lr=self.optim.lr, weight_decay=self.optim.weight_decay
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            lr=self.optim.lr,
            min_lr=self.optim.min_lr,
            warmup_ratio=self.optim.warmup_ratio,
            epochs=self.optim.epochs,
        )
        return [optimizer], [scheduler]


class StreamLightningModule(BaseLightningModule):
    def __init__(self,
                 num_grad_frame=3,
                 step_loss_weights=None,
                 final_step_only_start_epoch=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_grad_frame = num_grad_frame
        self.step_loss_weights = (
            None if step_loss_weights is None else [float(weight) for weight in step_loss_weights]
        )
        self.final_step_only_start_epoch = (
            None if final_step_only_start_epoch is None else int(final_step_only_start_epoch)
        )
        if self.step_loss_weights is not None and min(self.step_loss_weights) < 0:
            raise ValueError("step_loss_weights must be non-negative")

        # initialize multi-agent consistency module
        from .ma_consistency_module import GlobalConsistencyModule 
        if self.ma and not self.ma_eval_only:
            self.consitency_module = GlobalConsistencyModule(future_steps=100)
        return

    def training_step(self, data, batch_idx):
        total_step = len(data)
        num_grad_frames = min(self.num_grad_frame, total_step)
        num_no_grad_frames = total_step - num_grad_frames
        if self.step_loss_weights is not None and len(self.step_loss_weights) != total_step:
            raise ValueError(
                f"step_loss_weights must have {total_step} entries, got {len(self.step_loss_weights)}"
            )

        # iterate over all split points which does not require grad
        memory_dict = None
        self.eval()
        with torch.no_grad():
            for i in range(num_no_grad_frames):
                cur_data = data[i]
                cur_data['memory_dict'] = memory_dict
                out = self(cur_data)
                memory_dict = out['memory_dict']

        # iterate over all split points which require grad        
        self.train()
        sum_loss = 0
        loss_dict = {}
        for i in range(num_grad_frames):
            step_idx = i + num_no_grad_frames
            cur_data = data[step_idx]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            apply_drift_loss = (
                not getattr(self.model, 'drift_final_step_only', True)
                or i == num_grad_frames - 1
            )
            cur_loss, cur_loss_dict = self.cal_loss(
                out,
                cur_data,
                tag=f'step{step_idx}_',
                apply_drift_loss=apply_drift_loss,
            )
            loss_dict.update(cur_loss_dict)
            memory_dict = out['memory_dict']
            step_weight = (
                self.step_loss_weights[step_idx]
                if self.step_loss_weights is not None else 1.0
            )
            if (
                self.final_step_only_start_epoch is not None
                and self.current_epoch >= self.final_step_only_start_epoch
                and step_idx != total_step - 1
            ):
                step_weight = 0.0
            loss_dict[f'step{step_idx}_loss_weight'] = step_weight

            # apply multi-agent consistency
            if self.ma and self.ma_loss_each:
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consitency_module(ma_input)
                cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
                cur_data["target"] = ma_input["target"]
                ma_loss, __ = self.cal_loss(scene_out, cur_data, tag=f'step{step_idx}_', ma=True)
                step_loss = ma_loss * 0.9 + cur_loss * 1.1
                
                # stream predictions after consistency module to the following step
                if self.stream_the_scene:
                    y_hat = scene_out["y_hat"].permute(0, 2, 1, 3, 4)[ma_input["x_key_valid_mask"]]
                    B = y_hat.shape[0]
                    glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(memory_dict["rot_mat"]))
                    memory_dict["glo_y_hat"] = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)
                    memory_dict["x_mode"] = scene_out["x_mode"].permute(0, 2, 1, 3)[ma_input["x_key_valid_mask"]]
            else:
                step_loss = cur_loss
            sum_loss += step_weight * step_loss

        # unused
        if self.ma and not self.ma_loss_each:
            ma_input = self.getScenario(out, cur_data)
            scene_out = self.consitency_module(ma_input)
            cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
            cur_data["target"] = ma_input["target"]
            ma_loss, __ = self.cal_loss(scene_out, cur_data, tag=f'step{i + num_no_grad_frames}_', ma=True)
            sum_loss = ma_loss * 0.9 + sum_loss * 0.1    

        loss_dict['loss'] = sum_loss.item()
        for k, v in loss_dict.items():
            self.log(
                f'train/{k}',
                v,
                on_step=True,
                on_epoch=True,
                prog_bar=False,
                sync_dist=True,
                batch_size=len(data[-1]["scenario_id"])
            )
        return sum_loss

    def validation_step(self, data, batch_idx):
        memory_dict = None
        reg_loss_dict = {}
        all_outs = []

        self.eval()

        # iterate over all split points
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            _, cur_loss_dict = self.cal_loss(out, cur_data, tag=f'step{i}_', apply_drift_loss=False)
            reg_loss_dict[f'val/step{i}_reg_loss'] = cur_loss_dict[f'step{i}_reg_loss']
            memory_dict = out['memory_dict']
            
            # stream predictions after consistency module to the following step
            if self.ma and self.stream_the_scene:
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consitency_module(ma_input)
                cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
                cur_data["target"] = ma_input["target"]

                y_hat = scene_out["y_hat"].permute(0, 2, 1, 3, 4)[ma_input["x_key_valid_mask"]]
                B = y_hat.shape[0]
                glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(memory_dict["rot_mat"]))
                memory_dict["glo_y_hat"] = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)
                memory_dict["x_mode"] = scene_out["x_mode"].permute(0, 2, 1, 3)[ma_input["x_key_valid_mask"]]
    
            all_outs.append(out)
        
        # unused
        if self.ma and not self.stream_the_scene:
            ma_input = self.getScenario(out, cur_data)
            if not self.ma_eval_only:
                scene_out = self.consitency_module(ma_input)
            else:
                pi, indices = torch.sort(ma_input["pis"].permute(0, 2, 1), dim=1, descending=True)  
                y_hats = ma_input["y_hats"].permute(0, 2, 1, 3, 4)
                _, _, _, T, D = y_hats.shape
                indices_expanded = indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, T, D)
                y_hat = torch.gather(y_hats, dim=1, index=indices_expanded)
                pi = pi.mean(-1)
                scene_out = {"y_hat": y_hat, "pi": pi}

            cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
            cur_data["target"] = ma_input["target"]

        # compute single- or multi-agent metrics
        if self.ma:
            gt_len = ma_input['target'].shape[-2] 
            scene_out["y_hat"] = scene_out["y_hat"][:, :, :, :gt_len]
            metrics = self.metrics(scene_out, ma_input['target'], ma_input["x_key_valid_mask"])
        else:
            gt_len = data[-1]['target'][:, 0].shape[-2] 
            all_outs[-1]["y_hat"] = all_outs[-1]["y_hat"][:, :, :gt_len]

            metrics = self.metrics(
                all_outs[-1],
                data[-1]['target'][:, 0],
                sample_mask=self.target_sample_mask(data[-1]),
            )

        self.log_dict(
            reg_loss_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            sync_dist=True,
        )
        self.log_dict(
            metrics,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=1,
            sync_dist=True,
        )
        for k in self.val_scores.keys(): self.val_scores[k].append(metrics[k].item())
        return all_outs
    
    def test_step(self, data, batch_idx) -> None:
        memory_dict = None
        all_outs = []

        # iterate over all split points
        for i in range(len(data)):
            cur_data = data[i]
            cur_data['memory_dict'] = memory_dict
            out = self(cur_data)
            memory_dict = out['memory_dict']

            # stream predictions after consistency module to the following step
            if self.ma and self.stream_the_scene:
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consitency_module(ma_input)
                cur_data["scored_mask"] = ma_input["x_key_valid_mask"]
                cur_data["target"] = ma_input["target"]

                y_hat = scene_out["y_hat"].permute(0, 2, 1, 3, 4)[ma_input["x_key_valid_mask"]]
                B = y_hat.shape[0]
                glo_y_hat = torch.bmm(y_hat.detach().reshape(B, -1, 2), torch.inverse(memory_dict["rot_mat"]))
                memory_dict["glo_y_hat"] = glo_y_hat.reshape(B, y_hat.size(1), -1, 2)
                memory_dict["x_mode"] = scene_out["x_mode"].permute(0, 2, 1, 3)[ma_input["x_key_valid_mask"]]

            all_outs.append(out)

        # provide single- or multi-agent predictions
        if self.ma:
            # unused
            if not self.stream_the_scene:
                ma_input = self.getScenario(out, cur_data)
                scene_out = self.consitency_module(ma_input)
            self.submission_handler.format_data(ma_input, scene_out['y_hat'], scene_out['pi'])
        else:
            self.submission_handler.format_data(data[-1], all_outs[-1]['y_hat'][:, :, :60], all_outs[-1]['pi'])
    
    """
        Combine varying number of single-agent predictions in a scenario batch to per scene multi-agent inputs.
    """
    def getScenario(self, out, cur_data):
        if "bs_indices" not in cur_data.keys():
            ma_input = {
                'x_modes': out["memory_dict"]["x_mode"].unsqueeze(0),
                'origins': cur_data["origin"].unsqueeze(0),
                'thetas':  cur_data["theta"].unsqueeze(0),
                'x_key_valid_mask': torch.ones((1,  out["memory_dict"]["x_mode"].shape[0]), dtype=bool).cuda(),
                'y_hats': out["y_hat"],
                'pis': out["pi"]
            }
        else:
            from torch.nn.utils.rnn import pad_sequence
            bs_indices = torch.tensor(cur_data["bs_indices"])
            b = torch.unique(bs_indices).shape[0]

            x_modes = [out["memory_dict"]["x_mode"][bs_indices == i] for i in range(b)]
            x_modes = pad_sequence(x_modes, batch_first=True) 
            x_encoder = [out["memory_dict"]["x_encoder"][bs_indices == i] for i in range(b)]
            x_encoder = pad_sequence(x_encoder, batch_first=True) 
            origins = [cur_data["origin"][bs_indices == i] for i in range(b)]
            origins = pad_sequence(origins, batch_first=True) 
            thetas = [cur_data["theta"][bs_indices == i] for i in range(b)]
            thetas = pad_sequence(thetas, batch_first=True)    
            x_key_valid_mask = [cur_data["x_key_valid_mask"][:, 0][bs_indices == i] for i in range(b)]
            x_key_valid_mask = pad_sequence(x_key_valid_mask, batch_first=True,  padding_value=False)   

            y_hats = [out["y_hat"][bs_indices == i] for i in range(b)]
            y_hats = pad_sequence(y_hats, batch_first=True)   
            pi = [out["pi"][bs_indices == i] for i in range(b)]
            pi = pad_sequence(pi, batch_first=True)   

            if "target" in cur_data.keys():
                target = [cur_data["target"][:, 0][bs_indices == i] for i in range(b)]
                target = pad_sequence(target, batch_first=True)   
            else:
                target = torch.zeros_like(y_hats[:, :, 0])

            #print(x_modes.shape, origins.shape, thetas.shape, y_hats.shape, x_modes.shape, x_key_valid_mask.shape)

            ma_input = {
                'x_modes': x_modes,
                'origins': origins,
                'thetas':  thetas,
                'x_key_valid_mask': x_key_valid_mask,
                'y_hats': y_hats,
                'pis': pi,
                'target': target,
                'x_encoder': x_encoder,
                'scenario_id': list(dict.fromkeys(cur_data["scenario_id"])),
                'track_id': [ np.array(cur_data["track_id"])[bs_indices.numpy() == i].tolist() for i in range(b) ]
            }
        return ma_input
