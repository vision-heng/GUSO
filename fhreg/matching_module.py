# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from loguru import logger

INF = 1e9


def mask_border(m, b: int, v):
    """
    Mask tensor borders with a specific value.
    Args:
        m (torch.Tensor): Input tensor [N, H0, W0, H1, W1]
        b (int): Border width
        v (m.dtype): Value to fill
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def generate_random_mask(n, num_true):
    """
    Create a boolean mask with a fixed number of randomly placed True values.
    """
    mask = torch.zeros(n, dtype=torch.bool)
    indices = torch.randperm(n)[:num_true]
    mask[indices] = True
    return mask


class CoarseMatching(nn.Module):
    """
    Coarse-level Matching Module.
    Responsible for identifying global correspondences between feature grids 
    using Dual Softmax and thresholding.
    """
    def __init__(self, config, profiler, model_training=True):
        super().__init__()
        self.config = config
        self.profiler = profiler
        self.model_training = model_training
        
        # Configuration parameters
        self.thr = config['thr']
        self.use_sm = config['use_sm']
        self.inference = config['inference']
        self.border_rm = config['border_rm']
        self.temperature = config['dsmax_temperature']
        
        # Final linear projection layer
        self.final_proj = nn.Linear(256, 256, bias=True)

    def forward(self, feat_c0, feat_c1, data):
        """
        Args:
            feat_c0, feat_c1: [Batch, L, C] Normalized feature tensors.
            data: Data dictionary containing spatial shape information.
        """
        # ------------------------------------------------------------------
        # Part 1: Feature Projection and Similarity Computation
        # ------------------------------------------------------------------
        feat_c0 = self.final_proj(feat_c0)
        feat_c1 = self.final_proj(feat_c1)

        # Scale normalization for dot product attention
        scale = feat_c0.shape[-1] ** 0.5
        feat_c0 = feat_c0 / scale
        feat_c1 = feat_c1 / scale

        # Compute Similarity Matrix: [B, L0, L1]
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_c0, feat_c1) / self.temperature
        
        # ------------------------------------------------------------------
        # Part 2: Dual Softmax and Selection Strategy
        # ------------------------------------------------------------------
        axes = {
            'h0c': data['hw0_c'][0], 'w0c': data['hw0_c'][1],
            'h1c': data['hw1_c'][0], 'w1c': data['hw1_c'][1]
        }
        
        if self.inference:
            with torch.no_grad():
                # Perform Mutual Nearest Neighbor (MNN) check
                # Forward Softmax (Image 0 to 1)
                conf_matrix = F.softmax(sim_matrix, 2) if self.use_sm else sim_matrix
                mask = (conf_matrix > self.thr) * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0])
                
                # Backward Softmax (Image 1 to 0)
                conf_matrix = F.softmax(sim_matrix, 1) if self.use_sm else sim_matrix
                mask |= (conf_matrix > self.thr) * (conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])

                # Boundary removal
                mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes)  
                mask_border(mask, self.border_rm, False)
                mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)')  

                # Extract matched indices
                b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
                mconf = sim_matrix[b_ids, i_ids, j_ids] 

                coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids, 'mconf': mconf}
            
        else:
            # Training Mode: Preserve Softmax matrices for loss computation
            conf_0to1 = F.softmax(sim_matrix, 2)
            conf_1to0 = F.softmax(sim_matrix, 1)
            data.update({'conf_matrix_0_to_1': conf_0to1, 'conf_matrix_1_to_0': conf_1to0})

            with torch.no_grad():
                # Generate MNN Mask
                mask = (conf_0to1 > self.thr) * (conf_0to1 == conf_0to1.max(dim=2, keepdim=True)[0])
                mask |= (conf_1to0 > self.thr) * (conf_1to0 == conf_1to0.max(dim=1, keepdim=True)[0])

                # Apply border constraints
                mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes) 
                mask_border(mask, self.border_rm, False)
                mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)')   

                # Extract matches and combine confidence scores
                b_ids, i_ids, j_ids = mask.nonzero(as_tuple=True)
                mconf = torch.maximum(conf_0to1[b_ids, i_ids, j_ids], conf_1to0[b_ids, i_ids, j_ids])

                # Training Sampling Logic: Select subset of matches to control memory and loss balance
                if self.model_training:
                    num_candidates_max = max(mask.size(1), mask.size(2))
                    num_train = int(num_candidates_max * self.config['train_coarse_percent']) * mask.size(0)
                    num_pred = len(b_ids)
                    min_pad = self.config['train_pad_num_gt_min']
                    
                    # Sample predicted indices
                    if num_pred <= num_train - min_pad:
                        idx_pred = torch.arange(num_pred, device=mask.device)
                    else:
                        idx_pred = torch.randint(num_pred, (num_train - min_pad,), device=mask.device)

                    # Sample Ground Truth (GT) indices for padding
                    pad_len = max(num_train - num_pred, min_pad)
                    idx_gt = torch.randint(len(data['spv_b_ids']), (pad_len,), device=mask.device)
                    mconf_gt = torch.zeros(pad_len, device=mask.device) # GT padding confidence set to 0

                    # Concatenate predicted and GT padding indices
                    b_ids = torch.cat([b_ids[idx_pred], data['spv_b_ids'][idx_gt]])
                    i_ids = torch.cat([i_ids[idx_pred], data['spv_i_ids'][idx_gt]])
                    j_ids = torch.cat([j_ids[idx_pred], data['spv_j_ids'][idx_gt]])
                    mconf = torch.cat([mconf[idx_pred], mconf_gt])

                coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids, 'mconf': mconf}

        # ------------------------------------------------------------------
        # Part 3: Coordinate Reconstruction
        # ------------------------------------------------------------------
        b_ids, i_ids, j_ids = coarse_matches['b_ids'], coarse_matches['i_ids'], coarse_matches['j_ids']
        mconf = coarse_matches['mconf']

        # Map indices from feature grid back to image pixels
        scale_base = data['hw0_i'][0] / data['hw0_c'][0]
        scale0 = scale_base * data['scale0'][b_ids] if 'scale0' in data else scale_base
        scale1 = scale_base * data['scale1'][b_ids] if 'scale1' in data else scale_base

        mkpts0_c = torch.stack([i_ids % axes['w0c'], i_ids // axes['w0c']], dim=1) * scale0
        mkpts1_c = torch.stack([j_ids % axes['w1c'], j_ids // axes['w1c']], dim=1) * scale1

        # Identify predicted points vs GT padded points for downstream tasks
        is_pred = (mconf != 0)

        if self.inference:
            coarse_matches.update({
                'm_bids': b_ids[is_pred], 
                'mkpts0_c': mkpts0_c,
                'mkpts1_c': mkpts1_c,
                'mconf': mconf[is_pred]
            })
        else:
            coarse_matches.update({
                'gt_mask': ~is_pred,
                'm_bids': b_ids[is_pred], 
                'mkpts0_c': mkpts0_c[is_pred],
                'mkpts1_c': mkpts1_c[is_pred],
                'mkpts0_c_train': mkpts0_c,
                'mkpts1_c_train': mkpts1_c,
                'mconf': mconf[is_pred]
            })

        data.update(coarse_matches)
        return data


class FineSubMatching(nn.Module):
    """
    Fine-level and Sub-pixel Matching Module.
    Refines coarse matches to sub-pixel accuracy within local feature windows.
    """
    def __init__(self, config, profiler, model_training=True):
        super().__init__()
        self.config = config
        self.profiler = profiler
        self.model_training = model_training

        # Configuration parameters
        self.W = config['fine_window_size']
        self.temperature = config['fine']['dsmax_temperature']
        self.thr = config['fine']['thr']
        self.inference = config['fine']['inference']
        self.fine_spv_max = None 

        # Sub-pixel refinement layers
        dim_f = 64
        self.fine_proj = nn.Linear(dim_f, dim_f, bias=False)
        self.subpixel_mlp = nn.Sequential(
            nn.Linear(2 * dim_f, 2 * dim_f, bias=False),
            nn.ReLU(),
            nn.Linear(2 * dim_f, 4, bias=False)
        )

    def forward(self, feat_f0_unfold, feat_f1_unfold, data):
        """
        Args:
            feat_f0_unfold, feat_f1_unfold: [M, W*W, C] Cropped local window features.
            data: Metadata dictionary with coarse match indices.
        """
        M, WW, C = feat_f0_unfold.shape
        W = self.W

        # ------------------------------------------------------------------
        # Step 0: Boundary Case Handling
        # ------------------------------------------------------------------
        if M == 0:
            assert not self.model_training, "Training must include padded matches."
            empty_out = {
                'mkpts0_f': data['mkpts0_c'],
                'mkpts1_f': data['mkpts1_c'],
                'mconf_f': torch.zeros(0, device=feat_f0_unfold.device),
            }
            if not self.inference:
                empty_out.update({
                    'mkpts0_f_train': data['mkpts0_c_train'],
                    'mkpts1_f_train': data['mkpts1_c_train'],
                    'conf_matrix_fine': torch.zeros(1, WW, WW, device=feat_f0_unfold.device),
                    'b_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'i_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                    'j_ids_fine': torch.zeros(0, device=feat_f0_unfold.device),
                })
            data.update(empty_out)
            return

        # ------------------------------------------------------------------
        # Step 1: Feature Projection and Similarity Matrix
        # ------------------------------------------------------------------
        # FIX: Ensure variable names match the input 'feat_f0_unfold'
        feat_f0 = self.fine_proj(feat_f0_unfold)
        feat_f1 = self.fine_proj(feat_f1_unfold)

        # Scale-normalization
        scale = C ** 0.5
        feat_f0 = feat_f0 / scale  # Fixed: changed from feat_c0 to feat_f0
        feat_f1 = feat_f1 / scale  # Fixed: changed from feat_c1 to feat_f1

        # Compute Window-level Similarity Matrix: [M, WW, WW]
        sim_matrix = torch.einsum("nlc,nsc->nls", feat_f0, feat_f1) / self.temperature
        conf_matrix = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)

        # ------------------------------------------------------------------
        # Step 2: Fine-level Selection (Inference/Visualization logic)
        # ------------------------------------------------------------------
        with torch.no_grad():
            mask = conf_matrix > self.thr
            if mask.sum() == 0:
                mask[0, 0, 0] = True
                conf_matrix[0, 0, 0] = 1.0

            # Keep only the global maximum within each window
            mask = mask & (conf_matrix == conf_matrix.amax(dim=(1, 2), keepdim=True))

            mask_v, all_j_ids = mask.max(dim=2)
            b_ids_local, i_ids_local = torch.where(mask_v)
            j_ids_local = all_j_ids[b_ids_local, i_ids_local]
            mconf = conf_matrix[b_ids_local, i_ids_local, j_ids_local]

            # Reconstruct global coordinates
            b_ids_coarse = data['b_ids'][b_ids_local]
            scale_f2c = data['hw0_f'][0] // data['hw0_c'][0] 
            
            # 1/2 resolution anchor points
            mkpts0_anchor = torch.stack([
                data['i_ids'][b_ids_local] % data['hw0_c'][1], 
                data['i_ids'][b_ids_local] // data['hw0_c'][1]
            ], dim=1) * scale_f2c
            
            mkpts1_anchor = torch.stack([
                data['j_ids'][b_ids_local] % data['hw1_c'][1], 
                data['j_ids'][b_ids_local] // data['hw1_c'][1]
            ], dim=1) * scale_f2c

            # Window relative coordinates
            mkpts0_window = torch.stack([i_ids_local % W, i_ids_local // W], dim=1)
            mkpts1_window = torch.stack([j_ids_local % W, j_ids_local // W], dim=1)

            scale_i2f = data['hw0_i'][0] / data['hw0_f'][0]
            scale0 = scale_i2f * data['scale0'][b_ids_coarse] if 'scale0' in data else scale_i2f
            scale1 = scale_i2f * data['scale1'][b_ids_coarse] if 'scale1' in data else scale_i2f

        # ------------------------------------------------------------------
        # Step 3: Sub-pixel Regression Refinement
        # ------------------------------------------------------------------
        feat_cat = torch.cat([
            feat_f0_unfold[b_ids_local, i_ids_local], 
            feat_f1_unfold[b_ids_local, j_ids_local]
        ], dim=-1)
        
        delta = torch.tanh(self.subpixel_mlp(feat_cat)) * 0.5 
        delta0, delta1 = delta.chunk(2, dim=1)

        # ------------------------------------------------------------------
        # Step 4: Final Coordinate Assembly
        # ------------------------------------------------------------------
        pad = W // 2 if W % 2 != 0 else 0
        base0 = (mkpts0_window + mkpts0_anchor - pad)
        base1 = (mkpts1_window + mkpts1_anchor - pad)

        mkpts0_final = (base0 + delta0) * scale0
        mkpts1_final = (base1 + delta1) * scale1

        # ------------------------------------------------------------------
        # Step 5: Update Results Dictionary
        # ------------------------------------------------------------------
        is_pred = (mconf != 0)
        fine_results = {
            'm_bids': b_ids_coarse[is_pred],
            'mkpts0_f': mkpts0_final[is_pred].detach(),
            'mkpts1_f': mkpts1_final[is_pred].detach(),
            'mconf_f': mconf[is_pred]
        }

        if not self.inference:
            # Training logic for supervision
            num_matches = len(data['b_ids'])
            train_mask = slice(None) if (self.fine_spv_max is None or self.fine_spv_max > num_matches) \
                         else generate_random_mask(num_matches, self.fine_spv_max)

            fine_results.update({
                'mkpts0_f_train': (base0 * scale0 + delta0 * scale0),
                'mkpts1_f_train': (base1 * scale1 + delta1 * scale1),
                'b_ids_fine': data['b_ids'][train_mask],
                'i_ids_fine': data['i_ids'][train_mask],
                'j_ids_fine': data['j_ids'][train_mask],
                'conf_matrix_fine': conf_matrix[train_mask]
            })

        data.update(fine_results)