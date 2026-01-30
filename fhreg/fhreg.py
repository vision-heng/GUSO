import torch
from torch import nn
import torch.nn.functional as F
from einops.einops import rearrange

from utils.utils import (
    KeypointEncoder, 
    up_conv4, 
    MLPMixerEncoderLayer, 
    normalize_keypoints
)
from fhreg.matching_module import CoarseMatching, FineSubMatching
from utils.profiler import PassThroughProfiler
from fhreg.modalityharmonizer import ModalityHarmonizer, FourierPositionalEmbedding
from fhreg.wavelet_backbone import WFENet

# Ensure deterministic behavior for reproducibility
torch.backends.cudnn.deterministic = True
INF = 1e9


class FHReg(nn.Module):
    """
    FHReg: A Hierarchical Registration Network featuring Wavelet-enhanced features
    and a coarse-to-fine matching strategy.
    """
    def __init__(self, config, profiler=None, model_training=True):
        super().__init__()
        self.config = config
        self.profiler = profiler or PassThroughProfiler()
        self.d_model_c = self.config['coarse']['d_model']
        self.d_model_f = self.config['fine']['d_model']

        # Feature Extraction Backbone
        self.backbone = WFENet()

        # Positional and Keypoint Encoding
        self.keypoint_encoding = KeypointEncoder(self.d_model_c, [32, 64, 128, self.d_model_c])

        num_heads = 4
        descriptor_dim = 256
        add_scale_ori = False
        head_dim = descriptor_dim // num_heads
        
        # Cross-modality feature alignment
        self.modality_harmonizer = ModalityHarmonizer(descriptor_dim, num_heads, True)
        self.posenc = FourierPositionalEmbedding(2 + 2 * add_scale_ori, head_dim, head_dim)

        # Matching Modules
        self.coarse_matching = CoarseMatching(config['match_coarse'], self.profiler, model_training)
        self.fine_matching = FineSubMatching(config, self.profiler, model_training)

        # Feature Pyramid Network (FPN) for High-Resolution Refinement
        self.act = nn.GELU()
        dim = [256, 128, 64]
        self.up2 = up_conv4(dim[0], dim[1], dim[1])  # Upsample: 1/8 -> 1/4
        self.conv7a = nn.Conv2d(2 * dim[1], dim[1], kernel_size=3, padding=1)
        self.conv7b = nn.Conv2d(dim[1], dim[1], kernel_size=3, padding=1)
        
        self.up3 = up_conv4(dim[1], dim[2], dim[2])  # Upsample: 1/4 -> 1/2
        self.conv8a = nn.Conv2d(dim[2], dim[2], kernel_size=3, padding=1)
        self.conv8b = nn.Conv2d(dim[2], dim[2], kernel_size=3, padding=1)

        # Fine-grained Feature Enhancement (MLP-Mixer based)
        win_size = self.config['fine_window_size']
        self.fine_enc = nn.ModuleList([
            MLPMixerEncoderLayer(2 * win_size**2, 64) for _ in range(4)
        ])

    def forward(self, data, mode='test'):
        self.mode = mode
        
        # ------------------------------------------------------------------
        # Step 1: Wavelet-embedded Feature Extraction (WFE)
        # ------------------------------------------------------------------
        self.backbone(data)

        # Update metadata for hierarchical processing
        b, c, h8, w8 = data['bs'], data['c'], data['h_8'], data['w_8']
        data.update({
            'hw0_i': data['image_opt'].shape[2:],
            'hw1_i': data['image_sar'].shape[2:],
            'hw0_c': [h8, w8],
            'hw1_c': [h8, w8],
        })

        # ------------------------------------------------------------------
        # Step 2: Bidirectional Coarse Matching (BCM)
        # ------------------------------------------------------------------
        # Normalize keypoint grids and apply encoding
        kpts0 = normalize_keypoints(data['grid_8'], data['image_opt'].shape[-2:])
        kpts1 = normalize_keypoints(data['grid_8'], data['image_sar'].shape[-2:])
        
        # Inject Positional Encoding (PE) into flattened descriptors
        desc0 = data['feat_8_0'].flatten(2, 3) + self.keypoint_encoding(kpts0.transpose(1, 2))
        desc1 = data['feat_8_1'].flatten(2, 3) + self.keypoint_encoding(kpts1.transpose(1, 2))
        
        data.update({'feat_8_0': desc0, 'feat_8_1': desc1})

        # Feature Harmonization (Modal Alignment)
        if self.config['cem_type'] == 'transformer':
            self.modality_harmonizer(data)
        else:
            p_enc0, p_enc1 = self.posenc(kpts0), self.posenc(kpts1)
            # Harmonizer expects [B, C, L], necessitating dimension permutation
            desc0, desc1 = self.modality_harmonizer(
                desc0.transpose(1, 2), desc1.transpose(1, 2), p_enc0, p_enc1
            )
            data.update({
                'feat_8_0': desc0.transpose(1, 2), 
                'feat_8_1': desc1.transpose(1, 2)
            })

        # Execute global coarse matching
        self.coarse_matching(data['feat_8_0'].transpose(1, 2), data['feat_8_1'].transpose(1, 2), data)

        # ------------------------------------------------------------------
        # Step 3: Localized Fine Matching (LFM)
        # ------------------------------------------------------------------
        if self.config['matching_type'] == 'usm':
            # Skip fine matching for Unsupervised Matching (USM) mode
            data.update({'mkpts0_f': data['mkpts0_c'], 'mkpts1_f': data['mkpts1_c']})
            return data

        # 3.1 FPN-based High-Resolution Feature Recovery
        # ---------------------------------------------
        # Concatenate batches for efficient parallel processing: [2*B, C, H, W]
        feat_s8 = torch.cat([data['feat_8_0'], data['feat_8_1']], dim=0).view(2 * b, c, h8, -1)
        feat_s4 = torch.cat([data['feat_4_0'], data['feat_4_1']], dim=0)

        # Stage 1: Upsample 1/8 to 1/4 and fuse with skip connections
        feat_s4_up = self.up2(feat_s8) 
        feat_s4_refined = self.act(self.conv7a(torch.cat([feat_s4, feat_s4_up], dim=1)))
        feat_s4_final = self.act(self.conv7b(feat_s4_refined))

        # Stage 2: Upsample 1/4 to 1/2 for dense fine-matching
        feat_s2_up = self.up3(feat_s4_final)
        feat_fine = self.conv8b(self.act(self.conv8a(feat_s2_up)))

        feat_f0, feat_f1 = torch.chunk(feat_fine, 2, dim=0)
        data.update({'hw0_f': feat_f0.shape[2:], 'hw1_f': feat_f1.shape[2:]})

        # 3.2 Local Window Extraction (Unfolding)
        # ---------------------------------------------
        # Early exit if no coarse matches were found
        if data['b_ids'].shape[0] == 0:
            w_size = self.config['fine_window_size']
            empty = torch.empty(0, w_size**2, self.d_model_f, device=feat_s4.device)
            self.fine_matching(empty, empty, data)
            return data

        w_size = self.config['fine_window_size']
        data['resolution1'] = 8
        stride = 8 // self.config['resolution'][1]
        pad = 0 if w_size % 2 == 0 else w_size // 2

        def unfold_features(feat_map):
            # [N, C, H, W] -> [N, C*W*W, L] -> [N, L, W*W, C]
            unfolded = F.unfold(feat_map, kernel_size=(w_size, w_size), stride=stride, padding=pad)
            return rearrange(unfolded, 'n (c ww) l -> n l ww c', ww=w_size**2)

        feat_f0_unfold = unfold_features(feat_f0)
        feat_f1_unfold = unfold_features(feat_f1)

        # 3.3 Target Selection based on Coarse Matches
        # ---------------------------------------------
        feat_f0_selected = feat_f0_unfold[data['b_ids'], data['i_ids']]
        feat_f1_selected = feat_f1_unfold[data['b_ids'], data['j_ids']]

        # 3.4 Fine Transformer Refinement
        # ---------------------------------------------
        feat_merged = torch.cat([feat_f0_selected, feat_f1_selected], dim=1)
        feat_merged = feat_merged.transpose(1, 2)  # Prepare for [M, C, SeqLen] layers
        
        for layer in self.fine_enc:
            feat_merged = layer(feat_merged)
            
        feat_merged = feat_merged.transpose(1, 2)
        feat_f0_final = feat_merged[:, :w_size**2, :]
        feat_f1_final = feat_merged[:, w_size**2:, :]

        # Execute sub-pixel fine matching
        self.fine_matching(feat_f0_final, feat_f1_final, data)

        return data

    def load_state_dict(self, state_dict, *args, **kwargs):
        """
        Custom loader to handle naming discrepancies (e.g., removing 'matcher.' prefix).
        """
        for k in list(state_dict.keys()):
            if k.startswith('matcher.'):
                state_dict[k.replace('matcher.', '', 1)] = state_dict.pop(k)
        print('Weights successfully remapped and loaded.')
        return super().load_state_dict(state_dict, *args, **kwargs)