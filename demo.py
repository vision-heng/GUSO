import os
import warnings

import cv2
import rasterio
import numpy as np
import torch
import argparse

# Filter unnecessary warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# utilities and configs
from fhreg.fhreg import FHReg as FHRegModel
from utils.cvpr_ds_config import default_cfg
from configs.test_config import test_cfg
from utils.fsc import FSC

# Configuration Constants
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
RMSE_MAX_LIMIT = 5
RMSE_SUCCESS = 3

class FHReg:
    """
    FHReg Inference Model
    """
    def __init__(self, imsize=512, match_threshold=0.1, no_match_upscale=False, ckpt=None, device='cuda'):
        self.device = device
        self.imsize = imsize
        self.match_threshold = match_threshold
        self.no_match_upscale = no_match_upscale

        # Model configuration and weight loading
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        test_cfg['coarse_thr'] = self.match_threshold
        test_cfg['fine_thr'] = self.match_threshold
        
        self.model = FHRegModel(conf, model_training=False)
        ckpt_dict = torch.load(ckpt, map_location=torch.device('cpu'))
        
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
            
        self.model.load_state_dict(ckpt_dict, strict=True)
        print(f"Loading weights from: {ckpt}")
        self.model = self.model.eval().to(self.device)

        # Identifier naming
        self.ckpt_name = ckpt.split('/')[-1].split('.')[0]
        self.name = f'FHReg_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'

    def match_inference(self, gray1, gray2):
        """
        Perform matching inference on image pairs
        """
        batch = {'image_opt': gray1, 'image_sar': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()
        return kpts1, kpts2


def draw_matches(img0, img1, kp0, kp1, path, color='green'):
    """
    Visualize matching results and save to file
    """
    import matplotlib.pyplot as plt
    kp0_cv = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
    kp1_cv = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(kp0_cv))]
    
    match_color = (0, 255, 0) if color == 'green' else (255, 0, 0)
    show = cv2.drawMatches(img1, kp1_cv, img0, kp0_cv, matches, None, matchColor=match_color)
    
    plt.imshow(show)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().set_frame_on(False)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()


def read_image_rasterio(path):
    """
    Read satellite/remote sensing imagery using Rasterio.
    Supports both single-band (grayscale) and multi-band images.
    """
    try:
        with rasterio.open(path) as src:
            img = src.read()  # Shape: (C, H, W)
            if img.shape[0] == 1:
                # Squeeze single band to (H, W)
                img = img[0]
            else:
                # Transpose multi-band to (H, W, C)
                img = np.transpose(img, (1, 2, 0))
        return img.astype('float32')
    except Exception as e:
        print(f"Failed to read image at {path}. Error: {e}")
        return None


def drawMatches(img0, img1, kp0, kp1, path, color='green'):
    import matplotlib.pyplot as plt
    kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
    kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
    matches = [cv2.DMatch(i, i, 1) for i in range(len(kp0))]
    matchColor = (0, 255, 0) if color == 'green' else (255, 0, 0)
    show = cv2.drawMatches(img1, kp1, img0, kp0, matches, None, matchColor=matchColor)
    plt.imshow(show)
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.gca().set_frame_on(False)
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


def kde(x, std = 0.1, half = True, down = None):
    # use a gaussian kernel to estimate density
    if half:
        x = x.half() # Do it in half precision TODO: remove hardcoding
    if down is not None:
        scores = (-torch.cdist(x,x[::down])**2/(2*std**2)).exp()
    else:
        scores = (-torch.cdist(x,x)**2/(2*std**2)).exp()
    density = scores.sum(dim=-1)
    return density


def sample_existing_points_with_kde(out_pts1, out_pts2, num=5000, std=0.1):
    """
    Apply KDE balanced sampling to existing point pairs.
    
    Args:
        out_pts1, out_pts2: numpy.ndarray [N, 2]
        num: target number of samples
        std: standard deviation for KDE
    Returns:
        sampled_pts1, sampled_pts2: numpy.ndarray [M, 2]
    """
    # 1. Convert Numpy to Torch Tensor
    pts1 = torch.from_numpy(out_pts1).float()
    pts2 = torch.from_numpy(out_pts2).float()
    
    # Combine into matches [N, 4]
    matches = torch.cat([pts1, pts2], dim=1) 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    matches = matches.to(device)

    # 3. Expansion Factor
    # We take all current points as candidates for the KDE balancing
    good_matches = matches

    # 4. Balanced Sampling via KDE
    use_half = True if device.type == "cuda" else False
    down = 1 if device.type == "cuda" else 8
    
    # Calculate spatial density in the coordinate space
    # 
    density = kde(good_matches, std=std, half=use_half, down=down)
    
    # Sampling probability: Inverse of density (1/density)
    # This penalizes points in over-populated areas
    p = 1 / (density + 1)
    
    # Avoid picking isolated noise (optional, adjust threshold as needed)
    # p[density < 5] = 1e-7 

    # 5. Final Multinomial Sampling
    num_to_sample = min(num, len(good_matches))
    balanced_samples_idx = torch.multinomial(p, 
                                            num_samples=num_to_sample, 
                                            replacement=False)
    
    final_matches = good_matches[balanced_samples_idx]

    # 6. Convert back to Numpy N*2
    final_matches_np = final_matches.detach().cpu().numpy()
    sampled_pts1 = final_matches_np[:, :2]
    sampled_pts2 = final_matches_np[:, 2:]

    return sampled_pts1, sampled_pts2


def main(args):
    print(f"Loading model from: {args.ckpt}")
    matching_model = FHReg(ckpt=args.ckpt, device=args.device)

    print(f"Reading images...")
    img_opt = read_image_rasterio(args.path_opt)
    img_sar = read_image_rasterio(args.path_sar)

    img_opt_tensor = torch.from_numpy(img_opt).unsqueeze(0).permute(0, 3, 1, 2).to(args.device).float()
    img_sar_tensor = torch.from_numpy(img_sar).unsqueeze(0).permute(0, 3, 1, 2).to(args.device).float()

    print("Running matching inference...")
    init_pts1, init_pts2 = matching_model.match_inference(img_opt_tensor, img_sar_tensor)

    print(f"Running FSC with {args.transform} model...")
    _, _, out_pts1, out_pts2 = FSC(init_pts1, init_pts2, args.transform, args.error_t, 'Point')
    print(f"Points after FSC: {len(out_pts1)}")

    print(f"Sampling {args.num_samples} points using KDE (std={args.kde_std})...")
    sampled_1, sampled_2 = sample_existing_points_with_kde(
        out_pts1, out_pts2, num=args.num_samples, std=args.kde_std
    )

    os.makedirs(args.out_dir, exist_ok=True)
    base_name1 = os.path.basename(args.path_opt).replace('.png', '')
    base_name2 = os.path.basename(args.path_sar).replace('.png', '')
    save_name = f"{base_name1}_{base_name2}match_fsc{len(out_pts1)}_kde{len(sampled_1)}.png"
    show_path = os.path.join(args.out_dir, save_name)

    vis_opt = img_opt.astype('uint8')
    vis_sar = img_sar.astype('uint8')

    drawMatches(vis_opt, vis_sar, sampled_1, sampled_2, show_path, color=args.color)
    print(f"Success! Result saved to: {show_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="FHReg: UHR Optical and SAR Image Registration Pipeline")

    parser.add_argument('--path_opt', type=str, default="guso_testpairs/18_opt.png", help='Path to the input Optical image')
    parser.add_argument('--path_sar', type=str, default="guso_testpairs/18_sar.png", help='Path to the input SAR image')
    
    parser.add_argument('--ckpt', type=str, default="weights/fhreg_guso.ckpt", help='Path to .ckpt weights')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')
    
    parser.add_argument('--transform', type=str, default='affine', choices=['affine', 'perspective', 'similarity'], help='Transformation model')
    parser.add_argument('--error_t', type=float, default=3.0, help='FSC error threshold (pixels)')

    parser.add_argument('--num_samples', type=int, default=100, help='Final number of points to sample')
    parser.add_argument('--kde_std', type=float, default=0.1, help='KDE bandwidth (higher for more uniform distribution)')
    
    parser.add_argument('--out_dir', type=str, default='./results', help='Directory to save output images')
    parser.add_argument('--color', type=str, default='green', help='Color of matching lines')

    args = parser.parse_args()
    main(args)


