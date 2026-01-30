import os
import time
import warnings
from datetime import datetime

import cv2
import numpy as np
import torch
from tqdm import tqdm
from prettytable import PrettyTable
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

# Filter unnecessary warnings
warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch")

# utilities and configs
from fhreg.fhreg import FHReg as FHRegModel
from utils.cvpr_ds_config import default_cfg
from configs.test_config import test_cfg
from utils.rs_metrics import compute_rmse_and_inliers
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
    plt.savefig(path, bbox_inches='tight', pad_inches=0, dpi=600)
    plt.close()


class Dataset(Dataset):
    def __init__(self, root='/data1/yh/work7_compare/FHRegGithub/msaw_testnpy'):
        self.root = root
        self.data_list = os.listdir(root)


    def __len__(self):
        print(f"Dataset -> {len(self.data_list)} samples")
        return len(self.data_list)

    def __getitem__(self, idx):
        npy_path = os.path.join(self.root, self.data_list[idx])
        loaded_data = np.load(npy_path, allow_pickle=True).item()
        
        rel_opt = loaded_data['ori_opt']
        rel_sar = loaded_data['ori_sar']
        rel_h = loaded_data['h_opt2sar']

        h_tensor = torch.from_numpy(rel_h)
        img0_tensor = TF.to_tensor(rel_opt.astype(np.uint8))
        img1_tensor = TF.to_tensor(rel_sar.astype(np.uint8))

        return {
            'image_opt': img0_tensor,
            'image_sar': img1_tensor,
            'H_0to1': h_tensor,
        }
    


if __name__ == '__main__':
    # Settings
    dataset_name = 'msaw'  # Options: 'msaw', 'os'
    ckpt_path = "weights/fhreg_guso.ckpt"
    dataset_root = f'{dataset_name}_testnpy'

    # Initialize model and dataloader
    matching_model = FHReg(ckpt=ckpt_path, device='cuda')
    test_dataset = Dataset(root=dataset_root)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, num_workers=2, pin_memory=True
    )

    # Initialize metrics
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader), desc=f'Zero-Shot FHReg')
    n_data = len(test_dataloader)
    n_success = 0
    total_rmse, total_time, total_ncm = 0, 0, 0
    total_recall, total_precision = 0, 0
    
    # Low-threshold RMSE counters
    lt1, lt2, lt3 = 0, 0, 0

    for idx, data in pbar:
        gray1 = data['image_opt'].cuda()
        gray2 = data['image_sar'].cuda()
        h_matrix = data['H_0to1'][0, :, :].numpy()

        # Inference and Timing
        start_time = time.time()
        kpts1, kpts2 = matching_model.match_inference(gray1, gray2)
        elapsed_time = time.time() - start_time

        init_pts1, init_pts2 = kpts1, kpts2

        if len(init_pts1) >= 3:
            # Outlier removal using FSC (Fast Sample Consensus)
            _, _, out_pts1, out_pts2 = FSC(init_pts1, init_pts2, 'affine', 3, 'Point')
            i_nom = len(out_pts1)

            # Calculate RMSE and Inliers
            i_rmse, clean_pts1, _ = compute_rmse_and_inliers(
                h_matrix, out_pts1, out_pts2, RMSE_MAX_LIMIT
            )
            
            # Threshold Statistics
            if i_rmse < 1: lt1 += 1
            if i_rmse < 2: lt2 += 1
            if i_rmse < 3: lt3 += 1

            i_ncm = len(clean_pts1)
            
            # Cap RMSE for averaging
            if i_rmse > RMSE_MAX_LIMIT:
                i_rmse = RMSE_MAX_LIMIT
            if i_rmse < RMSE_SUCCESS:
                n_success += 1

            # Precision Metric calculation
            i_precision = i_ncm / i_nom if i_nom > 0 else 0

            # Recall Metric calculation
            _, init_clean1, _ = compute_rmse_and_inliers(
                h_matrix, init_pts1, init_pts2, RMSE_MAX_LIMIT
            )
            i_nc = len(init_clean1)
            i_recall = i_ncm / i_nc if i_nc > 0 else 0
            
            i_time = elapsed_time

        else:
            # Handling cases with insufficient points
            i_time = elapsed_time
            i_rmse = RMSE_MAX_LIMIT
            i_ncm, i_recall, i_precision = 0, 0, 0

        # Accumulate metrics
        total_rmse += i_rmse
        total_time += i_time
        total_ncm += i_ncm
        total_recall += i_recall
        total_precision += i_precision

        # Update progress bar
        pbar.set_postfix(
            RMSE=f"{i_rmse:.2f}", Time=f"{i_time:.2f}", 
            Recall=f"{i_recall:.2f}", Prec=f"{i_precision:.2f}", NCM=f"{i_ncm:.2f}"
        )

    # Compute Final Averages
    success_rate = n_success / n_data
    avg_rmse = total_rmse / n_data
    avg_time = total_time / n_data
    avg_ncm = total_ncm / n_data
    avg_recall = total_recall / n_data
    avg_precision = total_precision / n_data
    
    lt1_ratio, lt2_ratio, lt3_ratio = lt1 / n_data, lt2 / n_data, lt3 / n_data

    # Output Table
    table = PrettyTable()
    table.field_names = ["SR", "RMSE", "Avg Time", "NCM", "Recall", "Precision"]
    table.add_row([
        f"{success_rate:.4f}", f"{avg_rmse:.4f}", f"{avg_time:.4f}", 
        f"{avg_ncm:.4f}", f"{avg_recall:.4f}", f"{avg_precision:.4f}"
    ])
    
    print(f'\n------------------------- Results: FHReg -------------------------')
    print(table)
    print(f"RMSE Accuracy Ratios -> <1px: {lt1_ratio:.4f} | <2px: {lt2_ratio:.4f} | <3px: {lt3_ratio:.4f}")

    # Log results to local file
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(f"{dataset_name}_others_result.txt", "a", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f'------------------------- Model: FHReg -------------------------\n')
        f.write(str(table) + "\n")
        f.write(f"RMSE <1px: {lt1_ratio:.4f} | <2px: {lt2_ratio:.4f} | <3px: {lt3_ratio:.4f}\n\n")