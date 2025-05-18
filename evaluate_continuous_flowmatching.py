import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import random

# Imports from your project (adjust paths if necessary)
from main_flowmatching_v2 import RGBNIRPairedDataset # Dataset from continuous FM script
from flow_matching_models import VectorFieldUNet, sample_with_ode_solver # Continuous FM model and sampler

# --- Configuration ---
# TODO: Fill these in based on user input or a config file
CHECKPOINT_PATH_CFM = "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/models/flow_matching_v2/flow_model_epoch_200.pth" # !!! USER NEEDS TO PROVIDE !!!
DATA_ROOT_DIR = "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons"
SPLIT = 'test' 
IMAGE_SIZE = 256 # Example, should match training for CFM
NUM_SAMPLES_TO_VISUALIZE = 6
BATCH_SIZE = 4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "evaluation_results/CFM"

# CFM Model specific parameters (Must match the trained model's config)
# TODO: Fill these based on user input or config
NIR_CHANNELS = 1
RGB_CHANNELS = 3
UNET_BASE_CHANNELS = 64
UNET_NUM_LEVELS = 4 
TIME_EMB_DIM = 256
CONTINUOUS_TIME_EMB_MAX_PERIOD = 1000.0 # Used by VectorFieldUNet

# CFM Sampler (ODE Solver) specific parameters
# TODO: Fill these based on user input or config
FM_T_END = 1.0 # Final time for integration
FM_SAMPLING_STEPS = 50 # Number of evaluation points for ODE solver
FM_SOLVER_METHOD = 'dopri5' # Default from training script, can be changed


# --- Helper Functions (Copied/adapted from previous evaluation scripts) ---
def denormalize_image(tensor):
    return (tensor + 1) / 2.0

def calculate_sam(img1_chw, img2_chw, eps=1e-8):
    if img1_chw.ndim == 2: img1_chw = img1_chw.unsqueeze(0)
    if img2_chw.ndim == 2: img2_chw = img2_chw.unsqueeze(0)
    if len(img1_chw.shape) == 2: img1_chw = img1_chw.unsqueeze(0)
    if len(img2_chw.shape) == 2: img2_chw = img2_chw.unsqueeze(0)
    img1_flat = img1_chw.view(img1_chw.size(0), -1)
    img2_flat = img2_chw.view(img2_chw.size(0), -1)
    if img1_flat.size(0) == 1:
        img1_vec = img1_flat.squeeze(0); img2_vec = img2_flat.squeeze(0)
        cos_angle = torch.dot(img1_vec, img2_vec) / (torch.norm(img1_vec) * torch.norm(img2_vec) + eps)
        return torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
    else:
        print("Warning: SAM for multi-channel unexpected. Flattening all.")
        img1_vec = img1_flat.reshape(-1); img2_vec = img2_flat.reshape(-1)
        cos_angle = torch.dot(img1_vec, img2_vec) / (torch.norm(img1_vec) * torch.norm(img2_vec) + eps)
        return torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))

def sobel_edges(image_bchw, eps=1e-6):
    if image_bchw.size(1) > 1: image_bchw = torch.mean(image_bchw, dim=1, keepdim=True)
    sobel_x_k = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)
    sobel_y_k = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)
    sobel_x = sobel_x_k.repeat(image_bchw.size(1),1,1,1); sobel_y = sobel_y_k.repeat(image_bchw.size(1),1,1,1)
    pad = (sobel_x_k.shape[-1]-1)//2
    grad_x = F.conv2d(image_bchw, sobel_x, padding=pad, stride=1, groups=image_bchw.size(1))
    grad_y = F.conv2d(image_bchw, sobel_y, padding=pad, stride=1, groups=image_bchw.size(1))
    edge_mag = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    for b in range(edge_mag.shape[0]):
        slice_ = edge_mag[b]; min_v=slice_.min(); max_v=slice_.max()
        edge_mag[b] = (slice_-min_v)/(max_v-min_v+eps)
    return edge_mag.clamp(0,1)

# --- Main Evaluation Function ---
def evaluate_cfm_checkpoint():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Model
    print("Initializing Continuous Flow Matching (CFM) model...")
    cfm_model = VectorFieldUNet(
        nir_channels=NIR_CHANNELS,
        rgb_channels=RGB_CHANNELS,
        out_channels_vector_field=NIR_CHANNELS,
        base_channels=UNET_BASE_CHANNELS,
        num_levels=UNET_NUM_LEVELS,
        time_emb_dim=TIME_EMB_DIM,
        continuous_time_emb_max_period=CONTINUOUS_TIME_EMB_MAX_PERIOD
    ).to(DEVICE)

    # 2. Load Checkpoint
    print(f"Loading CFM checkpoint from: {CHECKPOINT_PATH_CFM}")
    try:
        checkpoint_data = torch.load(CHECKPOINT_PATH_CFM, map_location=DEVICE)
        state_dict = checkpoint_data.get('state_dict', checkpoint_data.get('model_state_dict', checkpoint_data))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_dp_checkpoint = any(k.startswith('module.') for k in state_dict.keys())
        if is_dp_checkpoint:
            print("DataParallel checkpoint detected. Remapping keys.")
            for k, v in state_dict.items(): new_state_dict[k[7:]] = v
            cfm_model.load_state_dict(new_state_dict)
        else:
            cfm_model.load_state_dict(state_dict)
        print("Successfully loaded CFM model checkpoint.")
    except Exception as e: print(f"Error loading CFM checkpoint: {e}"); return
    cfm_model.eval()

    # 3. Dataset & DataLoader
    print(f"Loading dataset: {DATA_ROOT_DIR}, split: {SPLIT}, size: {IMAGE_SIZE}")
    eval_dataset = RGBNIRPairedDataset(root_dir=DATA_ROOT_DIR, split=SPLIT, image_size=IMAGE_SIZE)
    if not eval_dataset: print(f"No images in CFM eval dataset '{SPLIT}'."); return
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8) # User changed num_workers

    # 4. Metrics
    psnr_m = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_m = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_m = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)
    all_psnr, all_ssim, all_lpips, all_sam, all_corr = [], [], [], [], []
    vis_rgb, vis_real, vis_fake, vis_diff, vis_real_e, vis_fake_e = [], [], [], [], [], []

    num_total = len(eval_dataset)
    num_vis = min(NUM_SAMPLES_TO_VISUALIZE, num_total)
    vis_indices = random.sample(range(num_total), num_vis) if num_vis > 0 else []
    print(f"Selected {len(vis_indices)} indices for visualization: {vis_indices}")

    # 5. Evaluation Loop
    print(f"Starting CFM evaluation on {num_total} samples...")
    processed_metrics = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(eval_loader, desc="Evaluating CFM")):
            if data_batch is None or not all(item is not None for item in data_batch):
                print(f"Skipping CFM batch {batch_idx} due to None data."); continue
            
            condition_rgb_b, real_nir_x1_b = data_batch # Adjusted unpacking for 2 items
            condition_rgb_b, real_nir_x1_b = condition_rgb_b.to(DEVICE), real_nir_x1_b.to(DEVICE)

            initial_noise_x0_b = torch.randn_like(real_nir_x1_b, device=DEVICE)
            t_span = torch.tensor([0.0, FM_T_END], device=DEVICE)
            
            generated_nir_x1_b = sample_with_ode_solver(
                model=cfm_model, initial_noise_x0=initial_noise_x0_b,
                condition_rgb=condition_rgb_b, t_span=t_span,
                num_eval_points=FM_SAMPLING_STEPS, device=DEVICE,
                solver_method=FM_SOLVER_METHOD
            )

            real_01 = denormalize_image(real_nir_x1_b.cpu().detach())
            fake_01 = denormalize_image(generated_nir_x1_b.cpu().detach())
            cond_01 = denormalize_image(condition_rgb_b.cpu().detach())
            real_01_cl = torch.nan_to_num(real_01.clamp(0,1)); fake_01_cl = torch.nan_to_num(fake_01.clamp(0,1))
            
            psnr_m.update(fake_01_cl.to(DEVICE), real_01_cl.to(DEVICE))
            ssim_m.update(fake_01_cl.to(DEVICE), real_01_cl.to(DEVICE))
            lpips_m.update( (fake_01_cl.repeat(1,3,1,1) if fake_01_cl.size(1)==1 else fake_01_cl).to(DEVICE),
                            (real_01_cl.repeat(1,3,1,1) if real_01_cl.size(1)==1 else real_01_cl).to(DEVICE) )

            for i in range(real_01.shape[0]):
                all_sam.append(calculate_sam(fake_01[i], real_01[i]).item())
                rf, ff = real_01[i].reshape(-1), fake_01[i].reshape(-1)
                if len(rf)>1: all_corr.append(torch.corrcoef(torch.stack([rf,ff]))[0,1].item());
                else: all_corr.append(float('nan'))
            processed_metrics += real_01.shape[0]

            current_batch_g_indices = list(range(batch_idx*BATCH_SIZE, batch_idx*BATCH_SIZE + real_01.shape[0]))
            for vis_g_idx in vis_indices:
                if vis_g_idx in current_batch_g_indices and len(vis_rgb) < num_vis:
                    idx_b = vis_g_idx - (batch_idx*BATCH_SIZE)
                    vis_rgb.append(cond_01[idx_b].unsqueeze(0))
                    vis_real.append(real_01[idx_b].unsqueeze(0))
                    vis_fake.append(fake_01[idx_b].unsqueeze(0))
                    vis_diff.append((fake_01[idx_b]-real_01[idx_b]).unsqueeze(0))
                    vis_real_e.append(sobel_edges(real_01[idx_b].unsqueeze(0).to(DEVICE)).cpu())
                    vis_fake_e.append(sobel_edges(fake_01[idx_b].unsqueeze(0).to(DEVICE)).cpu())
        
        avg_psnr,avg_ssim,avg_lpips = psnr_m.compute().item(), ssim_m.compute().item(), lpips_m.compute().item()
        avg_sam = np.mean([m for m in all_sam if not np.isnan(m)]) if all_sam else float('nan')
        avg_corr = np.mean([m for m in all_corr if not np.isnan(m)]) if all_corr else float('nan')

    print(f"\n--- CFM Metrics ({processed_metrics} samples, {SPLIT} set) ---")
    print(f"  PSNR: {avg_psnr:.4f} dB, SSIM: {avg_ssim:.4f}, LPIPS: {avg_lpips:.4f}")
    print(f"  SAM: {avg_sam:.4f} rad, CorrCoeff: {avg_corr:.4f}")

    # 6. Visualization
    if len(vis_rgb) == num_vis:
        print(f"\nGenerating CFM visualization for {num_vis} samples...")
        def prep(t_list, c=1): return torch.cat(t_list,dim=0).repeat(1,3,1,1) if c==1 and t_list[0].size(1)==1 else torch.cat(t_list,dim=0)
        def norm_diff(t_list): return ((torch.cat(t_list,dim=0)+1)/2).clamp(0,1).repeat(1,3,1,1)
        
        display_tensors = [
            prep(vis_rgb,3), prep(vis_real), prep(vis_fake),
            norm_diff(vis_diff), prep(vis_real_e), prep(vis_fake_e)
        ]
        collage_items = [item[i] for i in range(num_vis) for item in display_tensors]
        collage = vutils.make_grid(collage_items, nrow=num_vis, padding=5, normalize=False)
        
        plt.figure(figsize=(max(15,num_vis*2.5),18))
        plt.imshow(collage.permute(1,2,0).cpu().numpy())
        rows_desc = ["InputRGB","RealNIR","GenNIR(CFM)","Diff","RealEdge","GenEdge"]
        plt.title(f"CFM Eval (Ckpt:{os.path.basename(CHECKPOINT_PATH_CFM)}, Split:{SPLIT})\nRows:{';'.join(rows_desc)}",fontsize=10)
        plt.axis('off')
        fname = f"cfm_eval_coll_{os.path.splitext(os.path.basename(CHECKPOINT_PATH_CFM))[0]}_{SPLIT}.png"
        plt.savefig(os.path.join(OUTPUT_DIR,fname),bbox_inches='tight')
        print(f"Saved CFM collage: {os.path.join(OUTPUT_DIR,fname)}")
    else: print("CFM visualization skipped.")
    print(f"Finished CFM eval. Results in {OUTPUT_DIR}")

if __name__ == "__main__":
    if CHECKPOINT_PATH_CFM == "path/to/your/continuous_fm_model_epoch_XX.pth":
        print("ERROR: Update CHECKPOINT_PATH_CFM in script!")
    else: evaluate_cfm_checkpoint() 

# --- CFM Metrics (18067 samples, test set) ---
#   PSNR: 22.4407 dB, SSIM: 0.7249, LPIPS: 0.1659
#   SAM: 0.1610 rad, CorrCoeff: 0.8432