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
from main_flowmatching import RGBNIRPairedDataset # Assuming this is the correct dataset class
from flow_matching_models import VectorFieldUNet   # Your DFM model
from main_discrete_flowmatching import sample_with_discrete_steps # Your DFM sampler

# Helper functions from evaluate_checkpoint.py (SAM, Sobel, denormalize)
# These will be copied or redefined here

# --- Configuration ---
# TODO: Fill these in based on user input or a config file
CHECKPOINT_PATH_DFM = "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/models/discrete_flow_matching_v1/dfm_model_epoch_200.pth" # !!! USER NEEDS TO PROVIDE !!!
DATA_ROOT_DIR = "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons"
SPLIT = 'test'
IMAGE_SIZE = 128 # Example, should match training
NUM_SAMPLES_TO_VISUALIZE = 6
BATCH_SIZE = 4 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "evaluation_results/dfm"

# DFM Model specific parameters (Must match the trained model's config)
# TODO: Fill these based on user input or config
NIR_CHANNELS = 1
RGB_CHANNELS = 3
UNET_BASE_CHANNELS = 64
UNET_NUM_LEVELS = 3 
TIME_EMB_DIM = 256
# CONTINUOUS_TIME_EMB_MAX_PERIOD = 1000.0 # If your UNet uses this

# DFM Sampler specific parameters
# TODO: Fill these based on user input or config
FM_T_END = 1.0
DFM_SAMPLING_STEPS = 50 


# --- Helper Functions (Copied/adapted from evaluate_checkpoint.py) ---
def denormalize_image(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2.0

def calculate_sam(img1_chw, img2_chw, eps=1e-8):
    """
    Computes Spectral Angle Mapper (SAM) between two images (C, H, W).
    Assumes images are already in a suitable range (e.g., positive values after denormalization).
    img1_chw, img2_chw: PyTorch tensors (C, H, W)
    """
    if img1_chw.ndim == 2: img1_chw = img1_chw.unsqueeze(0)
    if img2_chw.ndim == 2: img2_chw = img2_chw.unsqueeze(0)
    if len(img1_chw.shape) == 2: img1_chw = img1_chw.unsqueeze(0)
    if len(img2_chw.shape) == 2: img2_chw = img2_chw.unsqueeze(0)

    img1_flat = img1_chw.view(img1_chw.size(0), -1)
    img2_flat = img2_chw.view(img2_chw.size(0), -1)

    if img1_flat.size(0) == 1:
        img1_vec = img1_flat.squeeze(0)
        img2_vec = img2_flat.squeeze(0)
        dot_product = torch.dot(img1_vec, img2_vec)
        norm_img1 = torch.norm(img1_vec)
        norm_img2 = torch.norm(img2_vec)
        cos_angle = dot_product / (norm_img1 * norm_img2 + eps)
        sam_rad = torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
        return sam_rad
    else:
        print("Warning: SAM called with multi-channel image where single channel was expected. Calculating SAM over flattened vectors.")
        img1_vec = img1_flat.reshape(-1)
        img2_vec = img2_flat.reshape(-1)
        dot_product = torch.dot(img1_vec, img2_vec)
        norm_img1 = torch.norm(img1_vec)
        norm_img2 = torch.norm(img2_vec)
        cos_angle = dot_product / (norm_img1 * norm_img2 + eps)
        sam_rad = torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
        return sam_rad

def sobel_edges(image_bchw, eps=1e-6):
    """
    Detects edges using Sobel filter.
    image_bchw: PyTorch tensor (B, C, H, W), expects C=1 (grayscale)
    """
    if image_bchw.size(1) > 1:
        image_bchw = torch.mean(image_bchw, dim=1, keepdim=True)

    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)
    
    sobel_x = sobel_x_kernel.repeat(image_bchw.size(1), 1, 1, 1)
    sobel_y = sobel_y_kernel.repeat(image_bchw.size(1), 1, 1, 1)
    
    padding = (sobel_x_kernel.shape[-1] -1) // 2
    
    grad_x = F.conv2d(image_bchw, sobel_x, padding=padding, stride=1, groups=image_bchw.size(1))
    grad_y = F.conv2d(image_bchw, sobel_y, padding=padding, stride=1, groups=image_bchw.size(1))

    edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    
    for b_idx in range(edge_magnitude.shape[0]):
        img_slice = edge_magnitude[b_idx]
        min_val = img_slice.min()
        max_val = img_slice.max()
        edge_magnitude[b_idx] = (img_slice - min_val) / (max_val - min_val + eps)
        
    return edge_magnitude.clamp(0,1)


# --- Main Evaluation Function ---
def evaluate_dfm_checkpoint():
    print(f"Using device: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Initialize Model
    print("Initializing DFM model...")
    dfm_model = VectorFieldUNet(
        nir_channels=NIR_CHANNELS,
        rgb_channels=RGB_CHANNELS,
        out_channels_vector_field=NIR_CHANNELS, # Output is the vector field for NIR
        base_channels=UNET_BASE_CHANNELS,
        num_levels=UNET_NUM_LEVELS,
        time_emb_dim=TIME_EMB_DIM
        # continuous_time_emb_max_period=CONTINUOUS_TIME_EMB_MAX_PERIOD # Add if your model uses it
    ).to(DEVICE)

    # 2. Load Checkpoint
    print(f"Loading DFM checkpoint from: {CHECKPOINT_PATH_DFM}")
    try:
        # Load checkpoint handling DataParallel or single GPU saves
        checkpoint_data = torch.load(CHECKPOINT_PATH_DFM, map_location=DEVICE)
        state_dict = checkpoint_data.get('state_dict', checkpoint_data.get('model_state_dict', checkpoint_data))
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        is_dataparallel_checkpoint = any(key.startswith('module.') for key in state_dict.keys())

        if is_dataparallel_checkpoint:
            print("Checkpoint is from DataParallel, loading into a single-GPU model setup for eval.")
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k
                new_state_dict[name] = v
            dfm_model.load_state_dict(new_state_dict)
        else:
            dfm_model.load_state_dict(state_dict)
        print("Successfully loaded DFM model checkpoint.")
    except Exception as e:
        print(f"Error loading DFM checkpoint: {e}. Please check path and model definition.")
        return

    dfm_model.eval()

    # 3. Initialize Dataset and DataLoader
    print(f"Loading dataset from: {DATA_ROOT_DIR}, split: {SPLIT}, image size: {IMAGE_SIZE}")
    eval_dataset = RGBNIRPairedDataset(
        root_dir=DATA_ROOT_DIR,
        split=SPLIT,
        preload_to_ram=False, # Keep false for evaluation usually
        image_size=IMAGE_SIZE
    )
    if len(eval_dataset) == 0:
        print(f"No images found for DFM evaluation in split '{SPLIT}'.")
        return
    
    eval_loader = DataLoader(
        eval_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=8 # Simpler for eval
    )

    # 4. Initialize Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)

    all_psnr, all_ssim, all_lpips, all_sam, all_corr_coeff = [], [], [], [], []
    vis_input_rgb, vis_real_nir, vis_fake_nir = [], [], []
    vis_diff_imgs, vis_real_edges, vis_fake_edges = [], [], []

    # Select random samples for visualization from the dataset
    num_total_dataset_samples = len(eval_dataset)
    actual_num_to_visualize = min(NUM_SAMPLES_TO_VISUALIZE, num_total_dataset_samples)
    vis_sample_indices = []
    if actual_num_to_visualize > 0:
        vis_sample_indices = random.sample(range(num_total_dataset_samples), actual_num_to_visualize)
        print(f"Selected {len(vis_sample_indices)} random indices for visualization: {vis_sample_indices}")

    # 5. Evaluation Loop
    print(f"Starting DFM evaluation on {len(eval_dataset)} samples...")
    total_processed_for_metrics = 0
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(eval_loader, desc="Evaluating DFM")):
            if data_batch is None or not all(item is not None for item in data_batch):
                print(f"Skipping DFM batch {batch_idx} due to None data.")
                continue
            
            # Dataset returns: input_for_generator (RGB), rgb_for_log, real_nir_target
            # For DFM, input_for_generator is the condition (RGB)
            # real_nir_target is x1 for the flow
            condition_rgb_batch, real_nir_x1_batch = data_batch
            condition_rgb_batch = condition_rgb_batch.to(DEVICE)
            real_nir_x1_batch = real_nir_x1_batch.to(DEVICE)

            # Generate initial noise x0 for sampling
            initial_noise_x0_batch = torch.randn_like(real_nir_x1_batch, device=DEVICE)
            
            # Sample using DFM sampler
            generated_nir_x1_batch = sample_with_discrete_steps(
                dfm_model, initial_noise_x0_batch, condition_rgb_batch,
                DFM_SAMPLING_STEPS, FM_T_END, DEVICE
            )

            # Denormalize for metrics and visualization
            real_nir_01 = denormalize_image(real_nir_x1_batch.cpu().detach())
            fake_nir_01 = denormalize_image(generated_nir_x1_batch.cpu().detach())
            condition_rgb_01 = denormalize_image(condition_rgb_batch.cpu().detach())

            # Clamp values to [0,1] after denorm as a safeguard for metrics
            real_nir_01 = torch.nan_to_num(real_nir_01.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)
            fake_nir_01 = torch.nan_to_num(fake_nir_01.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)

            # Send to device for torchmetrics
            real_nir_01_dev = real_nir_01.to(DEVICE)
            fake_nir_01_dev = fake_nir_01.to(DEVICE)

            psnr_metric.update(fake_nir_01_dev, real_nir_01_dev)
            ssim_metric.update(fake_nir_01_dev, real_nir_01_dev)
            
            # LPIPS expects 3 channels, repeat if necessary
            real_lpips_in = real_nir_01_dev.repeat(1,3,1,1) if real_nir_01_dev.size(1) == 1 else real_nir_01_dev
            fake_lpips_in = fake_nir_01_dev.repeat(1,3,1,1) if fake_nir_01_dev.size(1) == 1 else fake_nir_01_dev
            lpips_metric.update(fake_lpips_in, real_lpips_in)

            for i in range(real_nir_01.shape[0]):
                sam_val = calculate_sam(fake_nir_01[i], real_nir_01[i]) # Uses CPU tensors
                all_sam.append(sam_val.item())
                
                real_flat = real_nir_01[i].reshape(-1)
                fake_flat = fake_nir_01[i].reshape(-1)
                if len(real_flat) > 1 and len(fake_flat) > 1:
                    stacked_tensors = torch.stack([real_flat, fake_flat])
                    corr_matrix = torch.corrcoef(stacked_tensors)
                    all_corr_coeff.append(corr_matrix[0, 1].item())
                else:
                    all_corr_coeff.append(float('nan'))
            
            total_processed_for_metrics += real_nir_01.shape[0]

            # Store samples for visualization if their original index is in vis_sample_indices
            current_batch_global_indices = list(range(batch_idx * BATCH_SIZE, batch_idx * BATCH_SIZE + real_nir_01.shape[0]))
            for vis_global_idx in vis_sample_indices:
                if vis_global_idx in current_batch_global_indices and len(vis_input_rgb) < actual_num_to_visualize:
                    idx_in_batch = vis_global_idx - (batch_idx * BATCH_SIZE)
                    
                    vis_input_rgb.append(condition_rgb_01[idx_in_batch].unsqueeze(0)) # (1, C, H, W)
                    vis_real_nir.append(real_nir_01[idx_in_batch].unsqueeze(0))
                    vis_fake_nir.append(fake_nir_01[idx_in_batch].unsqueeze(0))

                    diff_img = fake_nir_01[idx_in_batch] - real_nir_01[idx_in_batch] # Range [-1, 1]
                    vis_diff_imgs.append(diff_img.unsqueeze(0))

                    # Edge maps (Sobel expects B,C,H,W on device)
                    real_edge = sobel_edges(real_nir_01[idx_in_batch].unsqueeze(0).to(DEVICE)).cpu()
                    fake_edge = sobel_edges(fake_nir_01[idx_in_batch].unsqueeze(0).to(DEVICE)).cpu()
                    vis_real_edges.append(real_edge)
                    vis_fake_edges.append(fake_edge)
        
        avg_psnr = psnr_metric.compute().item()
        avg_ssim = ssim_metric.compute().item()
        avg_lpips = lpips_metric.compute().item()
        avg_sam = np.mean([m for m in all_sam if not np.isnan(m) and not np.isinf(m)]) if all_sam else float('nan')
        avg_corr_coeff = np.mean([m for m in all_corr_coeff if not np.isnan(m) and not np.isinf(m)]) if all_corr_coeff else float('nan')

    print(f"\n--- DFM Evaluation Metrics (on {total_processed_for_metrics} samples from {SPLIT} set) ---")
    print(f"  PSNR: {avg_psnr:.4f} dB")
    print(f"  SSIM: {avg_ssim:.4f}")
    print(f"  LPIPS (AlexNet): {avg_lpips:.4f}")
    print(f"  SAM: {avg_sam:.4f} (radians, lower is better)")
    print(f"  Correlation Coefficient: {avg_corr_coeff:.4f}")

    # 6. Visualization
    if len(vis_input_rgb) == actual_num_to_visualize:
        print(f"\nGenerating visualization collage for {actual_num_to_visualize} samples...")
        input_rgb_t = torch.cat(vis_input_rgb, dim=0)
        real_nir_t = torch.cat(vis_real_nir, dim=0)
        fake_nir_t = torch.cat(vis_fake_nir, dim=0)
        diff_imgs_t = torch.cat(vis_diff_imgs, dim=0)
        real_edges_t = torch.cat(vis_real_edges, dim=0)
        fake_edges_t = torch.cat(vis_fake_edges, dim=0)

        # Prepare for display (e.g. ensure 3 channels for NIR/edges, normalize diff_imgs)
        real_nir_display_prep = real_nir_t.repeat(1,3,1,1) if real_nir_t.size(1) == 1 else real_nir_t
        fake_nir_display_prep = fake_nir_t.repeat(1,3,1,1) if fake_nir_t.size(1) == 1 else fake_nir_t
        diff_imgs_display_prep = ((diff_imgs_t + 1) / 2.0).clamp(0,1) # Normalize diff to [0,1]
        diff_imgs_display_prep = diff_imgs_display_prep.repeat(1,3,1,1) if diff_imgs_display_prep.size(1) == 1 else diff_imgs_display_prep
        real_edges_display_prep = real_edges_t.repeat(1,3,1,1) if real_edges_t.size(1) == 1 else real_edges_t
        fake_edges_display_prep = fake_edges_t.repeat(1,3,1,1) if fake_edges_t.size(1) == 1 else fake_edges_t

        collage_list = []
        all_types_tensor_display = [
            input_rgb_t, real_nir_display_prep, fake_nir_display_prep,
            diff_imgs_display_prep, real_edges_display_prep, fake_edges_display_prep
        ]
        for i in range(actual_num_to_visualize):
            for type_tensor in all_types_tensor_display:
                 collage_list.append(type_tensor[i])
        
        collage = vutils.make_grid(collage_list, nrow=actual_num_to_visualize, padding=5, normalize=False)
        plt.figure(figsize=(max(15, actual_num_to_visualize * 2.5), 18))
        plt.imshow(collage.permute(1, 2, 0).cpu().numpy())
        row_titles = ["Input RGB", "Real NIR", "Generated NIR (DFM)", "Difference", "Real Edges", "Gen. Edges"]
        title_str = f"DFM Evaluation (Checkpoint: {os.path.basename(CHECKPOINT_PATH_DFM)}, Split: {SPLIT})\nRows: {'; '.join(row_titles)}"
        plt.title(title_str, fontsize=10)
        plt.axis('off')
        collage_filename = f"dfm_evaluation_collage_{os.path.splitext(os.path.basename(CHECKPOINT_PATH_DFM))[0]}_{SPLIT}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, collage_filename), bbox_inches='tight')
        print(f"Saved DFM collage to {os.path.join(OUTPUT_DIR, collage_filename)}")
    else:
        print("Visualization skipped: Not enough samples collected or mismatch in numbers.")

    print(f"Finished DFM evaluation. Results saved in {OUTPUT_DIR}")

if __name__ == "__main__":
    # It's good practice to allow configuration overrides via command-line arguments or a config file
    # For simplicity, primary configuration is at the top of this script.
    # Users should modify CHECKPOINT_PATH_DFM and other TODOs directly in the script for now.
    
    # Example of how one might load a config file (e.g., JSON) in the future:
    # import json
    # config_path = 'path/to/dfm_eval_config.json'
    # if os.path.exists(config_path):
    #     with open(config_path, 'r') as f:
    #         eval_config = json.load(f)
    #         CHECKPOINT_PATH_DFM = eval_config.get('checkpoint_path', CHECKPOINT_PATH_DFM)
    #         # ... update other global config variables ...

    if CHECKPOINT_PATH_DFM == "path/to/your/dfm_model_epoch_XX.pth":
        print("ERROR: Please update CHECKPOINT_PATH_DFM in the script before running!")
    else:
        evaluate_dfm_checkpoint() 

# --- DFM Evaluation Metrics (on 18067 samples from test set) ---
#   PSNR: 20.3391 dB
#   SSIM: 0.7002
#   LPIPS (AlexNet): 0.1583
#   SAM: 0.1837 (radians, lower is better)
#   Correlation Coefficient: 0.7963

# Generating visualization collage for 6 samples...
# Saved DFM collage to evaluation_results/dfm/dfm_evaluation_collage_dfm_model_epoch_200_test.png
# Finished DFM evaluation. Results saved in evaluation_results/dfm