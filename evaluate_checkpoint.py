import os
import torch
import torch.nn as nn
import torch.nn.functional as F # For Sobel filter
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # Более точный импорт для LPIPS
import random # For random sampling

# Import model and dataset definitions from main_v2.py
from main_v2 import UNetDown, UNetUp, GeneratorUNetSmall, RGBNIRPairedDataset

#region Model and Dataset Definitions (COPY FROM main_v2.py)
# Make sure these class definitions are present in this file or imported.
# Placeholder: You need to copy UNetDown, UNetUp, GeneratorUNetSmall, and RGBNIRPairedDataset here.
#endregion Model and Dataset Definitions

# --- Configuration ---
CHECKPOINT_PATH = "models/rgb_to_nir_wgan_gp_v2_3_256/generator_epoch_39.pth" # FROM YOUR main_v2.py config
DATA_ROOT_DIR = "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons" # FROM YOUR main_v2.py config
SPLIT = 'test'       # 'val' or 'test' (if you have a separate test set)
IMAGE_SIZE = 256    # Must match the training image size for the checkpoint
NUM_SAMPLES_TO_VISUALIZE = 6 # Visualize 6 random samples
BATCH_SIZE = 4      # Reduced batch size for potentially memory intensive operations like LPIPS or many metrics
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_DIR = "evaluation_results/wgan" # Directory to save collages and other outputs
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Generator parameters (must match the trained model's config)
G_IN_CHANNELS = 3
G_OUT_CHANNELS = 1
G_BASE_CHANNELS = 32 # As in your main_v2.py config
G_NUM_LEVELS = 5   # As in your main_v2.py config

# --- Helper Functions ---
def denormalize_image(tensor):
    """Denormalizes a tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1) / 2.0

# def collate_fn_skip_none(batch):
#     """Collate function that filters out None items from a batch."""
#     batch = [item for item in batch if item[0] is not None and item[1] is not None and item[2] is not None]
#     if not batch:
#         return None, None, None
#     return torch.utils.data.dataloader.default_collate(batch)

# --- Helper Functions for New Metrics ---
def calculate_sam(img1_chw, img2_chw, eps=1e-8):
    """
    Computes Spectral Angle Mapper (SAM) between two images (C, H, W).
    Assumes images are already in a suitable range (e.g., positive values after denormalization).
    img1_chw, img2_chw: PyTorch tensors (C, H, W)
    """
    # If single channel, ensure it's treated as having a spectral dimension of 1
    if img1_chw.ndim == 2: # H, W
        img1_chw = img1_chw.unsqueeze(0)
    if img2_chw.ndim == 2: # H, W
        img2_chw = img2_chw.unsqueeze(0)
    
    # If images are (H,W) only (single channel), unsqueeze to (1,H,W)
    # This check might be redundant given the one above, but ensures correct shape.
    if len(img1_chw.shape) == 2: img1_chw = img1_chw.unsqueeze(0)
    if len(img2_chw.shape) == 2: img2_chw = img2_chw.unsqueeze(0)

    # Flatten spatial dimensions, keep channel dimension
    img1_flat = img1_chw.view(img1_chw.size(0), -1) # (C, H*W)
    img2_flat = img2_chw.view(img2_chw.size(0), -1) # (C, H*W)

    # For single-channel NIR, SAM is the angle between two large vectors representing the images.
    if img1_flat.size(0) == 1: # Single channel image
        img1_vec = img1_flat.squeeze(0) # (H*W)
        img2_vec = img2_flat.squeeze(0) # (H*W)
        dot_product = torch.dot(img1_vec, img2_vec)
        norm_img1 = torch.norm(img1_vec)
        norm_img2 = torch.norm(img2_vec)
        cos_angle = dot_product / (norm_img1 * norm_img2 + eps)
        # Clamp to avoid NaN due to precision issues before acos
        sam_rad = torch.acos(cos_angle.clamp(-1 + eps, 1 - eps))
        return sam_rad # SAM in radians
    else:
        # This case is for multi-spectral images, averaging SAM over pixels.
        # For our single channel NIR comparison, this branch shouldn't ideally be hit.
        # If it is, it implies an unexpected input format.
        # For now, we'll treat it as a single vector comparison as a fallback.
        print("Warning: SAM called with multi-channel image where single channel was expected. Calculating SAM over flattened vectors.")
        img1_vec = img1_flat.reshape(-1) # Flatten all dimensions
        img2_vec = img2_flat.reshape(-1) # Flatten all dimensions
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
        # If multi-channel, convert to grayscale (average for simplicity)
        # This shouldn't happen if we pass denormalized NIR (single channel)
        image_bchw = torch.mean(image_bchw, dim=1, keepdim=True)

    # Sobel kernels
    sobel_x_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)
    sobel_y_kernel = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=image_bchw.dtype, device=image_bchw.device).unsqueeze(0).unsqueeze(0)

    # Repeat for each input channel if C > 1, though we expect C=1
    # For C=1, these lines don't change sobel_x and sobel_y
    sobel_x = sobel_x_kernel.repeat(image_bchw.size(1), 1, 1, 1)
    sobel_y = sobel_y_kernel.repeat(image_bchw.size(1), 1, 1, 1)
    
    # Pad to keep original size
    padding = (sobel_x_kernel.shape[-1] -1) // 2
    
    # Apply convolution
    # Group convolution is used: each input channel is convolved with its own set of filters.
    # Since NIR is single channel (C=1), groups=1 behaves like standard conv.
    grad_x = F.conv2d(image_bchw, sobel_x, padding=padding, stride=1, groups=image_bchw.size(1))
    grad_y = F.conv2d(image_bchw, sobel_y, padding=padding, stride=1, groups=image_bchw.size(1))

    # Magnitude of gradients
    edge_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + eps)
    
    # Normalize to [0, 1] for visualization (per image in batch if B > 1)
    for b_idx in range(edge_magnitude.shape[0]):
        img_slice = edge_magnitude[b_idx]
        min_val = img_slice.min()
        max_val = img_slice.max()
        edge_magnitude[b_idx] = (img_slice - min_val) / (max_val - min_val + eps)
        
    return edge_magnitude.clamp(0,1)


# --- Main Evaluation Function ---
def evaluate_checkpoint():
    print(f"Using device: {DEVICE}")
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    print(f"Evaluating on: {DATA_ROOT_DIR}, split: {SPLIT}, image size: {IMAGE_SIZE}")

    # 1. Initialize Model
    generator = GeneratorUNetSmall(
        in_channels=G_IN_CHANNELS,
        out_channels=G_OUT_CHANNELS,
        base_channels=G_BASE_CHANNELS,
        num_levels=G_NUM_LEVELS
    ).to(DEVICE)

    # 2. Load Checkpoint
    try:
        checkpoint_data = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
        # Check if the checkpoint is a state_dict directly or a dict containing state_dict
        if isinstance(checkpoint_data, dict) and 'state_dict' in checkpoint_data: # For more complex checkpoints
            state_dict = checkpoint_data['state_dict']
        elif isinstance(checkpoint_data, dict) and 'model_state_dict' in checkpoint_data: # Common pattern
             state_dict = checkpoint_data['model_state_dict']
        else: # Assumed to be state_dict directly
            state_dict = checkpoint_data
        
        # Handle 'module.' prefix if the model was saved using DataParallel
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        model_is_dataparallel = isinstance(generator, nn.DataParallel)
        checkpoint_is_dataparallel = any(key.startswith('module.') for key in state_dict.keys())

        if checkpoint_is_dataparallel and not model_is_dataparallel:
            print("Checkpoint is from DataParallel, loading into a single-GPU model.")
            for k, v in state_dict.items():
                name = k[7:] if k.startswith('module.') else k # remove `module.`
                new_state_dict[name] = v
            generator.load_state_dict(new_state_dict)
        elif not checkpoint_is_dataparallel and model_is_dataparallel:
            print("Checkpoint is from a single-GPU model, loading into DataParallel model.")
            generator.module.load_state_dict(state_dict) # Add 'module.' prefix implicitly
        else: # Both are DataParallel or both are not
            print("Checkpoint and model DataParallel status match or both are single-GPU.")
            generator.load_state_dict(state_dict)
            
        print(f"Successfully loaded generator checkpoint from {CHECKPOINT_PATH}")
    except Exception as e:
        print(f"Error loading generator checkpoint: {e}. Please ensure the path and model definition are correct.")
        return

    generator.eval()

    # 3. Initialize Dataset and DataLoader for FULL evaluation
    eval_dataset_full = RGBNIRPairedDataset(
        root_dir=DATA_ROOT_DIR,
        split=SPLIT,
        preload_to_ram=False,
        image_size=IMAGE_SIZE
    )

    if len(eval_dataset_full) == 0:
        print(f"No images found in the dataset for split '{SPLIT}'. Please check paths and data.")
        return

    eval_loader_full = DataLoader(
        eval_dataset_full,
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0
        # collate_fn=collate_fn_skip_none # User commented this out
    )

    # 4. Initialize Metrics
    psnr_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)

    # Accumulators for global error maps
    all_error_maps_sum_abs = []
    all_error_maps_sum_sq = []
    total_samples_for_metrics = 0

    all_psnr = []
    all_ssim = []
    all_lpips = []
    all_sam = [] # New list for SAM
    all_corr_coeff = [] # New list for Correlation Coefficient

    # For visualization
    vis_sample_indices = []
    vis_input_rgb_display = []
    vis_real_nir_display = []
    vis_fake_nir_display = []
    diff_imgs_to_display = [] # For Difference Images
    real_edges_to_display = [] # For Real Edge Maps
    fake_edges_to_display = [] # For Fake Edge Maps

    print(f"Calculating metrics over the entire {SPLIT} set ({len(eval_dataset_full)} samples)...")
    with torch.no_grad():
        for batch_idx, data_batch in enumerate(tqdm(eval_loader_full, desc=f"Evaluating ALL {SPLIT} samples")):
            # DataLoader might error before this if a None item is encountered from dataset without a custom collate_fn
            if data_batch is None or not all(item is not None for item in data_batch):
                print(f"Warning: Skipping batch {batch_idx} due to None data. Consider using a collate_fn to handle loading errors.")
                continue
            
            input_gen_batch, rgb_for_log_batch, real_nir_batch = data_batch

            if input_gen_batch.size(0) == 0: # Should not happen if collate_fn is not filtering
                continue

            input_gen_batch = input_gen_batch.to(DEVICE)
            real_nir_batch = real_nir_batch.to(DEVICE)
            # rgb_for_log_batch is also moved to device if needed, but primarily for metrics we need input_gen and real_nir

            fake_nir_batch = generator(input_gen_batch)

            real_nir_batch_01 = denormalize_image(real_nir_batch.detach())
            fake_nir_batch_01 = denormalize_image(fake_nir_batch.detach())

            psnr_metric.update(fake_nir_batch_01, real_nir_batch_01)
            ssim_metric.update(fake_nir_batch_01, real_nir_batch_01)

            real_nir_lpips_input = real_nir_batch_01.repeat(1,3,1,1) if real_nir_batch_01.size(1) == 1 else real_nir_batch_01
            fake_nir_lpips_input = fake_nir_batch_01.repeat(1,3,1,1) if fake_nir_batch_01.size(1) == 1 else fake_nir_batch_01
            lpips_metric.update(fake_nir_lpips_input, real_nir_lpips_input)
            
            abs_error_batch = torch.abs(real_nir_batch_01.cpu() - fake_nir_batch_01.cpu())
            sq_error_batch = (real_nir_batch_01.cpu() - fake_nir_batch_01.cpu())**2
            for i in range(abs_error_batch.size(0)): # Iterate through samples in batch
                all_error_maps_sum_abs.append(abs_error_batch[i:i+1])
                all_error_maps_sum_sq.append(sq_error_batch[i:i+1])
            
            total_samples_for_metrics += input_gen_batch.size(0)

            # --- New Metrics Calculation per batch ---
            # 1. SAM (Spectral Angle Mapper) - per image in batch
            # SAM expects positive values, use denormalized [0,1]
            for idx_in_batch in range(real_nir_batch_01.shape[0]):
                sam_val = calculate_sam(fake_nir_batch_01[idx_in_batch], real_nir_batch_01[idx_in_batch])
                all_sam.append(sam_val.item())

            # 2. Correlation Coefficient - per image in batch
            # Use denormalized [0,1] or original [-1,1], should not matter significantly for correlation
            for idx_in_batch in range(real_nir_batch_01.shape[0]):
                real_flat = real_nir_batch_01[idx_in_batch].reshape(-1)
                fake_flat = fake_nir_batch_01[idx_in_batch].reshape(-1)
                if len(real_flat) > 1 and len(fake_flat) > 1:
                    stacked_tensors = torch.stack([real_flat, fake_flat])
                    corr_matrix = torch.corrcoef(stacked_tensors)
                    all_corr_coeff.append(corr_matrix[0, 1].item())
                else:
                    all_corr_coeff.append(float('nan'))

            # --- Store samples for visualization --- (This section was removed as it was non-functional)
            # The vis_sample_indices list is empty at this stage of the script.
            # Visualization samples are now processed in a dedicated loop later.

    if total_samples_for_metrics == 0:
        print("No samples were processed successfully for metrics. Aborting.")
        return

    avg_psnr = psnr_metric.compute().item()
    avg_ssim = ssim_metric.compute().item()
    avg_lpips = lpips_metric.compute().item()
    avg_sam = np.mean([m for m in all_sam if not np.isnan(m) and not np.isinf(m)]) if all_sam else float('nan')
    avg_corr_coeff = np.mean([m for m in all_corr_coeff if not np.isnan(m) and not np.isinf(m)]) if all_corr_coeff else float('nan')

    print(f"\n--- Evaluation Metrics (on ALL {total_samples_for_metrics} samples from {SPLIT} set) ---")
    print(f"  PSNR: {avg_psnr:.4f} dB (Higher is better)")
    print(f"  SSIM: {avg_ssim:.4f} (Higher is better, range [0,1])")
    print(f"  LPIPS (AlexNet): {avg_lpips:.4f} (Lower is better)")
    print(f"  SAM: {avg_sam:.4f} (Lower is better)")
    print(f"  Correlation Coefficient: {avg_corr_coeff:.4f} (Higher is better, range [-1,1])")

    # --- Visualization of Random Samples ---
    num_total_dataset_samples = len(eval_dataset_full)
    actual_num_to_visualize = min(NUM_SAMPLES_TO_VISUALIZE, num_total_dataset_samples)
    
    if actual_num_to_visualize > 0:
        print(f"\nPreparing visualization for {actual_num_to_visualize} random samples...")
        
        # Ensure dataset length is positive before sampling
        if num_total_dataset_samples == 0:
            print("Cannot select random samples for visualization as the dataset is empty.")
        else:
            random_indices = random.sample(range(num_total_dataset_samples), actual_num_to_visualize)

            with torch.no_grad():
                for i in tqdm(range(actual_num_to_visualize), desc="Processing random samples for visualization"):
                    idx = random_indices[i]
                    
                    # Get individual sample: input_gen_tensor, rgb_for_log_tensor, real_nir_tensor
                    # These are the three tensors returned by RGBNIRPairedDataset.__getitem__
                    sample_data = eval_dataset_full[idx] 
                    
                    if sample_data is None or not all(t is not None for t in sample_data):
                        print(f"Warning: Skipping random sample at index {idx} for visualization due to loading error or None data.")
                        continue
                    
                    input_gen_tensor, rgb_for_log_tensor, real_nir_tensor = sample_data

                    input_gen_tensor = input_gen_tensor.unsqueeze(0).to(DEVICE) # Add batch dim
                    real_nir_tensor = real_nir_tensor.unsqueeze(0).to(DEVICE)   # Add batch dim
                    rgb_for_log_tensor = rgb_for_log_tensor.unsqueeze(0).to(DEVICE) # Add batch dim

                    fake_nir_tensor = generator(input_gen_tensor)

                    # Denormalize images to [0,1] for display and some metrics
                    # .cpu().detach() is used as these are for lists that will be used by matplotlib later
                    input_rgb_01 = denormalize_image(rgb_for_log_tensor.cpu().detach())
                    real_nir_01 = denormalize_image(real_nir_tensor.cpu().detach())
                    fake_nir_01 = denormalize_image(fake_nir_tensor.cpu().detach())

                    vis_input_rgb_display.append(input_rgb_01) # Should be (1, 3, H, W)
                    vis_real_nir_display.append(real_nir_01)   # Should be (1, 1, H, W)
                    vis_fake_nir_display.append(fake_nir_01)   # Should be (1, 1, H, W)

                    # Difference Image: (fake_nir_01 - real_nir_01) results in range [-1, 1]
                    # These are already CPU tensors with shape (1,1,H,W)
                    diff_img_tensor = fake_nir_01 - real_nir_01
                    diff_imgs_to_display.append(diff_img_tensor)

                    # Edge Maps: sobel_edges expects tensor on DEVICE and returns tensor on DEVICE.
                    # Inputs real_nir_01, fake_nir_01 are (1,1,H,W) CPU tensors.
                    # Send to DEVICE for sobel_edges, then back to CPU for storage.
                    real_edge_map = sobel_edges(real_nir_01.to(DEVICE))
                    fake_edge_map = sobel_edges(fake_nir_01.to(DEVICE))
                    real_edges_to_display.append(real_edge_map.cpu())
                    fake_edges_to_display.append(fake_edge_map.cpu())
    
    if vis_input_rgb_display and len(vis_input_rgb_display) == actual_num_to_visualize: # Ensure we have all samples
        input_rgb_t = torch.cat(vis_input_rgb_display, dim=0)
        real_nir_t = torch.cat(vis_real_nir_display, dim=0) # These are already [0,1] from storage
        fake_nir_t = torch.cat(vis_fake_nir_display, dim=0) # These are already [0,1] from storage
        diff_imgs_t = torch.cat(diff_imgs_to_display, dim=0) # These are in [-1,1] approx from storage
        real_edges_t = torch.cat(real_edges_to_display, dim=0) # These are [0,1] from Sobel
        fake_edges_t = torch.cat(fake_edges_to_display, dim=0) # These are [0,1] from Sobel

        # Ensure NIR images are displayable (repeat single channel to 3 channels for vutils if necessary)
        # Input RGB is already 3 channels. Real/Fake NIR from dataset/generator are single channel.
        # The vis_real_nir_display and vis_fake_nir_display stored are [0,1] single channel.
        real_nir_display_prep = real_nir_t.repeat(1,3,1,1) if real_nir_t.size(1) == 1 else real_nir_t
        fake_nir_display_prep = fake_nir_t.repeat(1,3,1,1) if fake_nir_t.size(1) == 1 else fake_nir_t
        
        # Difference image: stored as (fake-real)/2.0, so in range [-1,1]. 
        # We want to visualize this: e.g. map to [0,1] or use a colormap.
        # Simple normalization to [0,1] for display:
        diff_imgs_display_prep = (diff_imgs_t + 1) / 2.0 
        diff_imgs_display_prep = diff_imgs_display_prep.clamp(0,1).repeat(1,3,1,1) if diff_imgs_display_prep.size(1) == 1 else diff_imgs_display_prep

        # Edge maps are already [0,1] single channel from Sobel function
        real_edges_display_prep = real_edges_t.repeat(1,3,1,1) if real_edges_t.size(1) == 1 else real_edges_t
        fake_edges_display_prep = fake_edges_t.repeat(1,3,1,1) if fake_edges_t.size(1) == 1 else fake_edges_t

        num_display_samples = input_rgb_t.size(0)
        collage_list = []
        
        # New layout: samples are columns, types are rows
        # Row 1: Input RGB
        # Row 2: Real NIR ([0,1])
        # Row 3: Generated NIR ([0,1])
        # Row 4: Difference (Gen-Real), normalized to [0,1]
        # Row 5: Real NIR Edges ([0,1])
        # Row 6: Generated NIR Edges ([0,1])

        all_types_tensor_display = [
            input_rgb_t,          # Already (B,3,H,W) and in displayable range (e.g. [0,1] if from dataloader with ToTensor)
            real_nir_display_prep,
            fake_nir_display_prep,
            diff_imgs_display_prep,
            real_edges_display_prep,
            fake_edges_display_prep
        ]
        
        for i in range(num_display_samples): # Iterate through samples (columns in grid)
            for type_tensor in all_types_tensor_display: # Iterate through types (rows in grid)
                 collage_list.append(type_tensor[i]) # Pick i-th sample of current type

        # make_grid expects a list of (C,H,W) tensors or a (B,C,H,W) tensor.
        # Our collage_list is already a list of (C,H,W) tensors if C=3 for display.
        collage = vutils.make_grid(collage_list, nrow=num_display_samples, padding=5, normalize=False, scale_each=False) # Normalize & scale_each handled per type if needed

        plt.figure(figsize=(max(15, num_display_samples * 2.5), 18)) # Increased height for more rows
        plt.imshow(collage.permute(1, 2, 0).cpu().numpy())
        
        row_titles = ["Input RGB", "Real NIR", "Generated NIR", "Difference (Gen-Real)", "Real NIR Edges", "Gen. NIR Edges"]
        title_string = f"Evaluation Samples (Checkpoint: {os.path.basename(CHECKPOINT_PATH)}, Split: {SPLIT})\nRows: {'; '.join(row_titles)}"
        plt.title(title_string, fontsize=10)
        plt.axis('off')
        
        collage_filename = f"detailed_evaluation_collage_{os.path.splitext(os.path.basename(CHECKPOINT_PATH))[0]}_{SPLIT}.png"
        plt.savefig(os.path.join(OUTPUT_DIR, collage_filename), bbox_inches='tight')
        print(f"Saved detailed collage to {os.path.join(OUTPUT_DIR, collage_filename)}")
        # plt.show() # Comment out if running non-interactively
    else:
        if not vis_input_rgb_display:
            print("No samples selected for visualization or visualization lists are empty.")
        else:
            print(f"Visualization skipped: expected {actual_num_to_visualize} samples, but found {len(vis_input_rgb_display)}.")

    print(f"Finished evaluation. Results saved in {OUTPUT_DIR}")


if __name__ == '__main__':
    evaluate_checkpoint() 

# --- Evaluation Metrics (on ALL 18067 samples from test set) ---
#   PSNR: 22.1696 dB (Higher is better)
#   SSIM: 0.7402 (Higher is better, range [0,1])
#   LPIPS (AlexNet): 0.2263 (Lower is better)
#   SAM: 0.1611 (Lower is better)
#   Correlation Coefficient: 0.8385 (Higher is better, range [-1,1])

# Preparing visualization for 6 random samples...
# Processing random samples for visualization: 100%|█████████████████████████████████████| 6/6 [00:00<00:00, 16.28it/s]
# Saved detailed collage to evaluation_results/wgan/detailed_evaluation_collage_generator_epoch_39_test.png
# Finished evaluation. Results saved in evaluation_results/wgan