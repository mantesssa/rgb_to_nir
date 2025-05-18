import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision import transforms
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# Imports from existing project files
from flow_matching_models import VectorFieldUNet # Assuming VectorFieldUNet can take float time
from main_flowmatching import RGBNIRPairedDataset # Re-using the dataset

# --- Discrete Sampler ---
@torch.no_grad()
def sample_with_discrete_steps(
    model, initial_noise_x0, condition_rgb, 
    num_discrete_steps, T_end, device
):
    """
    Generates images using iterative discrete steps (Euler method).
    model: The trained VectorFieldUNet.
    initial_noise_x0: Initial noise (e.g., from N(0,1)).
    condition_rgb: Conditional RGB image.
    num_discrete_steps: Total number of discrete steps K for sampling.
    T_end: The final time for the flow (typically 1.0).
    device: Computation device.
    """
    model.eval()
    x_k = initial_noise_x0.to(device)
    condition_rgb = condition_rgb.to(device)
    
    delta_t = T_end / num_discrete_steps
    
    for k_idx in tqdm(range(num_discrete_steps, 0, -1), desc="Discrete Sampling", leave=False):
        # Current time t_k for this step
        # Time goes from T_end down to 0 (or close to 0)
        # If k_idx is num_discrete_steps, t_k is T_end. If k_idx is 1, t_k is delta_t.
        current_t_value = k_idx * delta_t
        
        # Create a batch of current_t_value for the model
        # Model expects time as a [B] or [B,1] tensor
        time_t_k_batch = torch.full(
            (x_k.shape[0],), current_t_value, device=device, dtype=torch.float32
        )

        # Predict vector field v_theta(x_k, t_k, c)
        predicted_vector_field_v_theta = model(x_k, time_t_k_batch, condition_rgb)
        
        # Euler step: x_{k-1} = x_k - delta_t * v_theta(x_k, t_k, c)
        # (Note: this is a forward Euler step for dx/dt = v, so x(t+dt) = x(t) + dt*v. 
        #  If we are integrating dx/dt = -v then it would be x(t+dt) = x(t) - dt*v.
        #  For flow matching, dx/dt = v_t, so we step forward in t from 0 to T_end.
        #  However, the sampler usually works by discretizing the ODE from x0 to x1.
        #  Let's stick to the typical formulation x_t = (1-t)x0 + t*x1, u_t = x1 - x0.
        #  The model predicts u_t. To go from x0 to x1, we solve dx/dt = u_t(x_t, t).
        #  Here, we are generating by solving dx/dt = v_theta(x_t, t, c) from x_0 (noise) at t=0
        #  to x_1 (image) at t=T_end.
        #  So, x_{t+delta_t} = x_t + delta_t * v_theta(x_t, t, c).
        #  Our loop goes from k_idx = num_steps down to 1.
        #  Let's re-think the time direction for sampling.
        #  If we start with noise x_0 at t=0 and want x_1 at t=T_end.
        #  Loop from t=0 up to T_end - delta_t.
        #  x_current is initial_noise_x0
        #  for step_idx from 0 to num_discrete_steps-1:
        #     t_current = step_idx * delta_t
        #     v = model(x_current, t_current_batch, condition)
        #     x_current = x_current + delta_t * v
        #  This seems more aligned with solving dx/dt = v from t=0 to t=T_end.
        pass # Placeholder for correct loop

    # Re-implementing sampling loop correctly
    x_current = initial_noise_x0.clone().to(device) # Start with noise at t=0
    
    trajectory = [x_current.cpu()] # Optional: store trajectory

    for step_idx in tqdm(range(num_discrete_steps), desc="Discrete Euler Sampling", leave=False):
        current_t_value = step_idx * delta_t
        if current_t_value >= T_end: # Should not exceed T_end
            current_t_value = T_end - 1e-5 # Ensure t is within bounds if issues arise

        time_t_batch = torch.full(
            (x_current.shape[0],), current_t_value, device=device, dtype=torch.float32
        )
        
        # Predict vector field v_theta(x_t, t, c)
        # This v_theta is an approximation of u_t = x1 - x0 (for linear path)
        # or generally the velocity field d x_t / dt
        v_pred = model(x_current, time_t_batch, condition_rgb)
        
        # Euler step: x_{t+delta_t} = x_t + delta_t * v_pred
        x_current = x_current + delta_t * v_pred
        # trajectory.append(x_current.cpu()) # Optional

    model.train() # Set model back to training mode
    # return x_current, trajectory
    return x_current


# --- Evaluation Function (Discrete Flow Matching) ---
@torch.no_grad()
def evaluate_on_validation_set_discrete_fm(
    model, val_loader, device,
    psnr_metric_obj, ssim_metric_obj, lpips_metric_obj, epoch_num,
    config # For sampling parameters
):
    model.eval()
    psnr_metric_obj.reset(); ssim_metric_obj.reset(); lpips_metric_obj.reset()

    T_end = config.get('fm_t_end', 1.0)
    num_sampling_steps = config.get('dfm_sampling_steps', 50)

    for rgb_condition_val, real_nir_x1_val in tqdm(val_loader, desc=f"Evaluating DFM (Epoch {epoch_num})", leave=False):
        rgb_condition_val = rgb_condition_val.to(device)
        real_nir_x1_val = real_nir_x1_val.to(device)
        
        initial_noise_x0 = torch.randn_like(real_nir_x1_val, device=device)
        
        generated_nir_x1 = sample_with_discrete_steps(
            model, initial_noise_x0, rgb_condition_val,
            num_sampling_steps, T_end, device
        )
        
        # De-normalize and calculate metrics (same as in main_flowmatching.py)
        real_nir_norm = (real_nir_x1_val + 1) / 2.0
        gen_nir_norm = (generated_nir_x1 + 1) / 2.0
        real_nir_norm = torch.nan_to_num(real_nir_norm.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)
        gen_nir_norm = torch.nan_to_num(gen_nir_norm.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)

        psnr_metric_obj.update(gen_nir_norm, real_nir_norm)
        ssim_metric_obj.update(gen_nir_norm, real_nir_norm)
        if real_nir_norm.size(1) == 1: # Grayscale to 3-channel for LPIPS
            real_lpips = real_nir_norm.repeat(1,3,1,1)
            gen_lpips = gen_nir_norm.repeat(1,3,1,1)
        else:
            real_lpips = real_nir_norm
            gen_lpips = gen_nir_norm
        lpips_metric_obj.update(gen_lpips, real_lpips)

    epoch_psnr = psnr_metric_obj.compute().item()
    epoch_ssim = ssim_metric_obj.compute().item()
    epoch_lpips = lpips_metric_obj.compute().item()
    model.train()
    return epoch_psnr, epoch_ssim, epoch_lpips

# --- Main Training Function (Discrete Flow Matching) ---
def train_discrete_flowmatching(config):
    writer = SummaryWriter(log_dir=config['log_dir'])
    # ... (config logging) ...
    DEVICE = "cuda" if torch.cuda.is_available() and config.get('device_preference') == 'cuda' else "cpu"
    print(f"Using device: {DEVICE}")
    start_epoch = 0

    # 1. Model Initialization (same as main_flowmatching.py)
    flow_model_raw = VectorFieldUNet(
        nir_channels=config.get('nir_channels', 1),
        rgb_channels=config.get('rgb_channels', 3),
        out_channels_vector_field=config.get('nir_channels', 1),
        base_channels=config.get('unet_base_channels', 64),
        num_levels=config.get('unet_num_levels', 4),
        time_emb_dim=config.get('time_emb_dim', 256),
        continuous_time_emb_max_period=config.get('continuous_time_emb_max_period', 1000.0)
    )
    # ... (checkpoint loading, DataParallel, to(DEVICE) - same as main_flowmatching.py) ...
    # This part needs to be copied and adapted if not running from scratch
    if config.get('load_checkpoint', False):
        load_epoch = config.get('checkpoint_load_epoch', 0)
        if load_epoch > 0:
            model_path = os.path.join(config.get('checkpoint_dir', 'checkpoints_dfm'), f"dfm_model_epoch_{load_epoch}.pth")
            if os.path.exists(model_path):
                try:
                    flow_model_raw.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"Loaded DFM Model weights from epoch {load_epoch}"); start_epoch = load_epoch
                except Exception as e: print(f"Error loading DFM model checkpoint: {e}. Starting fresh."); start_epoch = 0
            else: print(f"DFM Model checkpoint not found at {model_path}. Starting fresh."); start_epoch = 0
        else: print("load_checkpoint is True, but checkpoint_load_epoch invalid. Starting fresh."); start_epoch = 0
    
    flow_model = flow_model_raw
    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        flow_model = nn.DataParallel(flow_model_raw); print(f"Using {torch.cuda.device_count()} GPUs for DFM.")
    flow_model.to(DEVICE)


    # 2. Optimizer (same)
    optimizer = optim.Adam(flow_model.parameters(), lr=config['learning_rate'], betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)))
    # ... (optimizer checkpoint loading - same) ...
    if config.get('load_checkpoint', False) and start_epoch > 0:
        opt_path = os.path.join(config.get('checkpoint_dir', 'checkpoints_dfm'), f"dfm_optimizer_epoch_{start_epoch}.pth")
        if os.path.exists(opt_path):
            try: optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE)); print(f"Loaded DFM Optimizer state.")
            except Exception as e: print(f"Error loading DFM optimizer state: {e}.")
        else: print("DFM Optimizer state not found.")


    # 3. Dataset and Dataloader (same)
    train_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'], split='train',
        preload_to_ram=config.get('preload_data', False),
        image_size=config.get('image_size', None)
    )
    val_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'], split='val',
        preload_to_ram=False, image_size=config.get('image_size', None)
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=config.get('num_workers',0), pin_memory=config.get('pin_memory',False), drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config['val_batch_size'], shuffle=False,
        num_workers=config.get('num_workers',0), pin_memory=config.get('pin_memory',False), drop_last=True
    )

    # 4. Metrics (same)
    psnr_val_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)

    # --- Training Loop ---
    flow_model.train()
    global_step = start_epoch * len(train_loader)
    T_end = config.get('fm_t_end', 1.0) # Should be consistent with continuous FM
    num_discrete_steps_train = config.get('dfm_training_steps', 50) # K for training
    print(f"Starting DFM training from Epoch {start_epoch + 1} with K={num_discrete_steps_train} training steps.")

    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(train_loader, desc=f"DFM Epoch {epoch+1}/{config['num_epochs']}", leave=True)
        epoch_loss_sum = 0.0

        for batch_idx, (rgb_condition, real_nir_x1) in enumerate(progress_bar):
            optimizer.zero_grad()
            rgb_condition = rgb_condition.to(DEVICE)
            real_nir_x1 = real_nir_x1.to(DEVICE)
            current_batch_size = real_nir_x1.shape[0]

            # 1. Sample discrete time step k ~ U{1, ..., K_train}
            # Convert k to float time t_k for path interpolation and model input
            # k_batch will be [B], values from 1 to num_discrete_steps_train
            k_batch = torch.randint(1, num_discrete_steps_train + 1, (current_batch_size,), device=DEVICE)
            
            # time_t_float_batch = (k_batch.float() / num_discrete_steps_train) * T_end
            # Small correction: if t_k is k/K * T, then t_0 = 0, t_K = T.
            # If we use k from 1 to K, then t_k represents the *end* of the k-th interval.
            # For path x_t = (1-t)x0 + t*x1, we need t in [0, T_end].
            # Let's use t_values based on k, ensuring they are in [~eps, T_end]
            # If k is sampled, t can be (k-1)/(K-1) * T for k in 1..K, or k/K * T for k in 0..K-1 for intervals
            # Or, more simply, sample t ~ U[eps, T_end] as before, and the model learns v(x_t, t)
            # The "discreteness" here is more about the *conceptual* framework of a fixed K,
            # and then a sampler that uses that K.
            # For DFM-SF, the paper implies training on specific discrete t_k.
            # Let's sample t from a discrete set of K points.
            # The paper (Appendix D.1) suggests sampling t uniformly from {1/N, ..., N/N} (or {t_i})
            # and then constructing x_t.
            
            # Simplest approach: sample continuous t, then use discrete sampler.
            # True DFM-SF: sample k, then t_k for training.
            time_t_float_batch = (k_batch.float() / num_discrete_steps_train) * T_end
            # To avoid t=0 if k can be 0, or issues with embedding if not strictly positive.
            # Since k is from 1 to K, time_t_float_batch is in [delta_t, T_end]. This is fine.

            # 2. Sample initial noise x0 ~ N(0,1)
            noise_x0 = torch.randn_like(real_nir_x1)

            # 3. Construct x_t on the path from x0 to x1 at time t_k
            # x_t = (1-t)x0 + t*x1. Ensure time_t_float_batch is correctly shaped.
            time_expanded = time_t_float_batch.view(-1, 1, 1, 1)
            x_t_k = (1 - time_expanded) * noise_x0 + time_expanded * real_nir_x1
            
            # 4. Define target vector field u_t = x1 - x0 (for linear path, sigma=0)
            target_vector_field_u_t_k = real_nir_x1 - noise_x0
            
            # 5. Predict vector field v_theta(x_t_k, t_k, c)
            # The model needs float time, which time_t_float_batch is.
            predicted_vector_field_v_theta = flow_model(x_t_k, time_t_float_batch, rgb_condition)

            # 6. Loss function (MSE)
            loss = F.mse_loss(predicted_vector_field_v_theta, target_vector_field_u_t_k)
            
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/Step_FM_MSE', loss.item(), global_step)
            epoch_loss_sum += loss.item()
            if batch_idx % config.get('log_freq_batch', 100) == 0:
                progress_bar.set_postfix({'FM MSE Loss': f'{loss.item():.4f}'})
            
            # --- Image Logging (using discrete sampler) ---
            if global_step % config.get('log_image_freq_step', 1000) == 0:
                flow_model.eval()
                with torch.no_grad():
                    num_log = min(4, current_batch_size) # Or config['val_batch_size']
                    log_rgb_cond = rgb_condition[:num_log]
                    log_real_nir_x1 = real_nir_x1[:num_log]
                    log_initial_noise = torch.randn_like(log_real_nir_x1[:num_log])
                    
                    log_sampling_steps = config.get('dfm_sampling_steps_log', 20)

                    generated_nir_log = sample_with_discrete_steps(
                        flow_model, log_initial_noise, log_rgb_cond,
                        log_sampling_steps, T_end, DEVICE
                    )
                    writer.add_image('Input/RGB_Condition', torchvision.utils.make_grid(log_rgb_cond, normalize=True, value_range=(-1,1)), global_step)
                    writer.add_image('Target/Real_NIR_x1', torchvision.utils.make_grid(log_real_nir_x1, normalize=True, value_range=(-1,1)), global_step)
                    writer.add_image('Generated/FlowMatch_NIR_x1', torchvision.utils.make_grid(generated_nir_log, normalize=True, value_range=(-1,1)), global_step)
                    writer.flush()
                flow_model.train()
            global_step += 1
        
        avg_loss_epoch = epoch_loss_sum / len(train_loader)
        writer.add_scalar('Loss_epoch/Avg_FM_MSE', avg_loss_epoch, epoch + 1)
        print(f"End of DFM Epoch {epoch+1} -> Avg FM MSE Loss: {avg_loss_epoch:.4f}")

        # --- Validation ---
        if val_loader is not None and (epoch + 1) % config.get('val_epoch_freq', 1) == 0:
            try:
                current_psnr, current_ssim, current_lpips = evaluate_on_validation_set_discrete_fm(
                    flow_model, val_loader, DEVICE, 
                    psnr_val_metric, ssim_val_metric, lpips_val_metric, epoch + 1,
                    config
                )
                writer.add_scalar('Val_epoch/PSNR', current_psnr, epoch + 1)
                writer.add_scalar('Val_epoch/SSIM', current_ssim, epoch + 1)
                writer.add_scalar('Val_epoch/LPIPS', current_lpips, epoch + 1)
                print(f"DFM Epoch {epoch+1} Val Metrics -> PSNR: {current_psnr:.4f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")
            except Exception as e: print(f"Error during DFM validation: {e}")

        # --- Checkpointing (save with "dfm_" prefix) ---
        if (epoch + 1) % config.get('save_epoch_freq', 5) == 0 or (epoch + 1) == config['num_epochs']:
            save_dir = config.get('checkpoint_dir', 'checkpoints_dfm')
            os.makedirs(save_dir, exist_ok=True)
            model_to_save = flow_model.module if isinstance(flow_model, nn.DataParallel) else flow_model
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, f"dfm_model_epoch_{epoch+1}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"dfm_optimizer_epoch_{epoch+1}.pth"))
            print(f"Saved DFM checkpoint for epoch {epoch+1} to {save_dir}")

    print("DFM Training finished.")
    writer.close()

if __name__ == "__main__":
    torch.manual_seed(42) # Ensure reproducibility
    
    # Configuration for Discrete Flow Matching
    config_dfm = {
        'learning_rate': 1e-4,
        'beta1': 0.9, 'beta2': 0.999,
        'batch_size': 16, # Adjust based on GPU memory
        'val_batch_size': 8, # Adjust based on GPU memory
        'num_epochs': 200,
        'root_dir': "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons", # !!! CHANGE PATH !!!
        'device_preference': 'cuda',
        'image_size': 128, # Start with smaller size for DFM testing
        'preload_data': False, 'num_workers': 4, 'pin_memory': True,

        # Model parameters (same as continuous version)
        'nir_channels': 1, 'rgb_channels': 3,
        'unet_base_channels': 64, 'unet_num_levels': 3, # Adjusted for 128px
        'time_emb_dim': 256,
        'continuous_time_emb_max_period': 1000.0,

        # Discrete Flow Matching specific parameters
        'fm_t_end': 1.0, # Final time T for flow
        'dfm_training_steps': 50,    # K_train: Number of discrete time steps to sample from during training
        'dfm_sampling_steps': 50,    # K_sample: Number of steps for Euler sampler during validation/logging
        'dfm_sampling_steps_log': 20,# K_sample for logging images (can be fewer)

        # Logging and Checkpointing
        'log_dir': 'runs/discrete_flow_matching_v1',
        'checkpoint_dir': 'models/discrete_flow_matching_v1',
        'log_freq_batch': 50,
        'log_image_freq_step': 500,
        'save_epoch_freq': 10,
        'val_epoch_freq': 5,
        'load_checkpoint': False, 'checkpoint_load_epoch': 7,
    }
    
    print(f"DFM Log directory: {config_dfm.get('log_dir')}")
    print(f"DFM Checkpoint directory: {config_dfm.get('checkpoint_dir')}")
    os.makedirs(config_dfm['log_dir'], exist_ok=True)
    os.makedirs(config_dfm['checkpoint_dir'], exist_ok=True)
    
    # Log config to TensorBoard (optional, simple version)
    try:
        import json
        # writer_main = SummaryWriter(log_dir=config_dfm['log_dir']) # Create writer here if needed for hparams
        # config_str = json.dumps(config_dfm, indent=4, sort_keys=True, default=str)
        # writer_main.add_text("Experiment_Config_DFM", f"<pre>{config_str}</pre>", 0)
        # writer_main.close() # Close if only used for config text
        pass # Assuming train_discrete_flowmatching handles its own writer for metrics
    except Exception as e: print(f"Could not log DFM config to text: {e}")

    train_discrete_flowmatching(config_dfm) 