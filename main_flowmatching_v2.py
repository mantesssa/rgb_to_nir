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

# Импорты из вашего нового файла flow_matching_models.py
from flow_matching_models import (
    VectorFieldUNet,
    sample_with_ode_solver,
    # SinusoidalPositionalEmbedding # Если понадобится напрямую
)

# Попытка импортировать компоненты из библиотеки flow_matching от Facebook Research
# Пользователю нужно будет установить: pip install flow_matching
# и, возможно, torchdiffeq: pip install torchdiffeq
FM_AVAILABLE = False
try:
    # Примерные импорты, которые могут понадобиться. 
    # Точные будут зависеть от того, как мы решим их использовать.
    # from flow_matching.models.utils import DBlock, UBlock, TimestepEmbedding # Если используем их архитектуру
    from flow_matching.path import ConditionalLinearPath # Пример пути
    # from flow_matching.losses import ConditionalFlowMatchingLoss # Пример функции потерь
    # from flow_matching.solvers import Euler, RK4 # Если используем их солверы вместо torchdiffeq напрямую
    FM_AVAILABLE = True
    print("Facebook Research 'flow_matching' library components seem to be available.")
except ImportError:
    print("Warning: Facebook Research 'flow_matching' library is not installed or some components are missing.")
    print("Please install it with: pip install flow_matching")
    print("The script will try to run with simplified custom implementations where possible,")
    print("but using the library is recommended for full functionality and correctness.")
    ConditionalLinearPath = None # Заглушка

# --- Класс датасета (аналогично main_diffusion.py) ---
class RGBNIRPairedDataset(Dataset):
    def __init__(self, root_dir, split='train', preload_to_ram=False, transform=None, image_size=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.rgb_dir = os.path.join(root_dir, f"{split}_A") # RGB images (condition)
        self.nir_dir = os.path.join(root_dir, f"{split}_B") # NIR images (target x1)
        self.image_size = image_size
        self.resize_transform = None
        if self.image_size:
            interpolation = Image.Resampling.LANCZOS if image_size > 256 else Image.Resampling.BILINEAR
            self.resize_transform = transforms.Resize((self.image_size, self.image_size), interpolation=interpolation)

        if not os.path.exists(self.rgb_dir): raise FileNotFoundError(f"RGB dir {self.rgb_dir} not found!")
        if not os.path.exists(self.nir_dir): raise FileNotFoundError(f"NIR dir {self.nir_dir} not found!")

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.nir_files = sorted([f for f in os.listdir(self.nir_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        assert len(self.rgb_files) == len(self.nir_files), "File counts mismatch"

        self.preload = preload_to_ram
        self.data_cache = []
        if self.preload: # ... (код прелоадинга такой же)
            print(f"Preloading {split} data to RAM...")
            for idx in tqdm(range(len(self.rgb_files)), desc=f"Preloading {split}"):
                self.data_cache.append(self._load_and_process_item(idx))
            print("Preloading complete.")

    def _load_and_process_item(self, idx):
        rgb_file = self.rgb_files[idx]
        nir_file = self.nir_files[idx]
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        nir_path = os.path.join(self.nir_dir, nir_file)
        try:
            rgb_image_pil = Image.open(rgb_path).convert('RGB')
            nir_image_pil = Image.open(nir_path).convert('L')
            if self.resize_transform:
                rgb_image_pil = self.resize_transform(rgb_image_pil)
                nir_image_pil = self.resize_transform(nir_image_pil)
            
            rgb_tensor_cond = transforms.ToTensor()(rgb_image_pil) * 2.0 - 1.0 # To [-1, 1]
            nir_tensor_x1 = transforms.ToTensor()(nir_image_pil) * 2.0 - 1.0   # To [-1, 1]
            return rgb_tensor_cond, nir_tensor_x1
        except Exception as e: # ... (обработка ошибок такая же)
            print(f"Error loading/processing image {rgb_file} or {nir_file} at index {idx}: {e}")
            raise e

    def __len__(self): return len(self.rgb_files)
    def __getitem__(self, idx):
        if self.preload and idx < len(self.data_cache): return self.data_cache[idx]
        return self._load_and_process_item(idx)

# --- Вспомогательная функция для оценки (Flow Matching) ---
@torch.no_grad()
def evaluate_on_validation_set_flowmatching(
    flow_model, val_loader, device, 
    psnr_metric_obj, ssim_metric_obj, lpips_metric_obj, epoch_num,
    config # Добавляем config для параметров сэмплинга
):
    flow_model.eval()
    psnr_metric_obj.reset(); ssim_metric_obj.reset(); lpips_metric_obj.reset()

    for rgb_condition_val, real_nir_x1_val in tqdm(val_loader, desc=f"Evaluating FlowMatching (Epoch {epoch_num})", leave=False):
        rgb_condition_val = rgb_condition_val.to(device)
        real_nir_x1_val = real_nir_x1_val.to(device)
        batch_size, nir_c, H, W = real_nir_x1_val.shape

        # Начальный шум для генерации (x0 ~ N(0,1))
        initial_noise_x0 = torch.randn_like(real_nir_x1_val, device=device)
        
        time_span = torch.tensor([0.0, config.get('fm_t_end', 1.0)], device=device)
        num_sampling_steps = config.get('fm_sampling_steps', 50) # Количество шагов для ОДУ решателя

        generated_nir_x1 = sample_with_ode_solver(
            model=flow_model,
            initial_noise_x0=initial_noise_x0,
            condition_rgb=rgb_condition_val,
            t_span=time_span,
            num_eval_points=num_sampling_steps, # Для простоты, num_eval_points = num_sampling_steps
            device=device,
            solver_method=config.get('fm_solver_method', 'dopri5')
        )
        # Денормализация и расчет метрик (аналогично diffusion)
        real_nir_norm = (real_nir_x1_val + 1) / 2.0
        gen_nir_norm = (generated_nir_x1 + 1) / 2.0
        real_nir_norm = torch.nan_to_num(real_nir_norm.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)
        gen_nir_norm = torch.nan_to_num(gen_nir_norm.clamp(0,1), nan=0.0, posinf=1.0, neginf=0.0)

        psnr_metric_obj.update(gen_nir_norm, real_nir_norm)
        ssim_metric_obj.update(gen_nir_norm, real_nir_norm)
        if real_nir_norm.size(1) == 1:
            real_lpips = real_nir_norm.repeat(1,3,1,1); gen_lpips = gen_nir_norm.repeat(1,3,1,1)
        else: real_lpips = real_nir_norm; gen_lpips = gen_nir_norm
        lpips_metric_obj.update(gen_lpips, real_lpips)

    epoch_psnr = psnr_metric_obj.compute().item()
    epoch_ssim = ssim_metric_obj.compute().item()
    epoch_lpips = lpips_metric_obj.compute().item()
    flow_model.train()
    return epoch_psnr, epoch_ssim, epoch_lpips

# --- Основная функция обучения Flow Matching ---
def train_flowmatching(config):
    writer = SummaryWriter(log_dir=config['log_dir'])
    # ... (логирование конфигурации аналогично)
    try:
        import json
        config_str = json.dumps(config, indent=4, sort_keys=True, default=str)
        writer.add_text("Experiment_Config", f"<pre>{config_str}</pre>", 0)
    except Exception as e: print(f"Could not log config: {e}")
    hparam_dict_to_log = {k: str(v) if not isinstance(v, (int, float, str, bool, type(None))) else v for k, v in config.items() if k != 'device_preference'}

    DEVICE = "cuda" if torch.cuda.is_available() and config.get('device_preference') == 'cuda' else "cpu"
    print(f"Using device: {DEVICE}")
    start_epoch = 0

    # 1. Инициализация модели U-Net для предсказания векторного поля
    flow_model_raw = VectorFieldUNet(
        nir_channels=config.get('nir_channels', 1),
        rgb_channels=config.get('rgb_channels', 3),
        out_channels_vector_field=config.get('nir_channels', 1),
        base_channels=config.get('unet_base_channels', 64),
        num_levels=config.get('unet_num_levels', 4),
        time_emb_dim=config.get('time_emb_dim', 256), # Размер эмбеддинга времени
        continuous_time_emb_max_period=config.get('continuous_time_emb_max_period', 1000.0) # Added for new embedding
    )
    # ... (загрузка чекпоинта модели, DataParallel, перемещение на DEVICE - аналогично diffusion)
    if config.get('load_checkpoint', False):
        load_epoch = config.get('checkpoint_load_epoch', 0)
        if load_epoch > 0:
            model_path = os.path.join(config.get('checkpoint_dir', 'checkpoints_fm'), f"flow_model_epoch_{load_epoch}.pth")
            if os.path.exists(model_path):
                try:
                    flow_model_raw.load_state_dict(torch.load(model_path, map_location='cpu'))
                    print(f"Loaded Flow Model weights from epoch {load_epoch}"); start_epoch = load_epoch
                except Exception as e: print(f"Error loading model checkpoint: {e}. Starting fresh."); start_epoch = 0
            else: print(f"Model checkpoint not found. Starting fresh."); start_epoch = 0
        else: print("load_checkpoint is True, but checkpoint_load_epoch invalid. Starting fresh."); start_epoch = 0
    
    flow_model = flow_model_raw
    if DEVICE == "cuda" and torch.cuda.device_count() > 1:
        flow_model = nn.DataParallel(flow_model_raw); print(f"Using {torch.cuda.device_count()} GPUs.")
    flow_model.to(DEVICE)

    # 2. Оптимизатор (аналогично diffusion)
    optimizer = optim.Adam(flow_model.parameters(), lr=config['learning_rate'], betas=(config.get('beta1', 0.9), config.get('beta2', 0.999)))
    # ... (загрузка состояния оптимизатора - аналогично diffusion)
    if config.get('load_checkpoint', False) and start_epoch > 0:
        opt_path = os.path.join(config.get('checkpoint_dir', 'checkpoints_fm'), f"optimizer_epoch_{start_epoch}.pth")
        if os.path.exists(opt_path):
            try: optimizer.load_state_dict(torch.load(opt_path, map_location=DEVICE)); print(f"Loaded Optimizer state.")
            except Exception as e: print(f"Error loading optimizer state: {e}.")
        else: print("Optimizer state not found.")

    # --- Probability Path (пример: линейный путь) ---
    # Если ConditionalLinearPath из flow_matching library доступен и подходит:
    # if ConditionalLinearPath is not None and FM_AVAILABLE:
    #     prob_path = ConditionalLinearPath(sigma=config.get('fm_path_sigma', 0.0)) # sigma=0 для детерминированного пути
    #     print("Using ConditionalLinearPath from flow_matching library.")
    # else:
    #     print("Using simplified custom linear path logic.")
    #     prob_path = None # Будем реализовывать логику пути вручную ниже

    # --- Dataset initialization ---
    train_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'],
        split='train',
        preload_to_ram=config.get('preload_data', False),
        image_size=config.get('image_size', None)
        )
    val_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'], 
        split='val', 
        preload_to_ram=False,
        image_size=config.get('image_size', None)
    )

    # --- Dataloader initialization ---
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config.get('num_workers',0),
        pin_memory=config.get('pin_memory',False),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config.get('num_workers',0),
        pin_memory=config.get('pin_memory',False),
        drop_last=True
    )


    # --- Метрики (аналогично) ---
    psnr_val_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE)

    flow_model.train()
    global_step = start_epoch * len(train_loader)
    print(f"Starting training from Epoch {start_epoch + 1}")
    # ... (инициализация переменных для метрик) ...
    # last_epoch_psnr, last_epoch_ssim, last_epoch_lpips = float('NaN'), float('NaN'), float('NaN')
    avg_loss_epoch = float('NaN')

    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=True)
        epoch_loss_sum = 0.0

        for batch_idx, (rgb_condition, real_nir_x1) in enumerate(progress_bar):
            optimizer.zero_grad()
            rgb_condition = rgb_condition.to(DEVICE)
            real_nir_x1 = real_nir_x1.to(DEVICE)
            current_batch_size = real_nir_x1.shape[0]

            # 1. Сэмплирование времени t ~ U(0, T_end) или из расписания
            # T_end обычно 1.0 для Flow Matching
            t_end = config.get('fm_t_end', 1.0)
            # Случайное время t для каждого примера в батче, из [eps, T_end] чтобы избежать t=0
            # (некоторые формулировки Flow Matching используют t в [eps, 1] или [0,1] по-разному)
            # Библиотека flow_matching может иметь свои утилиты для сэмплирования t.
            time_t = torch.rand(current_batch_size, device=DEVICE) * t_end # t в [0, T_end], float
            # Для эмбеддинга, который ожидает целые числа, нужно масштабирование.
            # Это временное решение, лучше использовать эмбеддинг, работающий с float [0,1]
            # или адаптировать SinusoidalPositionalEmbedding, или использовать утилиту из flow_matching lib.
            # time_t_for_embedding = (time_t * config.get('fm_time_scale_for_sin_emb', 1000)).long() # No longer needed

            # 2. Сэмплирование начальной точки x0 (например, из N(0,1))
            noise_x0 = torch.randn_like(real_nir_x1) # Гауссовский шум

            # 3. Построение x_t на пути от x0 к x1
            # Простейший линейный путь: x_t = (1-t)*x0 + t*x1
            # Вектор t нужно расширить для поэлементного умножения: [B] -> [B,1,1,1]
            time_t_expanded = time_t.view(-1, 1, 1, 1)
            x_t = (1 - time_t_expanded) * noise_x0 + time_t_expanded * real_nir_x1
            
            # 4. Определение целевого векторного поля u_t
            # Для линейного пути: u_t = x1 - x0
            target_vector_field_u_t = real_nir_x1 - noise_x0
            
            # 5. Предсказание векторного поля моделью v_theta(x_t, t, c)
            predicted_vector_field_v_theta = flow_model(x_t, time_t, rgb_condition) # Pass float time_t directly

            # 6. Функция потерь (например, L2 loss)
            # || v_theta(x_t, t, c) - u_t ||^2
            loss = F.mse_loss(predicted_vector_field_v_theta, target_vector_field_u_t)
            
            loss.backward()
            optimizer.step()

            writer.add_scalar('Loss/Step_FM_MSE', loss.item(), global_step)
            epoch_loss_sum += loss.item()
            if batch_idx % config.get('log_freq_batch', 100) == 0:
                progress_bar.set_postfix({'FM MSE Loss': f'{loss.item():.4f}'})
            
            # --- Логирование изображений (аналогично diffusion) ---
            if global_step % config.get('log_image_freq_step', 1000) == 0:
                flow_model.eval()
                with torch.no_grad():
                    num_log = min(4, current_batch_size)
                    log_rgb_cond = rgb_condition[:num_log]
                    log_real_nir_x1 = real_nir_x1[:num_log]
                    log_initial_noise = torch.randn_like(log_real_nir_x1)
                    
                    log_time_span = torch.tensor([0.0, config.get('fm_t_end', 1.0)], device=DEVICE)
                    log_sampling_steps = config.get('fm_sampling_steps_log', 20) 

                    generated_nir_log = sample_with_ode_solver(
                        flow_model, log_initial_noise, log_rgb_cond, 
                        log_time_span, log_sampling_steps, DEVICE,
                        config.get('fm_solver_method', 'dopri5')
                    )
                    writer.add_image('Input/RGB_Condition', torchvision.utils.make_grid(log_rgb_cond, normalize=True, value_range=(-1,1)), global_step)
                    writer.add_image('Target/Real_NIR_x1', torchvision.utils.make_grid(log_real_nir_x1, normalize=True, value_range=(-1,1)), global_step)
                    writer.add_image('Generated/FlowMatch_NIR_x1', torchvision.utils.make_grid(generated_nir_log, normalize=True, value_range=(-1,1)), global_step)
                    writer.flush()
                flow_model.train()
            global_step += 1
        
        avg_loss_epoch = epoch_loss_sum / len(train_loader)
        writer.add_scalar('Loss_epoch/Avg_FM_MSE', avg_loss_epoch, epoch + 1)
        print(f"End of Epoch {epoch+1} -> Avg FM MSE Loss: {avg_loss_epoch:.4f}")

        # --- Оценка на валидационном наборе (аналогично diffusion) ---
        if val_loader is not None and (epoch + 1) % config.get('val_epoch_freq', 1) == 0:
            try:
                current_psnr, current_ssim, current_lpips = evaluate_on_validation_set_flowmatching(
                    flow_model, val_loader, DEVICE, 
                    psnr_val_metric, ssim_val_metric, lpips_val_metric, epoch + 1,
                    config # Передаем config
                )
                writer.add_scalar('Val_epoch/PSNR', current_psnr, epoch + 1); writer.add_scalar('Val_epoch/SSIM', current_ssim, epoch + 1); writer.add_scalar('Val_epoch/LPIPS', current_lpips, epoch + 1)
                print(f"Epoch {epoch+1} Val Metrics -> PSNR: {current_psnr:.4f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")
                # last_epoch_psnr, last_epoch_ssim, last_epoch_lpips = current_psnr, current_ssim, current_lpips # Сохраняем для HParams
            except Exception as e: print(f"Error during validation: {e}")
        # ... (чекпоинты и HParams - аналогично diffusion) ...

        if (epoch + 1) % config.get('save_epoch_freq', 5) == 0 or (epoch + 1) == config['num_epochs']:
            save_dir = config.get('checkpoint_dir', 'checkpoints_fm'); os.makedirs(save_dir, exist_ok=True)
            model_to_save = flow_model.module if isinstance(flow_model, nn.DataParallel) else flow_model
            torch.save(model_to_save.state_dict(), os.path.join(save_dir, f"flow_model_epoch_{epoch+1}.pth"))
            torch.save(optimizer.state_dict(), os.path.join(save_dir, f"optimizer_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint for epoch {epoch+1}")

    print("Training finished.")
    # final_metrics = {
    #     'final_avg_fm_mse_loss': avg_loss_epoch,
    #     'completed_epochs': epoch + 1 if 'epoch' in locals() else 0,
    #     'final_psnr': last_epoch_psnr, 'final_ssim': last_epoch_ssim, 'final_lpips': last_epoch_lpips
    # }
    # try: writer.add_hparams(hparam_dict_to_log, final_metrics)
    # except Exception as e: print(f"Could not write HParams: {e}")
    writer.close()

if __name__ == "__main__":
    torch.manual_seed(42)
    config = {
        'learning_rate': 1e-4,
        'beta1': 0.9, 'beta2': 0.999,
        'batch_size': 32, # Может потребоваться корректировка
        'val_batch_size': 16,
        'num_epochs': 200, # Flow Matching может требовать много эпох
        'root_dir': "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons", # !!! ИЗМЕНИТЕ ПУТЬ !!!
        'device_preference': 'cuda',
        'image_size': 256, # Начните с меньшего размера
        'preload_data': False, 'num_workers': 4, 'pin_memory': True,

        'nir_channels': 1, 'rgb_channels': 3,
        'unet_base_channels': 64, 'unet_num_levels': 4, 'time_emb_dim': 256,
        
        # Параметры Flow Matching
        'fm_t_end': 1.0, # Конечное время для интегрирования и обучения
        'continuous_time_emb_max_period': 1000.0, # For ContinuousSinusoidalPositionalEmbedding, can be tuned
        # 'fm_time_scale_for_sin_emb': 1000, # No longer needed for SinusoidalPositionalEmbedding
        # 'fm_path_sigma': 0.0, # Для ConditionalLinearPath, если используется из библиотеки
        'fm_sampling_steps': 50, # Количество шагов для ОДУ-решателя при валидации
        'fm_sampling_steps_log': 20, # Количество шагов для ОДУ-решателя при логировании изображений
        'fm_solver_method': 'dopri5', # Метод ОДУ решателя (для torchdiffeq)

        'log_dir': 'runs/flow_matching_v3',
        'checkpoint_dir': 'models/flow_matching_v3',
        'log_freq_batch': 100,
        'log_image_freq_step': 1000, # Генерация дольше, логируем реже
        'save_epoch_freq': 10,
        'val_epoch_freq': 5,
        'load_checkpoint': False, 'checkpoint_load_epoch': 0,
    } # CUDA_VISIBLE_DEVICES=3 nohup python main_flowmatching.py > main_flowmatching.log 2>&1 &
    print(f"Log directory: {config.get('log_dir')}")
    print(f"Checkpoint directory: {config.get('checkpoint_dir')}")
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    train_flowmatching(config) 