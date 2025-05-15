# RGB_to_NIR/main_small.py
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
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity # Более точный импорт для LPIPS

# --- Класс датасета (RGB -> HSV для генератора, RGB для логгирования, NIR для таргета) ---
class RGBNIRPairedDataset(Dataset):
    def __init__(self, root_dir, split='train', preload_to_ram=False, transform=None, image_size=None):
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.rgb_dir = os.path.join(root_dir, f"{split}_A") # RGB images
        self.nir_dir = os.path.join(root_dir, f"{split}_B") # NIR images
        self.image_size = image_size
        self.resize_transform = None
        if self.image_size:
            interpolation = Image.Resampling.LANCZOS if image_size > 256 else Image.Resampling.BILINEAR
            self.resize_transform = transforms.Resize((self.image_size, self.image_size), interpolation=interpolation)

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB directory {self.rgb_dir} not found!")
        if not os.path.exists(self.nir_dir):
            raise FileNotFoundError(f"NIR directory {self.nir_dir} not found!")

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])
        self.nir_files = sorted([f for f in os.listdir(self.nir_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))])

        assert self.rgb_files == self.nir_files, \
            f"RGB and NIR files mismatch! RGB count: {len(self.rgb_files)}, NIR count: {len(self.nir_files)}."

        self.transform = transform # Likely unused based on previous config
        self.preload = preload_to_ram
        self.data_cache = [] # Renamed from self.data to avoid confusion

        if self.preload:
            print(f"Preloading {split} data to RAM (RGB -> HSV conversion)...")
            for idx in tqdm(range(len(self.rgb_files)), desc=f"Preloading {split}"):
                self.data_cache.append(self._load_and_process_item(idx))
            print("Preloading complete.")

    def _load_and_process_item(self, idx):
        rgb_file = self.rgb_files[idx]
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        nir_path = os.path.join(self.nir_dir, rgb_file) # Assuming NIR files have same names as RGB

        try:
            # Load RGB image
            rgb_image_pil = Image.open(rgb_path).convert('RGB')
            # Convert to HSV for generator input
            # hsv_image_pil = rgb_image_pil.convert('HSV')
            # hsv_image_pil = rgb_image_pil

            # Load NIR image
            nir_image_pil = Image.open(nir_path).convert('L')

            # Apply resizing if specified
            if self.resize_transform:
                rgb_image_pil_resized = self.resize_transform(rgb_image_pil)
                # hsv_image_pil_resized = self.resize_transform(hsv_image_pil)
                nir_image_pil_resized = self.resize_transform(nir_image_pil)
            else:
                rgb_image_pil_resized = rgb_image_pil
                # hsv_image_pil_resized = hsv_image_pil
                nir_image_pil_resized = nir_image_pil

            # Process HSV for generator input (normalized to [-1, 1])
            # PIL HSV channels are all in [0, 255]
            # hsv_array = np.array(hsv_image_pil_resized, dtype=np.float32)
            # hsv_tensor = torch.from_numpy(hsv_array).permute(2, 0, 1) / 127.5 - 1.0

            # Process original RGB for logging (normalized to [-1, 1])
            rgb_array_log = np.array(rgb_image_pil_resized, dtype=np.float32)
            rgb_tensor_log = torch.from_numpy(rgb_array_log).permute(2, 0, 1) / 127.5 - 1.0

            # Process NIR (normalized to [-1, 1])
            nir_array = np.array(nir_image_pil_resized, dtype=np.float32)
            nir_tensor = torch.from_numpy(nir_array).unsqueeze(0) / 127.5 - 1.0
            
            # return hsv_tensor, rgb_tensor_log, nir_tensor
            return rgb_tensor_log, rgb_tensor_log, nir_tensor


        except FileNotFoundError as e:
            print(f"Error: File not found for index {idx}, RGB: {rgb_path} or NIR: {nir_path}")
            raise e
        except Exception as e:
            print(f"Error loading/processing image {rgb_file} at index {idx}: {e}")
            raise e


    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.data_cache):
            try:
                return self.data_cache[idx]
            except IndexError:
                 print(f"IndexError: Could not retrieve preloaded data for index {idx}. Falling back to dynamic loading.")
        return self._load_and_process_item(idx)


# --- Вспомогательные модули для U-Net (без изменений) ---
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# --- Архитектура Генератора (U-Net Small) ---
# Принимает 3-канальный HSV, выдает 1-канальный NIR
class GeneratorUNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, num_levels=4): # in_channels is 3 for HSV
        super().__init__()
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        current_c = in_channels
        for i in range(num_levels):
            output_c = min(base_channels * (2**i), 512)
            use_dropout = 0.5 if i >= (num_levels - 2) else 0.0
            self.down_layers.append(UNetDown(current_c, output_c, normalize=(i!=0), dropout=use_dropout))
            current_c = output_c
        self.bottleneck = UNetDown(current_c, current_c, normalize=False, dropout=0.5)
        current_c_after_bottleneck = current_c
        current_c = current_c_after_bottleneck
        for i in range(num_levels):
            level_idx = num_levels - 1 - i
            output_c_convT = min(base_channels * (2**level_idx), 512)
            if level_idx == 0 and num_levels > 0: output_c_convT = base_channels
            skip_module_out_channels = self.down_layers[level_idx].model[0].out_channels
            use_dropout = 0.5 if i < (num_levels // 2 + 1) else 0.0
            self.up_layers.append(UNetUp(current_c, output_c_convT, dropout=use_dropout))
            current_c = output_c_convT + skip_module_out_channels
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(current_c, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        skip_connections = []
        for layer in self.down_layers:
            x = layer(x)
            skip_connections.append(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.up_layers):
            x = layer(x, skip_connections[i])
        return self.final_up(x)

# --- Архитектура Дискриминатора (Small) ---
# Принимает 3-канальный HSV (как условие) и 1-канальный NIR (реальный/фейковый)
class DiscriminatorSmall(nn.Module):
    # in_channels_condition: 3 for HSV
    # in_channels_target: 1 for NIR
    def __init__(self, in_channels_condition=3, in_channels_target=1, base_channels=32, num_levels=3):
        super().__init__()
        input_channels = in_channels_condition + in_channels_target
        layers = []
        current_c = input_channels
        ndf = base_channels
        for i in range(num_levels):
            output_c = min(ndf * (2**i), 512)
            layers.append(nn.Conv2d(current_c, output_c, kernel_size=4, stride=2, padding=1, bias=(i==0)))
            if i != 0: layers.append(nn.InstanceNorm2d(output_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_c = output_c
        output_c_s1 = min(ndf * (2**num_levels), 512)
        layers.append(nn.Conv2d(current_c, output_c_s1, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(output_c_s1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_c = output_c_s1
        layers.append(nn.Conv2d(current_c, 1, kernel_size=4, stride=1, padding=1, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, condition_hsv, target_nir): # Renamed arguments for clarity
        x = torch.cat([condition_hsv, target_nir], dim=1)
        return self.model(x)

# --- Функции потерь и Gradient Penalty ---
# condition_input здесь будет HSV
def compute_gradient_penalty(critic, real_target_nir, fake_target_nir, condition_input_hsv, device):
    batch_size = real_target_nir.size(0)
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_target_nir)
    interpolated_nir = (epsilon * real_target_nir + (1 - epsilon) * fake_target_nir).requires_grad_(True)
    current_batch_size = interpolated_nir.size(0)
    condition_input_hsv_matched = condition_input_hsv[:current_batch_size]
    interpolated_scores = critic(condition_input_hsv_matched, interpolated_nir)
    gradients = torch.autograd.grad(
        outputs=interpolated_scores, inputs=interpolated_nir,
        grad_outputs=torch.ones_like(interpolated_scores, device=device),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    gradients = gradients.view(current_batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

def critic_loss_fn(critic_real_out, critic_fake_out, gradient_penalty, lambda_gp):
    wgan_loss = torch.mean(critic_fake_out) - torch.mean(critic_real_out)
    loss_critic = wgan_loss + lambda_gp * gradient_penalty
    return loss_critic, wgan_loss

def generator_loss_fn(critic_fake_out, fake_nir, real_nir, lambda_l1):
    loss_G_gan = -torch.mean(critic_fake_out)
    loss_G_l1 = F.l1_loss(fake_nir, real_nir)
    total_loss_G = loss_G_gan + lambda_l1 * loss_G_l1
    return total_loss_G, loss_G_gan, loss_G_l1

# --- Вспомогательная функция для оценки на валидационном наборе ---
def evaluate_on_validation_set(generator_model, val_loader, device, psnr_metric_obj, ssim_metric_obj, lpips_metric_obj):
    generator_model.eval() # Переводим генератор в режим оценки
    
    # Сбрасываем состояние метрик перед новым вычислением
    psnr_metric_obj.reset()
    ssim_metric_obj.reset()
    lpips_metric_obj.reset()

    with torch.no_grad():
        for hsv_val, _, real_nir_val in tqdm(val_loader, desc="Evaluating on val set (epoch)", leave=False):
            hsv_val = hsv_val.to(device)
            real_nir_val = real_nir_val.to(device)
            
            fake_nir_val = generator_model(hsv_val)

            # Денормализация изображений из [-1, 1] в [0, 1] для метрик
            real_nir_val_norm = (real_nir_val + 1) / 2.0
            fake_nir_val_norm = (fake_nir_val + 1) / 2.0

            real_nir_val_norm = torch.nan_to_num(real_nir_val_norm, nan=0.0, posinf=1.0, neginf=0.0)
            fake_nir_val_norm = torch.nan_to_num(fake_nir_val_norm, nan=0.0, posinf=1.0, neginf=0.0)
            
            # PSNR и SSIM работают с одноканальными изображениями
            psnr_metric_obj.update(fake_nir_val_norm, real_nir_val_norm)
            ssim_metric_obj.update(fake_nir_val_norm, real_nir_val_norm)

            # Для LPIPS расширяем до 3 каналов, повторяя существующий канал
            # Убедимся, что тензоры имеют 4 измерения [N, C, H, W]
            if real_nir_val_norm.ndim == 3: # Если [C, H, W] без батча (маловероятно здесь, но для безопасности)
                real_nir_val_norm_lpips = real_nir_val_norm.unsqueeze(0).repeat(1, 3, 1, 1)
                fake_nir_val_norm_lpips = fake_nir_val_norm.unsqueeze(0).repeat(1, 3, 1, 1)
            elif real_nir_val_norm.ndim == 4: # Если [N, C, H, W]
                if real_nir_val_norm.size(1) == 1: # Проверяем, что это действительно одноканальное изображение
                    real_nir_val_norm_lpips = real_nir_val_norm.repeat(1, 3, 1, 1)
                    fake_nir_val_norm_lpips = fake_nir_val_norm.repeat(1, 3, 1, 1)
                else: # Если уже 3 канала (не должно быть для NIR, но на всякий случай)
                    real_nir_val_norm_lpips = real_nir_val_norm
                    fake_nir_val_norm_lpips = fake_nir_val_norm
            else:
                # Обработка неожиданного количества измерений, если потребуется
                raise ValueError(f"Unexpected number of dimensions for LPIPS input: {real_nir_val_norm.ndim}")

            lpips_metric_obj.update(fake_nir_val_norm_lpips, real_nir_val_norm_lpips)

    epoch_psnr = psnr_metric_obj.compute().item()
    epoch_ssim = ssim_metric_obj.compute().item()
    epoch_lpips = lpips_metric_obj.compute().item()
    
    generator_model.train() # Возвращаем генератор в режим обучения
    return epoch_psnr, epoch_ssim, epoch_lpips

# --- Основная функция обучения ---
def train_gan(config):
    writer = SummaryWriter(log_dir=config['log_dir'])

    # --- Логирование конфигурации как текста в начале ---
    try:
        import json
        config_str = json.dumps(config, indent=4, sort_keys=True, default=str) # default=str для несериализуемых объектов
        writer.add_text("Experiment_Config", f"<pre>{config_str}</pre>", 0) # Используем pre для форматирования
    except Exception as e:
        print(f"Could not log config as text: {e}")
        # Можно добавить более простое логирование, если json.dumps не удался
        try:
            simple_config_str = "\\n".join([f"{k}: {v}" for k, v in config.items()])
            writer.add_text("Experiment_Config_Fallback", f"<pre>{simple_config_str}</pre>", 0)
        except Exception as e_simple:
            print(f"Could not log simplified config as text: {e_simple}")


    # --- Логирование гиперпараметров ---
    hparam_dict_to_log = {}
    for k, v in config.items():
        if k in ['device_preference']: # Ключи, которые точно не нужно логировать как hparams
            continue
        if isinstance(v, (int, float, str, bool)) or v is None:
            hparam_dict_to_log[k] = v
        else:
            # Для других типов (например, пути, списки, если они есть) приводим к строке
            # Это позволит их видеть в таблице, но они не будут использоваться для построения графиков в HParams по умолчанию
            hparam_dict_to_log[k] = str(v) 

    DEVICE = "cuda" # Hardcoded as per existing script state
    
    # if torch.cuda.is_available() and config['device_preference'] == 'cuda': DEVICE = "cuda"
    # elif torch.backends.mps.is_available() and config['device_preference'] == 'mps': DEVICE = "mps"
    # else: DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    start_epoch = 0 # Initialize start_epoch

    # 1. Instantiate raw models
    generator_raw = GeneratorUNetSmall(
        in_channels=3, out_channels=1, # HSV in, NIR out
        base_channels=config['g_base_channels'], num_levels=config['g_num_levels']
    )
    critic_raw = DiscriminatorSmall(
        in_channels_condition=3, in_channels_target=1, # HSV condition, NIR target
        base_channels=config['d_base_channels'], num_levels=config['d_num_levels']
    )

    # 2. Load checkpoint into raw models if specified
    # Assumes checkpoints are saved from model.module.state_dict() or non-DP model (no 'module.' prefix)
    if config.get('load_checkpoint', False):
        load_epoch = config.get('checkpoint_load_epoch')
        if load_epoch is not None and load_epoch > 0:
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
            gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{load_epoch}.pth")
            crit_path = os.path.join(checkpoint_dir, f"critic_epoch_{load_epoch}.pth")
            models_exist = os.path.exists(gen_path) and os.path.exists(crit_path)

            if models_exist:
                try:
                    generator_raw.load_state_dict(torch.load(gen_path, map_location='cpu')) # Load to CPU first
                    critic_raw.load_state_dict(torch.load(crit_path, map_location='cpu'))   # Load to CPU first
                    print(f"Successfully loaded Generator and Critic weights (raw models) from epoch {load_epoch}")
                    start_epoch = load_epoch # Set start_epoch only if model weights are successfully loaded
                except Exception as e:
                    print(f"Error loading model checkpoint from epoch {load_epoch}: {e}. Starting fresh.")
                    start_epoch = 0 # Reset if loading failed
            else:
                print(f"Model checkpoint files for epoch {load_epoch} not found. Starting fresh.")
                start_epoch = 0
        else:
            print("Warning: load_checkpoint is True, but checkpoint_load_epoch invalid. Starting fresh.")
            start_epoch = 0

    # 3. Assign to main model variables and wrap with DataParallel if multiple GPUs, then move to DEVICE
    generator = generator_raw
    critic = critic_raw

    if DEVICE == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs for DataParallel. Visible devices: {os.getenv('CUDA_VISIBLE_DEVICES', 'All/Not set')}")
        generator = nn.DataParallel(generator_raw)
        critic = nn.DataParallel(critic_raw)
    
    generator.to(DEVICE)
    critic.to(DEVICE)

    # 4. Weights initialization (applied to the underlying module if DataParallel)
    if config.get('apply_weights_init', False) and start_epoch == 0: # Only if not loading from a checkpoint
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight.data, 0.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0) # Initialize bias to 0.0
            elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0) # Initialize bias to 0.0
        
        g_to_init = generator.module if isinstance(generator, nn.DataParallel) else generator
        c_to_init = critic.module if isinstance(critic, nn.DataParallel) else critic
        g_to_init.apply(weights_init)
        c_to_init.apply(weights_init)
        print("Applied weights initialization.")

    # 5. Optimizer Initialization (uses parameters of the final model objects: generator, critic)
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
    optimizer_D = optim.Adam(critic.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

    # 6. Load optimizer states if model weights were loaded successfully from a checkpoint
    if config.get('load_checkpoint', False) and start_epoch > 0: # Check start_epoch > 0
        load_epoch = config.get('checkpoint_load_epoch') # already validated if start_epoch > 0
        checkpoint_dir = config.get('checkpoint_dir', 'checkpoints') # Use same dir
        opt_gen_path = os.path.join(checkpoint_dir, f"optimizer_G_epoch_{load_epoch}.pth")
        opt_crit_path = os.path.join(checkpoint_dir, f"optimizer_D_epoch_{load_epoch}.pth")
        optimizers_exist = os.path.exists(opt_gen_path) and os.path.exists(opt_crit_path)
        if optimizers_exist:
            try:
                optimizer_G.load_state_dict(torch.load(opt_gen_path, map_location=DEVICE))
                optimizer_D.load_state_dict(torch.load(opt_crit_path, map_location=DEVICE))
                print(f"Successfully loaded Optimizer states from epoch {load_epoch}")
            except Exception as e:
                print(f"Error loading optimizer states from epoch {load_epoch}: {e}. Optimizers will be reinitialized for epoch {start_epoch}.")
                # If optimizer loading fails, it's often safer to treat it as starting fresh for optimizers,
                # or re-evaluate if start_epoch should be reset. Current logic continues with fresh optimizers.
        else:
            print(f"Optimizer states not found for epoch {load_epoch}. Optimizers will be reinitialized.")
    
    train_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'], split='train',
        preload_to_ram=config.get('preload_data', False),
        image_size=config.get('image_size', None)
    )
    num_workers = config.get('num_workers', 0)
    if os.name == 'nt' and num_workers > 0: print("Warning: num_workers > 0 on Windows can cause issues. Setting to 0."); num_workers = 0
    pin_memory_flag = config.get('pin_memory', True) if DEVICE == 'cuda' else False
    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory_flag, drop_last=True
    )
    print(f"Dataset size: {len(train_dataset)}, Image size: {config.get('image_size', 'original')}")
    print(f"Dataloader num_workers: {num_workers}, pin_memory: {pin_memory_flag}, drop_last: {train_loader.drop_last}")

    # --- Инициализация метрик для валидации ---
    # Перемещаем на DEVICE сразу
    psnr_val_metric = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
    ssim_val_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
    # Для LPIPS net_type='vgg' часто дает лучшие результаты, чем 'alex', но требует больше ресурсов
    # normalize=True означает, что LPIPS сам обработает вход [-1,1] или [0,1]
    lpips_val_metric = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True).to(DEVICE) 
    
    # Создание валидационного DataLoader
    val_loader = None
    try:
        val_dataset = RGBNIRPairedDataset(
            root_dir=config['root_dir'], 
            split='val', 
            preload_to_ram=False,
            image_size=config.get('image_size', None)
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset, 
                batch_size=config['batch_size'], 
                shuffle=False, 
                num_workers=0 
            )
            print(f"Validation dataset loaded: {len(val_dataset)} samples.")
        else:
            print("Validation dataset is empty. Metrics (PSNR, SSIM, LPIPS) will not be calculated.")
    except FileNotFoundError:
        print(f"Validation data not found in {config['root_dir']} (expected val_A, val_B). Metrics will not be calculated.")

    if start_epoch == 0: # Add graph only when starting fresh
        try:
            # DataLoader now yields: hsv_tensor, rgb_tensor_for_log, nir_tensor
            sample_hsv, sample_rgb_log, sample_nir = next(iter(train_loader))
            sample_hsv_dev = sample_hsv.to(DEVICE)
            sample_nir_dev = sample_nir.to(DEVICE) # For critic graph
            
            # Use .module if DataParallel for add_graph for clarity, though PyTorch might handle it
            g_for_graph = generator.module if isinstance(generator, nn.DataParallel) else generator
            c_for_graph = critic.module if isinstance(critic, nn.DataParallel) else critic
            
            writer.add_graph(g_for_graph, sample_hsv_dev)
            writer.add_graph(c_for_graph, (sample_hsv_dev, sample_nir_dev)) # Critic takes (condition, target)
            print("Model graphs added to TensorBoard.")
        except Exception as e:
            print(f"Could not add model graph to TensorBoard: {e}")

    generator.train(); critic.train()
    global_step = start_epoch * len(train_loader)
    print(f"Starting training from Epoch {start_epoch + 1}")

    # Инициализация переменных для хранения метрик последней эпохи
    last_epoch_psnr, last_epoch_ssim, last_epoch_lpips = float('NaN'), float('NaN'), float('NaN')

    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=True)
        epoch_loss_D_avg, epoch_loss_G_avg, epoch_wgan_loss_avg, epoch_l1_loss_avg = 0.0, 0.0, 0.0, 0.0

        # Unpack three items: hsv_tensor, rgb_tensor_for_log, nir_tensor
        for batch_idx, (real_hsv, real_rgb_for_log, real_nir) in enumerate(progress_bar):
            real_hsv = real_hsv.to(DEVICE)
            real_rgb_for_log = real_rgb_for_log.to(DEVICE) # For logging
            real_nir = real_nir.to(DEVICE)
            current_batch_size = real_hsv.size(0)

            # Train Critic
            for _ in range(config['n_critic']):
                optimizer_D.zero_grad()
                with torch.no_grad(): fake_nir = generator(real_hsv).detach()
                critic_real_out = critic(real_hsv, real_nir)
                critic_fake_out = critic(real_hsv, fake_nir)
                # Pass real_hsv as the condition to compute_gradient_penalty
                gradient_penalty = compute_gradient_penalty(critic, real_nir, fake_nir, real_hsv, DEVICE)
                loss_D, wgan_loss = critic_loss_fn(critic_real_out, critic_fake_out, gradient_penalty, config['lambda_gp'])
                loss_D.backward(); optimizer_D.step()

            # Train Generator
            optimizer_G.zero_grad()
            fake_nir_for_gen = generator(real_hsv)
            critic_fake_out_for_gen = critic(real_hsv, fake_nir_for_gen) # Critic conditions on real_hsv
            
            lambda_l1_val = config.get('lambda_l1', 100) # Получаем из конфига, по умолчанию 100
            loss_G, loss_G_gan, loss_G_l1 = generator_loss_fn(
                critic_fake_out_for_gen, 
                fake_nir_for_gen, 
                real_nir, # Передаем real_nir для L1 loss
                lambda_l1_val
            )

            loss_G.backward(); optimizer_G.step()

            writer.add_scalar('Loss/Critic_step', loss_D.item(), global_step)
            writer.add_scalar('Loss/WGAN_D_step', wgan_loss.item(), global_step)
            writer.add_scalar('Loss/Gradient_Penalty_step', gradient_penalty.item(), global_step)
            writer.add_scalar('Loss/Generator_GAN_step', loss_G_gan.item(), global_step) # Логируем GAN часть G loss
            writer.add_scalar('Loss/Generator_L1_step', loss_G_l1.item(), global_step)   # Логируем L1 часть G loss
            writer.add_scalar('Loss/Generator_Total_step', loss_G.item(), global_step) # Логируем общую G loss
            
            epoch_loss_D_avg += loss_D.item(); 
            epoch_loss_G_avg += loss_G.item(); # Общий G loss
            epoch_wgan_loss_avg += wgan_loss.item()
            epoch_l1_loss_avg += loss_G_l1.item() # L1 loss

            if batch_idx % config.get('log_freq_batch', 50) == 0:
                progress_bar.set_postfix({'Loss D': f'{loss_D.item():.4f}', 'Loss G': f'{loss_G.item():.4f}', 'GP': f'{gradient_penalty.item():.4f}', 'L1': f'{loss_G_l1.item():.4f}'})
            
            # Log weights and gradients distributions
            if global_step % config.get('log_weights_freq_step', config.get('log_image_freq_step', 200)) == 0:
                for name, param in generator.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        writer.add_histogram(f'Generator/Weights/{name}', param.data, global_step)
                        writer.add_histogram(f'Generator/Gradients/{name}', param.grad.data, global_step)
                for name, param in critic.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        writer.add_histogram(f'Critic/Weights/{name}', param.data, global_step)
                        writer.add_histogram(f'Critic/Gradients/{name}', param.grad.data, global_step)

            if global_step % config.get('log_image_freq_step', 200) == 0:
                with torch.no_grad():
                    num_images_to_log = min(4, current_batch_size)
                    # Use real_rgb_for_log for visualizing the "input"
                    img_grid_input_rgb = torchvision.utils.make_grid(real_rgb_for_log[:num_images_to_log], normalize=True, value_range=(-1,1))
                    img_grid_real_nir = torchvision.utils.make_grid(real_nir[:num_images_to_log], normalize=True, value_range=(-1,1))
                    img_grid_fake_nir = torchvision.utils.make_grid(fake_nir_for_gen[:num_images_to_log], normalize=True, value_range=(-1,1))
                    
                    writer.add_image('Input/Original_RGB_for_Log', img_grid_input_rgb, global_step)
                    writer.add_image('Target/Real_NIR', img_grid_real_nir, global_step)
                    writer.add_image('Generated/Fake_NIR_from_HSV', img_grid_fake_nir, global_step)
                    writer.flush() # Добавить здесь
            global_step += 1
        
        avg_loss_D = epoch_loss_D_avg / len(train_loader)
        avg_loss_G = epoch_loss_G_avg / len(train_loader)
        avg_wgan_loss = epoch_wgan_loss_avg / len(train_loader)
        avg_l1_loss = epoch_l1_loss_avg / len(train_loader) # Средний L1 loss за эпоху

        writer.add_scalar('Loss_epoch/Critic', avg_loss_D, epoch + 1)
        writer.add_scalar('Loss_epoch/WGAN_D', avg_wgan_loss, epoch + 1)
        writer.add_scalar('Loss_epoch/Generator_Total', avg_loss_G, epoch + 1)
        writer.add_scalar('Loss_epoch/Generator_L1', avg_l1_loss, epoch + 1) # Логируем средний L1 за эпоху

        print(f"End of Epoch {epoch+1}/{config['num_epochs']} -> Avg Loss D: {avg_loss_D:.4f}, Avg WGAN D: {avg_wgan_loss:.4f}, Avg Loss G (Total): {avg_loss_G:.4f}, Avg L1 G: {avg_l1_loss:.4f}")

        # --- Оценка на валидационном наборе и логирование метрик после каждой эпохи ---
        if val_loader is not None:
            try:
                current_psnr, current_ssim, current_lpips = evaluate_on_validation_set(
                    generator, val_loader, DEVICE, psnr_val_metric, ssim_val_metric, lpips_val_metric
                )
                writer.add_scalar('Val_epoch/PSNR', current_psnr, epoch + 1)
                writer.add_scalar('Val_epoch/SSIM', current_ssim, epoch + 1)
                writer.add_scalar('Val_epoch/LPIPS', current_lpips, epoch + 1) # Lower is better for LPIPS
                print(f"Epoch {epoch+1} Val Metrics -> PSNR: {current_psnr:.4f}, SSIM: {current_ssim:.4f}, LPIPS: {current_lpips:.4f}")
                last_epoch_psnr, last_epoch_ssim, last_epoch_lpips = current_psnr, current_ssim, current_lpips
            except Exception as e:
                print(f"Error during validation metrics calculation for epoch {epoch+1}: {e}")
                # Оставляем NaN, если была ошибка
                last_epoch_psnr, last_epoch_ssim, last_epoch_lpips = float('NaN'), float('NaN'), float('NaN')
        else:
            print(f"Epoch {epoch+1}: Skipping validation metrics as val_loader is not available.")

        if (epoch + 1) % config.get('save_epoch_freq', 5) == 0 or (epoch + 1) == config['num_epochs']:
            save_checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            
            # Save .module.state_dict() if DataParallel to make checkpoints portable
            g_state_dict = generator.module.state_dict() if isinstance(generator, nn.DataParallel) else generator.state_dict()
            c_state_dict = critic.module.state_dict() if isinstance(critic, nn.DataParallel) else critic.state_dict()

            torch.save(g_state_dict, os.path.join(save_checkpoint_dir, f"generator_epoch_{epoch+1}.pth"))
            torch.save(c_state_dict, os.path.join(save_checkpoint_dir, f"critic_epoch_{epoch+1}.pth"))
            torch.save(optimizer_G.state_dict(), os.path.join(save_checkpoint_dir, f"optimizer_G_epoch_{epoch+1}.pth"))
            torch.save(optimizer_D.state_dict(), os.path.join(save_checkpoint_dir, f"optimizer_D_epoch_{epoch+1}.pth"))
            print(f"Saved checkpoint for epoch {epoch+1} in {save_checkpoint_dir} (Models saved as .module.state_dict if DataParallel)")

    print("Training finished.")

    # --- Логирование метрик для HParams ---
    # Собираем финальные метрики (потери за последнюю эпоху и метрики качества за последнюю эпоху)
    final_metrics = {
        'final_avg_loss_D': avg_loss_D if 'avg_loss_D' in locals() else float('NaN'),
        'final_avg_loss_G_total': avg_loss_G if 'avg_loss_G' in locals() else float('NaN'), # Обновлено имя
        'final_avg_loss_G_l1': avg_l1_loss if 'avg_l1_loss' in locals() else float('NaN'),    # Добавлен L1
        'final_avg_wgan_D': avg_wgan_loss if 'avg_wgan_loss' in locals() else float('NaN'),
        'completed_epochs': epoch + 1 if 'epoch' in locals() else 0,
        'final_psnr': last_epoch_psnr, 
        'final_ssim': last_epoch_ssim, 
        'final_lpips': last_epoch_lpips 
    }

    try:
        writer.add_hparams(hparam_dict_to_log, final_metrics)
    except Exception as e:
        print(f"Could not write HParams: {e}")
        print("HParam dict:", hparam_dict_to_log)
        print("Metric dict:", final_metrics)

    writer.close()

if __name__ == "__main__":
    torch.manual_seed(42)
    config = {
        'learning_rate': 0.0002,
        'beta1': 0.5,
        'beta2': 0.999,
        'batch_size': 128,
        'num_epochs': 20, 
        'lambda_gp': 6, 
        'lambda_l1': 100, # <-- Установлено значение 100 для lambda_l1
        'n_critic': 4,
        'root_dir': "/workspace-SR004.nfs2/nabiev/yolo_mamba_prjct/alakey/RGB_to_NIR/data/sen12ms_All_seasons", # !!! ИЗМЕНИТЕ НА ВАШ ПУТЬ !!!
        'device_preference': 'cuda',
        'image_size': 128, # Changed to 128
        'preload_data': False,
        'num_workers': 4,
        'pin_memory': False, # pin_memory=False for MPS generally
        'g_base_channels': 32, 'g_num_levels': 5,
        'd_base_channels': 32, 'd_num_levels': 4,
        
        'apply_weights_init': True,
        'log_dir': 'runs/rgb_to_nir_wgan_gp_v2_2', # New log_dir for 128px
        'checkpoint_dir': 'models/rgb_to_nir_wgan_gp_v2_2', # New checkpoint_dir for 128px
        'log_freq_batch': 100, 
        'log_image_freq_step': 500, 
        'log_weights_freq_step': 500, # New parameter to control frequency of weight/gradient logging
        'save_epoch_freq': 1,
        'load_checkpoint': False, # Set to True to load a checkpoint
        'checkpoint_load_epoch': 0, # Specify epoch to load if load_checkpoint is True
    } # CUDA_VISIBLE_DEVICES=0 python main_small_svg_p.py  
    print(config.get('log_dir'))
    print(config.get('checkpoint_dir'))

    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    train_gan(config)
