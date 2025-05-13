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
from torchvision import transforms # Added for resizing

# --- Класс датасета (с возможностью изменения размера) ---
class RGBNIRPairedDataset(Dataset):
    def __init__(self, root_dir, split='train', preload_to_ram=False, transform=None, image_size=None): # Added image_size
        super().__init__()
        self.root_dir = root_dir
        self.split = split
        self.rgb_dir = os.path.join(root_dir, f"{split}_A")
        self.nir_dir = os.path.join(root_dir, f"{split}_B")
        self.image_size = image_size
        self.resize_transform = None
        if self.image_size:
            # Use INTER_AREA for downsampling, INTER_LINEAR or INTER_CUBIC for upsampling
            interpolation = Image.Resampling.LANCZOS if image_size > 256 else Image.Resampling.BILINEAR # Example heuristic
            self.resize_transform = transforms.Resize((self.image_size, self.image_size), interpolation=interpolation)

        if not os.path.exists(self.rgb_dir):
            raise FileNotFoundError(f"RGB directory {self.rgb_dir} not found!")
        if not os.path.exists(self.nir_dir):
            raise FileNotFoundError(f"NIR directory {self.nir_dir} not found!")

        self.rgb_files = sorted([f for f in os.listdir(self.rgb_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]) # Added tif support
        self.nir_files = sorted([f for f in os.listdir(self.nir_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]) # Added tif support

        assert self.rgb_files == self.nir_files, \
            f"RGB and NIR files mismatch! RGB count: {len(self.rgb_files)}, NIR count: {len(self.nir_files)}. Check file names and counts."

        self.transform = transform # Note: original script had transform=None, so this is likely unused.
        self.preload = preload_to_ram
        self.data = []

        if self.preload:
            print(f"Preloading {split} data to RAM...")
            for idx in tqdm(range(len(self.rgb_files)), desc=f"Preloading {split}"):
                rgb_file = self.rgb_files[idx]
                rgb_path = os.path.join(self.rgb_dir, rgb_file)
                nir_path = os.path.join(self.nir_dir, rgb_file)
                try:
                    rgb_image = Image.open(rgb_path).convert('RGB')
                    nir_image = Image.open(nir_path).convert('L')

                    if self.resize_transform:
                        rgb_image = self.resize_transform(rgb_image)
                        nir_image = self.resize_transform(nir_image)

                    rgb_array = np.array(rgb_image, dtype=np.float32)
                    rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1) / 127.5 - 1.0

                    nir_array = np.array(nir_image, dtype=np.float32)
                    nir_tensor = torch.from_numpy(nir_array).unsqueeze(0) / 127.5 - 1.0

                    self.data.append((rgb_tensor, nir_tensor))
                except Exception as e:
                    print(f"Error loading image {rgb_file} during preload: {e}")
            print("Preloading complete.")

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        if self.preload and idx < len(self.data):
             # Check bounds if preloading failed for some items
             try:
                 return self.data[idx]
             except IndexError:
                 print(f"IndexError: Could not retrieve preloaded data for index {idx}. Falling back to dynamic loading.")
                 # Fallback to dynamic loading below

        # Dynamic loading or fallback from failed preload
        rgb_file = self.rgb_files[idx]
        rgb_path = os.path.join(self.rgb_dir, rgb_file)
        nir_path = os.path.join(self.nir_dir, rgb_file)
        try:
            rgb_image = Image.open(rgb_path).convert('RGB')
            nir_image = Image.open(nir_path).convert('L')

            if self.resize_transform:
                rgb_image = self.resize_transform(rgb_image)
                nir_image = self.resize_transform(nir_image)

            rgb_array = np.array(rgb_image, dtype=np.float32)
            # Normalize to [-1, 1]
            rgb_tensor = torch.from_numpy(rgb_array).permute(2, 0, 1) / 127.5 - 1.0

            nir_array = np.array(nir_image, dtype=np.float32)
            # Normalize to [-1, 1]
            nir_tensor = torch.from_numpy(nir_array).unsqueeze(0) / 127.5 - 1.0

        except FileNotFoundError as e:
            print(f"Error: File not found for index {idx}, RGB: {rgb_path}, NIR: {nir_path}")
            # Return dummy data or skip? Raising error stops training.
            # For now, re-raise. Consider alternative handling if needed.
            raise e
        except Exception as e:
            print(f"Error loading image {rgb_file} at index {idx}: {e}")
            # Return dummy data or skip? Raising error stops training.
            # For now, re-raise. Consider alternative handling if needed.
            raise e

        # if self.transform: # Original script used transform=None; if used, ensure synchronized transforms
        #     # rgb_tensor = self.transform(rgb_tensor)
        #     # nir_tensor = self.transform(nir_tensor)
        #     pass

        return rgb_tensor, nir_tensor

# --- Вспомогательные модули для U-Net (без изменений) ---
class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout > 0: # Ensure dropout is only added if value > 0
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
        if dropout > 0: # Ensure dropout is only added if value > 0
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        return x

# --- Уменьшенная Архитектура Генератора (U-Net Small) ---
class GeneratorUNetSmall(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, base_channels=32, num_levels=4):
        super().__init__()

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        # Encoder
        current_c = in_channels
        for i in range(num_levels):
            output_c = base_channels * (2**i)
            output_c = min(output_c, 512) # Cap max channels
            use_dropout = 0.5 if i >= (num_levels - 2) else 0.0 # Dropout deeper layers
            self.down_layers.append(UNetDown(current_c, output_c, normalize=(i!=0), dropout=use_dropout))
            current_c = output_c

        # Bottleneck
        self.bottleneck = UNetDown(current_c, current_c, normalize=False, dropout=0.5)
        current_c_after_bottleneck = current_c # Store bottleneck output channels

        # Decoder
        current_c = current_c_after_bottleneck # Start decoder input with bottleneck output channels
        for i in range(num_levels):
            level_idx = num_levels - 1 - i # Corresponding down layer index for skip connection

            # Calculate output channels for ConvTranspose layer in UNetUp
            output_c_convT = base_channels * (2**level_idx)
            # Ensure the last upsampling layer (before final_up) has correct number of channels if levels > 0
            if level_idx == 0 and num_levels > 0:
                output_c_convT = base_channels
            output_c_convT = min(output_c_convT, 512) # Cap max channels

            # Get the number of channels from the skip connection
            # Output channels of the corresponding down_layer model's first Conv2d layer
            skip_module_out_channels = self.down_layers[level_idx].model[0].out_channels

            # Apply dropout to earlier layers of decoder (closer to bottleneck)
            use_dropout = 0.5 if i < (num_levels // 2 +1) else 0.0

            self.up_layers.append(UNetUp(current_c, output_c_convT, dropout=use_dropout))
            # Update current channels for the next layer: channels after ConvTranspose + channels from skip connection
            current_c = output_c_convT + skip_module_out_channels

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(current_c, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Outputs in [-1, 1]
        )

    def forward(self, x):
        skip_connections = []
        for layer in self.down_layers:
            x = layer(x)
            skip_connections.append(x)

        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1] # Reverse for decoder

        for i, layer in enumerate(self.up_layers):
            # Make sure skip connection shape matches if necessary (e.g., due to padding)
            # Usually U-Net architectures are designed to match automatically
            x = layer(x, skip_connections[i])

        return self.final_up(x)


# --- Уменьшенная Архитектура Дискриминатора (Small) ---
class DiscriminatorSmall(nn.Module):
    def __init__(self, in_channels_rgb=3, in_channels_nir=1, base_channels=32, num_levels=3):
        super().__init__()
        input_channels = in_channels_rgb + in_channels_nir

        layers = []
        current_c = input_channels
        ndf = base_channels

        # Downsampling blocks (stride 2)
        for i in range(num_levels):
            output_c = ndf * (2**i)
            output_c = min(output_c, 512)
            layers.append(nn.Conv2d(current_c, output_c, kernel_size=4, stride=2, padding=1, bias=(i==0))) # No bias if norm, add bias to first layer
            if i != 0: # No norm in first layer typically
                layers.append(nn.InstanceNorm2d(output_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            current_c = output_c

        # One more layer with stride 1 (PatchGAN style)
        output_c_s1 = min(ndf * (2**num_levels), 512)
        layers.append(nn.Conv2d(current_c, output_c_s1, kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.InstanceNorm2d(output_c_s1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        current_c = output_c_s1

        # Final convolution to produce a single score value (no sigmoid for WGAN-GP)
        layers.append(nn.Conv2d(current_c, 1, kernel_size=4, stride=1, padding=1, bias=True))

        self.model = nn.Sequential(*layers)

    def forward(self, rgb_image, nir_image):
        x = torch.cat([rgb_image, nir_image], dim=1)
        return self.model(x)


# --- Функции потерь и Gradient Penalty (без изменений) ---
def compute_gradient_penalty(critic, real_nir, fake_nir, condition_rgb, device):
    batch_size = real_nir.size(0)
    # Epsilon for interpolation needs to match the batch dimension
    epsilon = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_nir)
    interpolated_nir = (epsilon * real_nir + (1 - epsilon) * fake_nir).requires_grad_(True)

    # Ensure condition_rgb matches the batch size of interpolated_nir if it was potentially modified
    current_batch_size = interpolated_nir.size(0)
    condition_rgb_matched = condition_rgb[:current_batch_size]

    interpolated_scores = critic(condition_rgb_matched, interpolated_nir)

    gradients = torch.autograd.grad(
        outputs=interpolated_scores,
        inputs=interpolated_nir,
        grad_outputs=torch.ones_like(interpolated_scores, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(current_batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    # Gradient penalty is calculated as the mean squared distance from norm 1
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

def critic_loss_fn(critic_real_out, critic_fake_out, gradient_penalty, lambda_gp):
    # WGAN loss: difference between expectation of critic scores for real and fake data
    wgan_loss = torch.mean(critic_fake_out) - torch.mean(critic_real_out)
    # Total critic loss: WGAN loss + gradient penalty
    loss_critic = wgan_loss + lambda_gp * gradient_penalty
    return loss_critic, wgan_loss # Return wgan_loss separately for logging if needed

def generator_loss_fn(critic_fake_out):
    # Generator aims to maximize the critic's score for fake images, equivalent to minimizing the negative score
    loss_gen = -torch.mean(critic_fake_out)
    return loss_gen

# --- Основная функция обучения ---
def train_gan(config):
    # Setup TensorBoard writer
    writer = SummaryWriter(log_dir=config['log_dir'])

    # Setup device
    if torch.cuda.is_available() and config['device_preference'] == 'cuda':
        DEVICE = "cuda"
    elif torch.backends.mps.is_available() and config['device_preference'] == 'mps':
        DEVICE = "mps"
        # torch.mps.empty_cache() # Might help sometimes on MPS
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    # --- Инициализация Моделей ---
    generator = GeneratorUNetSmall(
        in_channels=3,
        out_channels=1,
        base_channels=config['g_base_channels'],
        num_levels=config['g_num_levels']
    ).to(DEVICE)

    critic = DiscriminatorSmall(
        in_channels_rgb=3,
        in_channels_nir=1,
        base_channels=config['d_base_channels'],
        num_levels=config['d_num_levels']
    ).to(DEVICE)

    # --- (Опционально) Инициализация Весов ---
    # Apply weights initialization only if not loading a checkpoint
    if config.get('apply_weights_init', False) and not config.get('load_checkpoint', False):
        def weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1 :
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None: nn.init.constant_(m.bias.data, 0)
        generator.apply(weights_init)
        critic.apply(weights_init)
        print("Applied weights initialization.")

    # --- Инициализация Оптимизаторов ---
    optimizer_G = optim.Adam(generator.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))
    optimizer_D = optim.Adam(critic.parameters(), lr=config['learning_rate'], betas=(config['beta1'], config['beta2']))

    # --- Загрузка Чекпоинта (если включено) ---
    start_epoch = 0 # Default start epoch
    if config.get('load_checkpoint', False):
        load_epoch = config.get('checkpoint_load_epoch')
        if load_epoch is not None and load_epoch > 0:
            checkpoint_dir = config.get('checkpoint_dir', 'checkpoints') # Use the same dir as for saving
            gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{load_epoch}.pth")
            crit_path = os.path.join(checkpoint_dir, f"critic_epoch_{load_epoch}.pth")
            # Optional: Load optimizer states as well
            opt_gen_path = os.path.join(checkpoint_dir, f"optimizer_G_epoch_{load_epoch}.pth")
            opt_crit_path = os.path.join(checkpoint_dir, f"optimizer_D_epoch_{load_epoch}.pth")


            models_exist = os.path.exists(gen_path) and os.path.exists(crit_path)
            optimizers_exist = os.path.exists(opt_gen_path) and os.path.exists(opt_crit_path)

            if models_exist:
                try:
                    # map_location ensures compatibility across devices (e.g., load GPU model on CPU)
                    generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
                    critic.load_state_dict(torch.load(crit_path, map_location=DEVICE))
                    print(f"Successfully loaded Generator and Critic weights from epoch {load_epoch}")

                    # Load optimizers if files exist
                    if optimizers_exist:
                         optimizer_G.load_state_dict(torch.load(opt_gen_path, map_location=DEVICE))
                         optimizer_D.load_state_dict(torch.load(opt_crit_path, map_location=DEVICE))
                         print(f"Successfully loaded Optimizer states from epoch {load_epoch}")
                    else:
                        print(f"Optimizer state files not found for epoch {load_epoch}. Initializing optimizers from scratch.")

                    # Set the starting epoch for the training loop
                    start_epoch = load_epoch # Continue from the next epoch

                except Exception as e:
                    print(f"Error loading checkpoint from epoch {load_epoch}: {e}")
                    print("Starting training from scratch.")
                    start_epoch = 0 # Reset if loading failed
            else:
                print(f"Checkpoint model files for epoch {load_epoch} not found in {checkpoint_dir}. Starting training from scratch.")
                start_epoch = 0
        else:
            print("Warning: 'load_checkpoint' is True, but 'checkpoint_load_epoch' is not specified or is 0. Starting training from scratch.")
            start_epoch = 0

    # --- Инициализация Датасета и DataLoader ---
    train_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'],
        split='train',
        preload_to_ram=config.get('preload_data', False),
        image_size=config.get('image_size', None) # Pass image_size
    )

    # Adjust num_workers based on OS and config
    num_workers = config.get('num_workers', 0)
    if os.name == 'nt' and num_workers > 0 :
        print(f"Warning: num_workers > 0 ({num_workers}) on Windows can cause issues. Setting to 0.")
        num_workers = 0

    # Determine pin_memory based on device
    pin_memory_flag = config.get('pin_memory', True) if DEVICE == 'cuda' else False

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory_flag,
        drop_last=True # Drop last batch if it's smaller than batch_size, avoids issues with batch norm or GP
    )
    print(f"Dataset size: {len(train_dataset)}, Image size: {config.get('image_size', 'original')}")
    print(f"Dataloader num_workers: {num_workers}, pin_memory: {pin_memory_flag}, drop_last: {train_loader.drop_last}")


    # --- Добавление графов в TensorBoard ---
    # Needs a sample batch; run this after DataLoader is initialized
    # Also, ensure this runs only once, ideally before the loop
    if start_epoch == 0: # Only add graph if starting fresh
        try:
            sample_rgb, sample_nir = next(iter(train_loader))
            # Ensure sample batch is on the correct device
            sample_rgb_dev = sample_rgb.to(DEVICE)
            sample_nir_dev = sample_nir.to(DEVICE)

            # Add generator graph
            writer.add_graph(generator, sample_rgb_dev)
            # Add critic graph - Pass inputs as a tuple for multiple inputs
            writer.add_graph(critic, (sample_rgb_dev, sample_nir_dev))
            print("Model graphs added to TensorBoard.")
        except Exception as e:
            print(f"Could not add model graph to TensorBoard: {e}")


    # --- Цикл Обучения ---
    generator.train()
    critic.train()
    # Adjust global step based on loaded epoch
    global_step = start_epoch * len(train_loader) # Approximation, assumes constant loader length

    print(f"Starting training from Epoch {start_epoch + 1}")
    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}", leave=True)
        epoch_loss_D_avg = 0.0
        epoch_loss_G_avg = 0.0
        epoch_wgan_loss_avg = 0.0 # For logging raw WGAN loss

        for batch_idx, (real_rgb, real_nir) in enumerate(progress_bar):
            real_rgb = real_rgb.to(DEVICE)
            real_nir = real_nir.to(DEVICE)
            current_batch_size = real_rgb.size(0) # Get actual batch size

            # --- 1. Обучение Критика (Discriminator) ---
            # Train Critic more often than Generator (n_critic iterations)
            for _ in range(config['n_critic']):
                optimizer_D.zero_grad()
                with torch.no_grad(): # No need to track gradients for generator here
                    fake_nir = generator(real_rgb).detach() # Detach to avoid backprop into generator

                critic_real_out = critic(real_rgb, real_nir)
                critic_fake_out = critic(real_rgb, fake_nir) # Use detached fake_nir

                gradient_penalty = compute_gradient_penalty(critic, real_nir, fake_nir, real_rgb, DEVICE)
                loss_D, wgan_loss = critic_loss_fn(critic_real_out, critic_fake_out, gradient_penalty, config['lambda_gp'])
                loss_D.backward()
                optimizer_D.step()

            # --- 2. Обучение Генератора ---
            # Train Generator less frequently
            optimizer_G.zero_grad()
            # Generate new fake images, this time tracking gradients through generator
            fake_nir_for_gen = generator(real_rgb)
            # Get critic's score for these new fake images
            critic_fake_out_for_gen = critic(real_rgb, fake_nir_for_gen)
            # Calculate generator loss based on critic's score
            loss_G = generator_loss_fn(critic_fake_out_for_gen)
            loss_G.backward()
            optimizer_G.step()

            # --- Логирование ---
            writer.add_scalar('Loss/Critic_step', loss_D.item(), global_step)
            writer.add_scalar('Loss/WGAN_D_step', wgan_loss.item(), global_step) # Log raw WGAN component
            writer.add_scalar('Loss/Gradient_Penalty_step', gradient_penalty.item(), global_step)
            writer.add_scalar('Loss/Generator_step', loss_G.item(), global_step)

            epoch_loss_D_avg += loss_D.item()
            epoch_loss_G_avg += loss_G.item()
            epoch_wgan_loss_avg += wgan_loss.item()

            # Update progress bar less frequently for performance
            if batch_idx % config.get('log_freq_batch', 50) == 0:
                 progress_bar.set_postfix({
                    'Loss D': f'{loss_D.item():.4f}',
                    'Loss G': f'{loss_G.item():.4f}',
                    'GP': f'{gradient_penalty.item():.4f}'
                })

            # Log images periodically
            if global_step % config.get('log_image_freq_step', 200) == 0:
                with torch.no_grad():
                    num_images_to_log = min(4, current_batch_size) # Log fewer images
                    # Ensure normalization is correct for visualization (-1 to 1 input -> 0 to 1 output for make_grid)
                    img_grid_real_rgb = torchvision.utils.make_grid(real_rgb[:num_images_to_log], normalize=True, value_range=(-1, 1))
                    img_grid_real_nir = torchvision.utils.make_grid(real_nir[:num_images_to_log], normalize=True, value_range=(-1, 1))
                    img_grid_fake_nir = torchvision.utils.make_grid(fake_nir_for_gen[:num_images_to_log], normalize=True, value_range=(-1, 1))

                    writer.add_image('Input/Real_RGB', img_grid_real_rgb, global_step)
                    writer.add_image('Target/Real_NIR', img_grid_real_nir, global_step)
                    writer.add_image('Generated/Fake_NIR', img_grid_fake_nir, global_step)

            global_step += 1

        # --- Логирование в конце эпохи ---
        avg_loss_D = epoch_loss_D_avg / len(train_loader)
        avg_loss_G = epoch_loss_G_avg / len(train_loader)
        avg_wgan_loss = epoch_wgan_loss_avg / len(train_loader)
        writer.add_scalar('Loss_epoch/Critic', avg_loss_D, epoch + 1)
        writer.add_scalar('Loss_epoch/WGAN_D', avg_wgan_loss, epoch + 1)
        writer.add_scalar('Loss_epoch/Generator', avg_loss_G, epoch + 1)
        print(f"End of Epoch {epoch+1}/{config['num_epochs']} -> Avg Loss D: {avg_loss_D:.4f}, Avg WGAN D: {avg_wgan_loss:.4f}, Avg Loss G: {avg_loss_G:.4f}")

        # --- Сохранение чекпоинта ---
        if (epoch + 1) % config.get('save_epoch_freq', 5) == 0 or (epoch + 1) == config['num_epochs']:
            save_checkpoint_dir = config.get('checkpoint_dir', 'checkpoints')
            os.makedirs(save_checkpoint_dir, exist_ok=True)
            gen_save_path = os.path.join(save_checkpoint_dir, f"generator_epoch_{epoch+1}.pth")
            crit_save_path = os.path.join(save_checkpoint_dir, f"critic_epoch_{epoch+1}.pth")
            opt_gen_save_path = os.path.join(save_checkpoint_dir, f"optimizer_G_epoch_{epoch+1}.pth")
            opt_crit_save_path = os.path.join(save_checkpoint_dir, f"optimizer_D_epoch_{epoch+1}.pth")

            torch.save(generator.state_dict(), gen_save_path)
            torch.save(critic.state_dict(), crit_save_path)
            # Save optimizer states
            torch.save(optimizer_G.state_dict(), opt_gen_save_path)
            torch.save(optimizer_D.state_dict(), opt_crit_save_path)

            print(f"Saved checkpoint for epoch {epoch+1} in {save_checkpoint_dir}")

    print("Training finished.")
    writer.close()

if __name__ == "__main__":
    config = {
        # --- Основные параметры обучения ---
        'learning_rate': 0.0002,
        'beta1': 0.5,           # Adam beta1
        'beta2': 0.999,         # Adam beta2
        'batch_size': 16,       # Уменьшите, если не хватает VRAM
        'num_epochs': 20,       # Общее количество эпох для обучения
        'lambda_gp': 10,        # Вес градиентного штрафа
        'n_critic': 5,          # Количество обновлений критика на одно обновление генератора

        # --- Параметры датасета и устройства ---
        'root_dir': "/Users/mantesssa/Downloads/archive/sen12ms_All_seasons", # !!! ИЗМЕНИТЕ НА ВАШ ПУТЬ !!!
        'device_preference': 'mps', # 'mps', 'cuda', or 'cpu'
        'image_size': 64,       # Размер изображений (e.g., 64, 128). None для оригинального.
        'preload_data': False,  # Загружать ли данные в RAM (быстрее, но требует много RAM)
        'num_workers': 4,       # Количество процессов для загрузки данных (0 для Windows/MPS обычно)
        'pin_memory': True,    # Обычно True для CUDA, False для MPS/CPU

        # --- Параметры архитектуры моделей ---
        'g_base_channels': 32,  # Базовое количество каналов генератора
        'g_num_levels': 4,      # Количество уровней down/up в генераторе
        'd_base_channels': 32,  # Базовое количество каналов дискриминатора
        'd_num_levels': 3,      # Количество уровней с stride=2 в дискриминаторе
        'apply_weights_init': False, # Применять ли инициализацию весов (Xavier/Normal)

        # --- Параметры логирования и сохранения ---
        'log_dir': 'runs/rgb_to_nir_wgan_gp_small_v32_resume_test', # Папка для логов TensorBoard
        'checkpoint_dir': 'models/rgb_to_nir_wgan_gp_small_v32', # Папка для сохранения/загрузки чекпоинтов
        'log_freq_batch': 100,     # Как часто выводить лог потерь в консоль (каждые N батчей)
        'log_image_freq_step': 500, # Как часто логировать изображения в TensorBoard (каждые N шагов)
        'save_epoch_freq': 1,     # Как часто сохранять чекпоинт (каждые N эпох)

        # --- Параметры для загрузки чекпоинта ---
        'load_checkpoint': True,          # Установить в True для загрузки чекпоинта
        'checkpoint_load_epoch': 1,       # Номер эпохи, с которой загружать (например, 2). Обучение начнется с эпохи 3.
    }

    # Create directories if they don't exist
    os.makedirs(config['log_dir'], exist_ok=True)
    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    train_gan(config)