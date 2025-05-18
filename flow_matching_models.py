import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tqdm import tqdm

# Попытка импортировать odeint из torchdiffeq. Если его нет, будет ошибка при использовании.
# Пользователю нужно будет установить: pip install torchdiffeq
try:
    from torchdiffeq import odeint
except ImportError:
    print("Warning: torchdiffeq is not installed. ODE solving will not be available.")
    print("Please install it with: pip install torchdiffeq")
    odeint = None

# --- Вспомогательные модули для U-Net (аналогично diffusion_models.py) ---
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
        if x.shape[2:] != skip_input.shape[2:]:
            skip_input = F.interpolate(skip_input, size=x.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat((x, skip_input), 1)
        return x

class OriginalSinusoidalPositionalEmbedding(nn.Module): # Renamed from SinusoidalPositionalEmbedding
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # Expects integer time
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1: # zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1))
        return embeddings

class ContinuousSinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000.0):
        super().__init__()
        self.dim = dim
        self.max_period = max_period # Added max_period as a parameter

    def forward(self, time_float): # Expects float time, e.g., in [0, 1]
        if time_float.ndim == 1:
            time_float = time_float.unsqueeze(-1) # Ensure it's [B, 1]
        
        device = time_float.device
        half_dim = self.dim // 2
        
        # Frequencies for the sinusoidal embedding
        # This is a common way to generate frequencies for continuous inputs
        div_term = torch.exp(torch.arange(0, half_dim, device=device).float() * -(math.log(self.max_period) / (half_dim -1 if half_dim > 1 else 1.0 )))
        
        embeddings = time_float * div_term # [B, 1] * [half_dim] -> [B, half_dim]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) # [B, dim]
        
        if self.dim % 2 == 1: # zero pad if dim is odd
            embeddings = F.pad(embeddings, (0,1)) # Corrected padding for the last dimension
        return embeddings

# --- U-Net для предсказания Векторного Поля v(x_t, t, c) ---
# Принимает: x_t (NIR на пути от шума к данным), время t, условие RGB
# Выдает: векторное поле v_theta (той же размерности, что и x_t)
class VectorFieldUNet(nn.Module):
    def __init__(self, nir_channels=1, rgb_channels=3, out_channels_vector_field=1, # out_channels_vector_field обычно = nir_channels
                 base_channels=64, num_levels=4, time_emb_dim=256, continuous_time_emb_max_period=1000.0): # Added continuous_time_emb_max_period
        super().__init__()
        
        self.time_mlp = nn.Sequential(
            ContinuousSinusoidalPositionalEmbedding(time_emb_dim, max_period=continuous_time_emb_max_period), # Using the new embedding
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Начальный сверточный слой для объединения x_t и условия RGB
        self.initial_conv = nn.Conv2d(nir_channels + rgb_channels, base_channels, kernel_size=3, padding=1)

        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        current_c = base_channels
        # Down-sampling path
        for i in range(num_levels):
            output_c = min(base_channels * (2**i), 512)
            down_block = nn.ModuleDict({
                'unet_down': UNetDown(current_c, output_c, normalize=(i!=0)),
                'time_proj': nn.Linear(time_emb_dim, current_c) 
            })
            self.down_layers.append(down_block)
            current_c = output_c
        
        self.bottleneck_time_proj = nn.Linear(time_emb_dim, current_c)
        self.bottleneck = UNetDown(current_c, current_c, normalize=False)
        
        # Up-sampling path (логика как в NoisePredictionUNet)
        # ... (включая final_up_conv для сохранения размера)
        # Важно, чтобы выходной размер соответствовал входному x_t
        for i in range(num_levels):
            level_idx = num_levels - 1 - i
            skip_channels_from_down = self.down_layers[level_idx]['unet_down'].model[0].out_channels
            output_c_convT = min(base_channels * (2**level_idx), 512)
            if level_idx == 0 and num_levels > 0: output_c_convT = base_channels
            
            unet_up_in_channels = current_c
            unet_up_out_channels = output_c_convT

            up_block = nn.ModuleDict({
                'unet_up': UNetUp(unet_up_in_channels, unet_up_out_channels),
                'time_proj': nn.Linear(time_emb_dim, unet_up_in_channels)
            })
            self.up_layers.append(up_block)
            current_c = unet_up_out_channels + skip_channels_from_down

        # Финальный слой для предсказания векторного поля. 
        # Должен сохранять пространственные размеры, если U-Net симметрична.
        # Если U-Net уменьшает размер (как в предыдущей версии diffusion_models до фикса), 
        # то здесь нужен ConvTranspose2d.
        # Предполагаем, что UNetDown/UNetUp сбалансированы, и bottleneck не уменьшает разрешение необратимо.
        # Однако, если unet_num_levels = 4, то 128 -> 64 -> 32 -> 16 -> 8 (down) -> bottleneck (8x8)
        # Затем up: 8 -> 16 -> 32 -> 64 -> 128. 
        # Значит, если final_up_conv не используется, current_c соответствует разрешению 128x128.
        self.final_up_conv = nn.ConvTranspose2d(current_c, out_channels_vector_field, kernel_size=4, stride=2, padding=1)
        # Если бы U-Net была несимметричной (например, bottleneck уменьшал размер больше, чем восстанавливается)
        # или если бы мы хотели на один уровень меньше в up-пути, тогда бы понадобился final_up_conv.
        # Для flow matching U-Net обычно симметрична по разрешению.

    def forward(self, x_t, time, condition_rgb):
        # 1. Временной эмбеддинг
        # Время t должно быть в [0, 1] (или T_end). ContinuousSinusoidalPositionalEmbedding ожидает float.
        t_emb = self.time_mlp(time) # [B, time_emb_dim]

        # 2. Объединение x_t и RGB, начальная свертка
        # x_t здесь - это NIR изображение на пути от шума к данным
        x = torch.cat((x_t, condition_rgb), dim=1)
        x = self.initial_conv(x) 
        
        skip_connections = []

        # 3. Down-sampling
        for block in self.down_layers:
            time_projection = block['time_proj'](t_emb)[:, :, None, None]
            x = x + time_projection 
            x = block['unet_down'](x)
            skip_connections.append(x)

        # 4. Bottleneck
        time_projection_bottleneck = self.bottleneck_time_proj(t_emb)[:, :, None, None]
        x = x + time_projection_bottleneck
        x = self.bottleneck(x)

        skip_connections = skip_connections[::-1]

        # 5. Up-sampling
        for i, block in enumerate(self.up_layers):
            time_projection = block['time_proj'](t_emb)[:, :, None, None]
            x = x + time_projection
            x = block['unet_up'](x, skip_connections[i])
            
        # 6. Финальная свертка для получения векторного поля
        vector_field = self.final_up_conv(x)
        return vector_field

# --- Функция для генерации с помощью ОДУ-решателя ---
# Эта функция будет оберткой для вызова ОДУ-решателя (например, из torchdiffeq или flow_matching.solver)

class ODEFunc(nn.Module):
    """Вспомогательный класс для передачи в odeint."""
    def __init__(self, model, condition_rgb):
        super().__init__()
        self.model = model
        self.condition_rgb = condition_rgb

    def forward(self, t, x):
        # Ожидается, что x - это текущее состояние (например, NIR изображение)
        # t - текущее время (скаляр)
        # Модель должна принимать батчированные t, поэтому расширяем t
        if x.ndim == 3: # Если одиночное изображение без батча (маловероятно здесь)
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        t_batched = torch.full((batch_size,), t.item(), device=x.device, dtype=torch.float32) 
        # Убедимся, что condition_rgb соответствует батчу x
        # Это может потребовать более сложной логики, если condition_rgb не батчирован изначально
        # или если solver обрабатывает батчи по-своему.
        # Для простоты предположим, что condition_rgb уже готов.
        
        # Модель VectorFieldUNet ожидает (x_t, time, condition_rgb)
        # где time - это тензор [B] float. t здесь - скаляр от ОДУ решателя.
        return self.model(x, t_batched, self.condition_rgb)

@torch.no_grad()
def sample_with_ode_solver(model, initial_noise_x0, condition_rgb, t_span, num_eval_points, device, solver_method='dopri5'):
    """
    Генерирует изображения, решая ОДУ dx/dt = v(x, t, c).
    model: Обученная VectorFieldUNet.
    initial_noise_x0: Начальный шум (например, из N(0,1)), форма [B, C, H, W].
    condition_rgb: Условное RGB изображение, форма [B, 3, H, W].
    t_span: Временной интервал для интегрирования (например, torch.tensor([0.0, 1.0])).
    num_eval_points: Количество точек, в которых сохраняется решение (включая начальную и конечную).
    device: Устройство для вычислений.
    solver_method: Метод ОДУ-решателя (для torchdiffeq.odeint).
    """
    if odeint is None:
        raise RuntimeError("torchdiffeq.odeint is not available. Please install torchdiffeq.")

    model.eval()
    
    # Убедимся, что все на нужном устройстве
    initial_noise_x0 = initial_noise_x0.to(device)
    condition_rgb = condition_rgb.to(device)
    t_eval = torch.linspace(t_span[0], t_span[1], num_eval_points, device=device)

    # Создаем экземпляр функции для ОДУ
    # Важно: condition_rgb передается и сохраняется в ode_func для использования на каждом шаге
    ode_func = ODEFunc(model, condition_rgb).to(device)
    
    # Решаем ОДУ
    # odeint(func, y0, t, ...)
    # y0 - начальное условие (initial_noise_x0)
    # t - тензор временных точек, в которых нужно вычислить решение (t_eval)
    solution_trajectory = odeint(
        ode_func, 
        initial_noise_x0, 
        t_eval, 
        method=solver_method,
        atol=1e-5, # Абсолютная и относительная точности, можно настроить
        rtol=1e-5 
    )
    
    # Нас интересует конечное изображение x1, которое находится в solution_trajectory[-1]
    # solution_trajectory имеет форму [num_eval_points, B, C, H, W]
    generated_image_x1 = solution_trajectory[-1]
    
    model.train()
    return generated_image_x1


if __name__ == '__main__':
    # --- Пример использования (для базовой отладки) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Параметры
    batch_size = 2
    image_size = 64 
    nir_channels = 1
    rgb_channels = 3
    T_final_time = 1.0 # Конечное время для Flow Matching (обычно 1.0)

    # Модель
    fm_unet_model = VectorFieldUNet(
        nir_channels=nir_channels, 
        rgb_channels=rgb_channels, 
        out_channels_vector_field=nir_channels,
        base_channels=32, 
        num_levels=3,     
        time_emb_dim=128,
        continuous_time_emb_max_period=1000.0 # Added for the new embedding
    ).to(device)

    # Входные данные (фейковые)
    dummy_x_t_nir = torch.randn(batch_size, nir_channels, image_size, image_size, device=device)
    dummy_condition_rgb = torch.randn(batch_size, rgb_channels, image_size, image_size, device=device)
    # Для Flow Matching время t обычно в [0, 1]
    dummy_t_float = torch.rand(batch_size, device=device) # t в [0,1]
    # dummy_t_int_for_embedding = (dummy_t_float * 1000).long() # No longer needed for embedding

    # Тест прямого прохода U-Net
    predicted_vector_field = fm_unet_model(dummy_x_t_nir, dummy_t_float, dummy_condition_rgb) # Pass float time directly
    print(f"VectorFieldUNet output shape (predicted_vector_field): {predicted_vector_field.shape}")

    # Тест генерации (если torchdiffeq установлен)
    if odeint is not None:
        print("Testing ODE sampling...")
        initial_noise = torch.randn(batch_size, nir_channels, image_size, image_size, device=device)
        time_span = torch.tensor([0.0, T_final_time], device=device)
        num_sampling_steps = 20 # Количество шагов для решателя ОДУ (не путать с T в диффузии)
        
        generated_images = sample_with_ode_solver(
            fm_unet_model, 
            initial_noise, 
            dummy_condition_rgb, 
            time_span, 
            num_sampling_steps, 
            device
        )
        print(f"sample_with_ode_solver output shape (generated_images): {generated_images.shape}")
    else:
        print("Skipping ODE sampling test as torchdiffeq is not installed.")

    print("Flow Matching model skeleton components created.")
    # Потребуется from tqdm import tqdm, если будет использоваться где-то в p_sample_loop или аналогах. 