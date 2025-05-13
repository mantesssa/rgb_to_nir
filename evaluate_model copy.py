# RGB_to_NIR/evaluate_model.py
import os
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont # Added ImageDraw, ImageFont for labels
import torchvision.transforms.functional as TF
from torchvision import transforms
from torch.utils.data import DataLoader 
import matplotlib.pyplot as plt # Still useful for quick display if needed
from torch import nn 
from tqdm import tqdm

# --- Импортируем классы из скрипта обучения ---
try:
    from main_small_from_chpnt import RGBNIRPairedDataset, GeneratorUNetSmall
    print("Successfully imported classes from main_small_from_chpnt.py")
except ImportError as e:
    print(f"Error importing from main_small_from_chpnt.py: {e}")
    print("Please ensure main_small_from_chpnt.py is in the same directory or accessible in PYTHONPATH.")
    raise

# --- Evaluation Configuration ---
eval_config = {
    'root_dir': "/Users/mantesssa/Downloads/archive/sen12ms_All_seasons", 
    'eval_split': 'val', 
    'device_preference': 'cpu', 
    'image_size': 64,      

    'g_base_channels': 32,
    'g_num_levels': 4,

    'checkpoint_dir_load': '/Users/mantesssa/Documents/skoltech_study/Gen AI/HW1/RGB_to_NIR/kaggle1_results/models/rgb_to_nir_wgan_gp_small_v1',
    'checkpoint_load_epoch': 10, 

    'output_dir': 'evaluation_results/rgb_to_nir_small_v1_epoch10_imported_fix', 
    'num_images_to_evaluate': 10, # Количество изображений для включения в большую картинку
    'display_images': False, 
    'save_individual_images': True, # Сохранять ли отдельные изображения как раньше
    'save_composite_image': True, # Сохранять ли большую сводную картинку
    'composite_image_name': 'evaluation_summary.png',
}

def evaluate_generator(config):
    if torch.cuda.is_available() and config['device_preference'] == 'cuda':
        DEVICE = "cuda"
    elif torch.backends.mps.is_available() and config['device_preference'] == 'mps':
        DEVICE = "mps"
    else:
        DEVICE = "cpu"
    print(f"Using device: {DEVICE}")

    generator = GeneratorUNetSmall(
        in_channels=3, out_channels=1,
        base_channels=config['g_base_channels'], num_levels=config['g_num_levels']
    ).to(DEVICE)

    load_epoch = config.get('checkpoint_load_epoch')
    checkpoint_dir = config.get('checkpoint_dir_load')
    if load_epoch is not None and load_epoch > 0 and checkpoint_dir:
        gen_path = os.path.join(checkpoint_dir, f"generator_epoch_{load_epoch}.pth")
        if os.path.exists(gen_path):
            try:
                generator.load_state_dict(torch.load(gen_path, map_location=DEVICE))
                print(f"Successfully loaded Generator weights from epoch {load_epoch} from {gen_path}")
            except Exception as e: print(f"Error loading G checkpoint: {e}"); return
        else: print(f"G checkpoint not found: {gen_path}"); return
    else: print("G checkpoint epoch/dir not specified"); return

    eval_dataset = RGBNIRPairedDataset(
        root_dir=config['root_dir'], split=config['eval_split'],
        image_size=config.get('image_size', None)
    )
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, num_workers=0)
    print(f"Evaluating on dataset: {config['eval_split']}, size: {len(eval_dataset)}")

    output_dir = config['output_dir']
    if config.get('save_individual_images', False) or config.get('save_composite_image', False):
        os.makedirs(output_dir, exist_ok=True)
        print(f"Saving results to: {output_dir}")

    generator.eval()

    # Списки для хранения PIL изображений для сводной картинки
    all_input_pils = []
    all_fake_nir_pils = []
    all_real_nir_pils = []
    processed_filenames = []


    with torch.no_grad():
        for i, data_batch in enumerate(tqdm(eval_loader, desc="Evaluating model")):
            if config['num_images_to_evaluate'] is not None and i >= config['num_images_to_evaluate']:
                break
            
            try:
                real_rgb_tensor, real_nir_tensor = data_batch
                # Генерируем имя файла на основе индекса, если оно не приходит из датасета
                # Если ваш RGBNIRPairedDataset все же возвращает имя файла третьим элементом, используйте его:
                # real_rgb_tensor, real_nir_tensor, file_name_base = data_batch
                # if isinstance(file_name_base, (list,tuple)): file_name_base = file_name_base[0]
                file_name_base = f"eval_image_{i:04d}"

            except ValueError as e:
                print(f"ValueError unpacking batch {i}: {e}. Data: {data_batch}. Skipping.")
                continue

            if real_rgb_tensor is None:
                print(f"Skipping item {i} (RGB tensor is None)")
                continue
            
            real_rgb_tensor = real_rgb_tensor.to(DEVICE)
            if real_nir_tensor is not None and real_nir_tensor.nelement() > 0:
                 real_nir_tensor = real_nir_tensor.to(DEVICE)

            fake_nir_tensor = generator(real_rgb_tensor)
            
            input_rgb_normalized = (real_rgb_tensor.squeeze(0).cpu() + 1.0) / 2.0 
            input_rgb_pil = TF.to_pil_image(input_rgb_normalized.clamp(0,1), mode='RGB')

            fake_nir_normalized = (fake_nir_tensor.squeeze(0).cpu() + 1.0) / 2.0 
            fake_nir_pil = TF.to_pil_image(fake_nir_normalized.clamp(0,1), mode='L') 

            real_nir_pil = None
            has_real_nir = False
            if real_nir_tensor is not None and real_nir_tensor.nelement() > 0 : 
                real_nir_normalized = (real_nir_tensor.squeeze(0).cpu() + 1.0) / 2.0
                real_nir_pil = TF.to_pil_image(real_nir_normalized.clamp(0,1), mode='L')
                has_real_nir = True

            # Сохраняем PIL изображения для большой картинки
            all_input_pils.append(input_rgb_pil)
            all_fake_nir_pils.append(fake_nir_pil)
            all_real_nir_pils.append(real_nir_pil) # Будет None, если нет реального NIR
            processed_filenames.append(file_name_base)


            if config.get('save_individual_images', False):
                input_rgb_pil.save(os.path.join(output_dir, f"{file_name_base}_input_RGB.png"))
                fake_nir_pil.save(os.path.join(output_dir, f"{file_name_base}_generated_NIR.png"))
                if real_nir_pil is not None:
                    real_nir_pil.save(os.path.join(output_dir, f"{file_name_base}_real_NIR.png"))
                
            if config.get('display_images', False) and i < 5 : # Показывать только первые несколько
                num_cols_display = 3 if has_real_nir else 2
                fig, axs = plt.subplots(1, num_cols_display, figsize=(num_cols_display * 4, 4))
                fig.suptitle(f"{file_name_base}")
                axs[0].imshow(input_rgb_pil); axs[0].set_title("Input RGB"); axs[0].axis('off')
                axs[1].imshow(fake_nir_pil, cmap='gray'); axs[1].set_title("Generated NIR"); axs[1].axis('off')
                if has_real_nir:
                    axs[2].imshow(real_nir_pil, cmap='gray'); axs[2].set_title("Real NIR"); axs[2].axis('off')
                plt.tight_layout(); plt.show()

    # --- Создание и сохранение большой сводной картинки ---
    if config.get('save_composite_image', False) and all_input_pils:
        num_images = len(all_input_pils)
        img_width, img_height = all_input_pils[0].size

        # Определяем, есть ли хотя бы одно реальное NIR изображение для третьей строки
        include_real_nir_row = any(p is not None for p in all_real_nir_pils)
        num_rows_composite = 3 if include_real_nir_row else 2
        
        padding = 5  # Отступ между картинками
        title_height = 30 # Высота для подписей строк
        
        total_width = num_images * (img_width + padding) - padding 
        total_height = num_rows_composite * (img_height + padding) - padding + title_height

        composite_image = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(composite_image)
        try:
            # Попытка загрузить шрифт, если не найден - используется стандартный
            font = ImageFont.truetype("arial.ttf", 15)
        except IOError:
            font = ImageFont.load_default()

        # Подписи строк
        row_labels = ["Input RGB", "Generated NIR", "Real NIR (Ground Truth)"]
        
        current_y = 0
        # Рисуем подпись для первой строки
        if num_images > 0 : draw.text((padding, current_y + padding), row_labels[0], fill="black", font=font)
        current_y += title_height // num_rows_composite # Распределяем высоту подписи на строки

        for i in range(num_images):
            x_offset = i * (img_width + padding)
            # Input RGB
            composite_image.paste(all_input_pils[i], (x_offset, current_y))
        current_y += img_height + padding

        if num_images > 0 : draw.text((padding, current_y - title_height // num_rows_composite + padding ), row_labels[1], fill="black", font=font)

        for i in range(num_images):
            x_offset = i * (img_width + padding)
            # Generated NIR (конвертируем в RGB для вставки, если она 'L')
            composite_image.paste(all_fake_nir_pils[i].convert('RGB'), (x_offset, current_y))
        current_y += img_height + padding

        if include_real_nir_row:
            if num_images > 0 : draw.text((padding, current_y - title_height // num_rows_composite + padding), row_labels[2], fill="black", font=font)
            for i in range(num_images):
                x_offset = i * (img_width + padding)
                if all_real_nir_pils[i] is not None:
                    # Real NIR (конвертируем в RGB для вставки, если она 'L')
                    composite_image.paste(all_real_nir_pils[i].convert('RGB'), (x_offset, current_y))
                else:
                    # Если реального NIR нет, рисуем серый квадрат или оставляем белым
                    placeholder = Image.new('RGB', (img_width, img_height), color='lightgray')
                    composite_image.paste(placeholder, (x_offset, current_y))
        
        composite_path = os.path.join(output_dir, config['composite_image_name'])
        composite_image.save(composite_path)
        print(f"Saved composite evaluation image to: {composite_path}")

    print("Evaluation finished.")

if __name__ == "__main__":
    evaluate_generator(eval_config)