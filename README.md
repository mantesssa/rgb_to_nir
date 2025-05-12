## Архитектура Модели

Модель основана на архитектуре Conditional WGAN-GP. Она состоит из двух основных компонентов: Генератора и Критика (Дискриминатора).

### Генератор (Generator)

Генератор построен на основе архитектуры U-Net. Он принимает на вход RGB-изображение (3 канала) и генерирует соответствующее NIR-изображение (1 канал). U-Net хорошо подходит для задач image-to-image трансляции благодаря наличию skip-соединений, которые позволяют передавать низкоуровневые признаки из энкодера напрямую в декодер, улучшая детализацию генерируемых изображений.

```mermaid
graph LR
    subgraph Generator_UNet  [Генератор (U-Net)]
        Input_RGB[RGB Image (H x W x 3)] --> G_Encoder{Encoder};
        G_Encoder --> G_Bottleneck{Bottleneck};
        G_Bottleneck --> G_Decoder{Decoder};
        G_Encoder -.-> G_Decoder; % Skip Connections
        G_Decoder --> Output_NIR[Generated NIR Image (H x W x 1)];
    end
```

### Критик / Дискриминатор (Critic / Discriminator)

Критик представляет собой сверточную нейронную сеть (PatchGAN-подобная архитектура). Он принимает на вход пару изображений: RGB-изображение (условие) и NIR-изображение (либо реальное из датасета, либо сгенерированное Генератором). Задача Критика — оценить, насколько "реалистичной" является представленная NIR-компонента в контексте данного RGB-изображения. Критик WGAN-GP не использует сигмоиду на выходе и обусловливается выдавать более высокие значения для реальных пар и более низкие для сгенерированных.

```mermaid
graph LR
    subgraph Critic_PatchGAN [Критик (PatchGAN-like)]
        C_Input_RGB[RGB Image (H x W x 3)] --> C_Concat{Concatenate};
        C_Input_NIR[Real/Fake NIR Image (H x W x 1)] --> C_Concat;
        C_Concat --> C_CNN{Convolutional Layers};
        C_CNN --> C_Output[Critic Score];
    end
```

### Функция Потерь WGAN-GP
(...остальное без изменений...)
