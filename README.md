# TFCNN — Reconocedor de Dígitos con CNN

Aplicación educativa interactiva que implementa una **Red Neuronal Convolucional (CNN)** para reconocer dígitos manuscritos del dataset **MNIST** (0-9). Construida con **PyTorch** y **Streamlit**, incluye visualizaciones didácticas del funcionamiento de las CNN y explicaciones sobre inteligencia artificial.

## Datos del equipo y la institución

- Integrantes: Ariel Colatto, Lucas Oviedo, Coral Tolazzi
- Cuatrimestre y Año: 1er Cuatrimestre del 2026
- Institución: Instituto Superior de Formación Técnica N°197 (Instituto Tecnológico Beltrán) 
- Área: Departamento de Ciencia y Tecnología
- Materia: Modelización de Sistemas de IA
- Docente: Alejandro Luis Bonavita

## Requisitos

- Python 3.9+
- PyTorch
- Streamlit
- Matplotlib
- NumPy
- Pillow
- streamlit-drawable-canvas
- torchvision
- tqdm

## Instalación

```bash
pip install -r requirements.txt
```

## Uso

### Entrenar el modelo

```bash
python train.py
```

Entrena la CNN durante 15 épocas con el dataset MNIST, guarda los pesos en `MulticlassCNN.pth` y genera el gráfico de curvas de entrenamiento en `assets/curvas_entrenamiento.png`.

### Ejecutar la aplicación web

```bash
streamlit run app.py
```

Abre una interfaz interactiva con cuatro secciones:

1. **¿Qué es la IA?** — Introducción conceptual con analogías y ejemplos cotidianos.
2. **¿Qué es una CNN?** — Explicación del funcionamiento de las redes convolucionales, incluyendo una animación interactiva que simula el deslizamiento de un filtro sobre una imagen y genera el mapa de características en tiempo real.
3. **¿Cómo se entrena?** — Simulación interactiva de entrenamiento con parámetros ajustables (épocas, tasa de aprendizaje, ruido).
4. **Prueba el modelo** — Lienzo para dibujar dígitos a mano alzada, predicción en vivo, mapas de activación de cada capa convolucional, explicación visual Grad-CAM y entrenamiento interactivo sobre un subconjunto de MNIST.

### Predecir desde terminal

```bash
python predict.py ruta/de/la/imagen.png
```

## Arquitectura de la CNN

| Capa | Detalle | Salida |
|---|---|---|
| Conv1 | 3×3, 1→16 canales, padding=1 | (16, 28, 28) |
| BatchNorm + ReLU + MaxPool 2×2 | | (16, 14, 14) |
| Conv2 | 3×3, 16→32 canales, padding=1 | (32, 14, 14) |
| BatchNorm + ReLU + MaxPool 2×2 | | (32, 7, 7) |
| Dropout2d (25%) | | |
| Conv3 | 3×3, 32→64 canales, padding=1 | (64, 7, 7) |
| BatchNorm + ReLU + Dropout2d (25%) | | |
| Flatten | 64 × 7 × 7 = 3136 | (3136,) |
| FC1 → ReLU → BatchNorm → Dropout (50%) | 3136 → 128 | (128,) |
| FC2 | 128 → 10 | (10,) |

## Tests

```bash
pytest tests/
```

Incluye tests unitarios para:
- Forma de salida del modelo (`test_model_output_shape`)
- Predicción por lotes sin NaN (`test_model_batch_output`)
- Hooks de Grad-CAM (`test_model_gradcam_hooks`)
- Generación de mapa de calor (`test_generate_gradcam_heatmap`)
- Superposición de heatmap (`test_overlay_heatmap`, `test_overlay_heatmap_from_tensor`)
- Visualización de heatmap (`test_plot_heatmap_only`)
- Rango de predicción (`test_prediction_range`)
- Probabilidades Softmax (`test_softmax_is_probability`)

## Estructura del proyecto

```
TFCNN/
├── app.py                       # Aplicación Streamlit
├── train.py                     # Entrenamiento del modelo
├── predict.py                   # Predicción desde terminal
├── src/
│   ├── model.py                 # Arquitectura CNN
│   ├── gradcam.py               # Generación de mapas Grad-CAM
│   └── __init__.py
├── tests/
│   ├── test_model.py
│   ├── test_gradcam.py
│   ├── test_predict.py
│   └── __init__.py
├── dataset/MNIST/raw/           # Dataset MNIST descargado
├── assets/                      # Imágenes y diagramas
├── MulticlassCNN.pth            # Pesos entrenados
└── requirements.txt
```
