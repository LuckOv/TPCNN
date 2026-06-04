import streamlit as st
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gradcam
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from modelo_cnn import CNN

# Configuración de la página
st.set_page_config(page_title="Reconocedor de Dígitos con CNN", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ CARGA DEL MODELO (cache) ------------------
@st.cache_resource
def load_model():
    model = CNN(in_channels=1, num_classes=10)
    model.load_state_dict(torch.load("MulticlassCNN.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------ MENÚ LATERAL ------------------
st.sidebar.title("📚 Navegación")
seccion = st.sidebar.radio(
    "Ir a:",
    ["🤖 ¿Qué es la IA?",
     "🧠 ¿Qué es una CNN?",
     "📊 ¿Cómo se entrena?",
     "✍️ Prueba el modelo"]
)

# =============================================================================
# SECCIÓN 1: QUÉ ES LA IA
# =============================================================================
if seccion == "🤖 ¿Qué es la IA?":
    st.title("🤖 Inteligencia Artificial (IA)")
    st.markdown("""
    La **Inteligencia Artificial** es un campo de la computación que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana, como:
    - Reconocer imágenes y sonidos
    - Tomar decisiones
    - Aprender de la experiencia

    ### ¿Cómo aprende una IA?
    Una IA aprende a partir de **datos**. En lugar de programar reglas explícitas (si ves un círculo, entonces es un 0), le mostramos miles de ejemplos y ella misma descubre los patrones.

    ### Tipos de IA
    - **IA débil**: Diseñada para una tarea específica (reconocer dígitos, jugar ajedrez).
    - **IA fuerte**: (teórica) igual o superior a la humana.

    Nuestra aplicación usa una **Red Neuronal Convolucional (CNN)**, un tipo de IA débil especializada en visión por computadora.
    """)
    st.image("https://www.ibm.com/content/dam/connectedassets-adobe-cms/worldwide-content/creative-assets/s-migr/ul/g/8a/ai-article.png", caption="La IA imita el aprendizaje humano", width=400)

# =============================================================================
# SECCIÓN 2: QUÉ ES UNA CNN (con partes interactivas)
# =============================================================================
elif seccion == "🧠 ¿Qué es una CNN?":
    st.title("🧠 Red Neuronal Convolucional (CNN)")
    st.markdown("""
    Una CNN es un modelo de IA diseñado para **procesar imágenes**. Se inspira en el sistema visual de los animales.

    ### Arquitectura de nuestra CNN
    """)

    # Diagrama estático (reutilizamos la función)
    def draw_architecture():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.axis('off')
        blocks = [
            ("Entrada\n28x28x1", (0, 0.5)),
            ("Conv1 + ReLU\n8 filtros\n3x3", (1, 0.5)),
            ("MaxPool\n2x2", (2, 0.5)),
            ("Conv2 + ReLU\n16 filtros\n3x3", (3, 0.5)),
            ("MaxPool\n2x2", (4, 0.5)),
            ("Aplanar\n16*7*7", (5, 0.5)),
            ("Densa (FC)\n10 salidas", (6, 0.5)),
            ("Softmax\nDígito", (7, 0.5))
        ]
        for texto, (x, y) in blocks:
            ax.text(x, y, texto, ha='center', va='center',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), fontsize=10)
            if x < 7:
                ax.annotate('', xy=(x+0.8, y), xytext=(x+0.2, y), arrowprops=dict(arrowstyle='->', lw=1.5))
        ax.set_xlim(-0.5, 8)
        ax.set_ylim(0, 1)
        return fig

    st.pyplot(draw_architecture())
    st.caption("Arquitectura: Convolución → Pooling → Convolución → Pooling → Capa densa → Softmax")

    # Botones interactivos para explicar cada parte
    st.subheader("🔍 Haz clic en cualquier bloque para entenderlo")
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

    explicaciones = {
        "entrada": "📥 **Entrada**: Imagen en escala de grises de 28x28 píxeles. Cada píxel tiene un valor entre 0 (negro) y 1 (blanco).",
        "conv1": "🔍 **Capa convolucional 1**: Aplica 8 filtros de 3x3 que detectan bordes, líneas y curvas. Genera 8 mapas de 28x28.",
        "relu1": "⚡ **ReLU**: Convierte valores negativos a cero. Introduce no linealidad para aprender patrones complejos.",
        "pool1": "📉 **MaxPooling 1**: Reduce el tamaño a 14x14 tomando el máximo en regiones 2x2. Hace el modelo robusto a deformaciones.",
        "conv2": "🔍 **Capa convolucional 2**: Aplica 16 filtros de 3x3 sobre los 8 mapas anteriores. Combina patrones simples en formas más complejas.",
        "relu2": "⚡ **ReLU**: Igual que antes, aplicada a la segunda convolución.",
        "pool2": "📉 **MaxPooling 2**: Reduce a 7x7. Ahora tenemos 16 mapas de 7x7.",
        "flatten": "📄 **Aplanamiento**: Convierte los 16*7*7 = 784 valores en un vector de 784 números.",
        "fc": "🧠 **Capa densa**: 10 neuronas (una por dígito). Cada neurona asigna una puntuación.",
        "softmax": "🎯 **Softmax**: Convierte puntuaciones en probabilidades (suman 1). El dígito con mayor probabilidad es la predicción."
    }

    with col1:
        if st.button("📥 Entrada", key="btn_entrada"):
            st.session_state["explicacion_cnn"] = explicaciones["entrada"]
    with col2:
        if st.button("🔍 Conv1+ReLU", key="btn_conv1"):
            st.session_state["explicacion_cnn"] = explicaciones["conv1"] + "\n\n" + explicaciones["relu1"]
    with col3:
        if st.button("📉 Pool1", key="btn_pool1"):
            st.session_state["explicacion_cnn"] = explicaciones["pool1"]
    with col4:
        if st.button("🔍 Conv2+ReLU", key="btn_conv2"):
            st.session_state["explicacion_cnn"] = explicaciones["conv2"] + "\n\n" + explicaciones["relu2"]
    with col5:
        if st.button("📉 Pool2", key="btn_pool2"):
            st.session_state["explicacion_cnn"] = explicaciones["pool2"]
    with col6:
        if st.button("📄 Aplanar", key="btn_flatten"):
            st.session_state["explicacion_cnn"] = explicaciones["flatten"]
    with col7:
        if st.button("🧠 Densa", key="btn_fc"):
            st.session_state["explicacion_cnn"] = explicaciones["fc"]
    with col8:
        if st.button("🎯 Softmax", key="btn_softmax"):
            st.session_state["explicacion_cnn"] = explicaciones["softmax"]

    if "explicacion_cnn" not in st.session_state:
        st.session_state["explicacion_cnn"] = "👆 Haz clic en cualquier bloque para conocer su función."
    st.info(st.session_state["explicacion_cnn"])

# =============================================================================
# SECCIÓN 3: CÓMO SE ENTRENA (gráficos y simulación)
# =============================================================================
elif seccion == "📊 ¿Cómo se entrena?":
    # ---------- SIMULACIÓN INTERACTIVA ----------
    st.subheader("📈 Simulación interactiva del aprendizaje")
    st.markdown("Ajusta los parámetros y observa cómo mejora la red durante el entrenamiento.")

    col_sim1, col_sim2, col_sim3 = st.columns(3)
    with col_sim1:
        epochs_sim = st.slider("Número de épocas", min_value=1, max_value=20, value=10, step=1)
    with col_sim2:
        lr_sim = st.select_slider("Tasa de aprendizaje (learning rate)", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
    with col_sim3:
        noise_sim = st.slider("Ruido (variabilidad)", min_value=0.0, max_value=0.1, value=0.05, step=0.01)

    if st.button("🚀 Simular entrenamiento", type="primary"):
        # Simular curvas de pérdida y precisión
        epochs_range = np.arange(1, epochs_sim + 1)
        # Modelo teórico: pérdida decrece exponencialmente, precisión crece logarítmicamente
        loss_curve = np.exp(-epochs_range / (epochs_sim / 2)) + np.random.normal(0, noise_sim, epochs_sim)
        loss_curve = np.clip(loss_curve, 0, 1.5)  # evitar valores negativos
        acc_curve = (1 - np.exp(-epochs_range / (epochs_sim / 3))) * 100
        acc_curve = acc_curve + np.random.normal(0, noise_sim * 10, epochs_sim)
        acc_curve = np.clip(acc_curve, 0, 100)

        # Mostrar gráfico combinado
        fig, ax1 = plt.subplots(figsize=(8, 4))
        color_loss = 'tab:red'
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida (Loss)', color=color_loss)
        ax1.plot(epochs_range, loss_curve, color=color_loss, marker='o', label='Pérdida')
        ax1.tick_params(axis='y', labelcolor=color_loss)

        ax2 = ax1.twinx()
        color_acc = 'tab:blue'
        ax2.set_ylabel('Precisión (%)', color=color_acc)
        ax2.plot(epochs_range, acc_curve, color=color_acc, marker='s', label='Precisión')
        ax2.tick_params(axis='y', labelcolor=color_acc)

        plt.title(f'Curvas de entrenamiento (lr={lr_sim})')
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Mostrar valores finales
        st.success(f"**Resultado final** después de {epochs_sim} épocas:\n"
                   f"- Pérdida final: {loss_curve[-1]:.3f}\n"
                   f"- Precisión final: {acc_curve[-1]:.1f}%")
        st.caption("Nota: Esta es una simulación didáctica. En un entrenamiento real, las curvas pueden tener más oscilaciones y depender de muchos factores.")
    else:
        st.info("Presiona el botón 'Simular entrenamiento' para ver cómo evolucionan la pérdida y la precisión.")

    # Mantén el resto de la sección (imagen y explicación fija) como estaba:
    st.markdown("### 📉 Pérdida vs 📈 Precisión real del modelo entrenado")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*_1t6N0PxLrHq4e4Qk8KkAw.png", caption="Curvas típicas de entrenamiento", width=500)
    st.markdown("""
    - **Pérdida (loss)**: Mide qué tan equivocada está la red. Queremos que baje.
    - **Precisión (accuracy)**: Porcentaje de aciertos. Queremos que suba.
    - Nuestro modelo logró **98.57%** de precisión en datos de prueba.
    """)
# =============================================================================
# SECCIÓN 4: PRUEBA EL MODELO (canvas + predicción + Grad-CAM + mapas de activación)
# =============================================================================
else:  # "✍️ Prueba el modelo"
    st.title("✍️ Prueba el modelo")
    st.markdown("Dibuja un dígito en el lienzo y la CNN lo identificará, mostrándote además **cómo está pensando**.")

    col1, col2 = st.columns([1, 1.5])

    # ------------- LADO IZQUIERDO: CANVAS Y PREDICCIÓN -------------
    with col1:
        st.subheader("🎨 Dibuja tu número")
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#000000",
            width=280,
            height=280,
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            img = canvas_result.image_data.astype(np.uint8)
            img_pil = Image.fromarray(img[:, :, 0])
            img_pil = img_pil.resize((28, 28))

            invert = st.checkbox("Invertir colores (si el fondo es blanco)", value=False)
            if invert:
                img_pil = Image.eval(img_pil, lambda x: 255 - x)

            st.image(img_pil, caption="Imagen redimensionada a 28x28", width=100)

            transform = transforms.ToTensor()
            tensor = transform(img_pil).unsqueeze(0)

            with torch.no_grad():
                output = model(tensor)
                pred = torch.argmax(output, dim=1).item()
                probs = F.softmax(output[0], dim=0)

            st.success(f"### ✅ Dígito reconocido: **{pred}**")
            st.write("**Confianza por dígito:**")
            for i, p in enumerate(probs):
                st.progress(float(p), text=f"{i}: {p:.1%}")

            # Grad-CAM opcional
            st.markdown("---")
            show_gradcam = st.checkbox("🔍 Mostrar explicación visual (Grad-CAM) - ¿por qué eligió ese número?")
            if show_gradcam:
                with st.spinner("Generando mapa de calor explicativo..."):
                    tensor_grad = tensor.clone().detach().requires_grad_(True).to(device)
                    model.train()
                    model.enable_gradcam()
                    model.zero_grad()
                    output_grad = model(tensor_grad)
                    output_grad[0, pred].backward()
                    heatmap = gradcam.generate_gradcam_heatmap(model, tensor_grad, pred)
                    model.eval()
                    model.disable_gradcam()

                    col_g1, col_g2, col_g3 = st.columns(3)
                    with col_g1:
                        st.image(img_pil, caption="Original", width=120)
                    with col_g2:
                        fig_heat = gradcam.plot_heatmap_only(heatmap)
                        st.pyplot(fig_heat)
                        plt.close(fig_heat)
                    with col_g3:
                        overlay_img = gradcam.overlay_heatmap(img_pil, heatmap, alpha=0.6)
                        st.image(overlay_img, caption="Superposición", width=120)

                    with st.expander("📖 ¿Qué significa este mapa de calor?"):
                        st.markdown(f"""
                        - **Rojo/amarillo**: áreas más importantes para clasificar como **{pred}**.
                        - **Azul**: áreas con poca influencia.
                        - Si la red se fijó en partes equivocadas, entenderás por qué se confundió.
                        """)
        else:
            st.info("Dibuja algo en el lienzo de la izquierda...")

    # ------------- LADO DERECHO: MAPAS DE ACTIVACIÓN -------------
    with col2:
        st.subheader("🧠 ¿Qué está viendo la CNN?")
        if canvas_result.image_data is not None:
            with torch.no_grad():
                _ = model(tensor)
                act_conv1 = model.act_conv1[0]  # [8,28,28]
                act_conv2 = model.act_conv2[0]  # [16,14,14]

            def norm_act(act):
                act = act.cpu().numpy()
                return (act - act.min()) / (act.max() - act.min() + 1e-8)

            act1_norm = norm_act(act_conv1)
            act2_norm = norm_act(act_conv2)

            st.markdown("##### Capa 1 (8 filtros)")
            fig1, axes = plt.subplots(2, 4, figsize=(8, 4))
            for i, ax in enumerate(axes.flat):
                if i < 8:
                    ax.imshow(act1_norm[i], cmap='hot')
                    ax.set_title(f'Filtro {i}')
                ax.axis('off')
            st.pyplot(fig1)

            st.markdown("##### Capa 2 (16 filtros) - primeros 8")
            fig2, axes = plt.subplots(2, 4, figsize=(8, 4))
            for i, ax in enumerate(axes.flat):
                if i < 8:
                    ax.imshow(act2_norm[i], cmap='hot')
                    ax.set_title(f'Filtro {i}')
                ax.axis('off')
            st.pyplot(fig2)

            st.caption("""
            **Interpretación**: Cada filtro resalta diferentes partes del número (bordes, curvas, ángulos).  
            La segunda capa combina esos patrones en formas más complejas.
            """)
        else:
            st.warning("Dibuja un dígito para ver cómo la CNN lo analiza paso a paso.")

    # ------------- OPCIONAL: ENTRENAMIENTO REAL (dentro de la misma sección) -------------
    st.markdown("---")
    with st.expander("⚙️ Entrenamiento interactivo (avanzado)"):
        st.markdown("Puedes **re-entrenar el modelo** sobre un subconjunto de MNIST para ver cómo mejora en tiempo real.")
        col_train1, col_train2, col_train3 = st.columns(3)
        with col_train1:
            num_epochs_real = st.number_input("Épocas", 1, 3, 2)
        with col_train2:
            subset_size = st.number_input("Tamaño subconjunto", 100, 2000, 500, step=100)
        with col_train3:
            batch_size_train = st.number_input("Batch size", 16, 128, 64, step=16)

        if st.button("🚀 Comenzar entrenamiento real"):
            import torch.optim as optim
            from torch.utils.data import Subset, DataLoader
            import torchvision.datasets as datasets

            transform_mnist = transforms.ToTensor()
            full_train = datasets.MNIST(root="dataset/", train=True, download=True, transform=transform_mnist)
            indices = np.random.choice(len(full_train), subset_size, replace=False)
            subset = Subset(full_train, indices)
            train_loader = DataLoader(subset, batch_size=batch_size_train, shuffle=True)

            test_dataset = datasets.MNIST(root="dataset/", train=False, download=True, transform=transform_mnist)
            test_loader = DataLoader(test_dataset, batch_size=128)

            model.train()
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = torch.nn.CrossEntropyLoss()

            progress_bar = st.progress(0)
            status_text = st.empty()
            total_batches = len(train_loader) * num_epochs_real
            batch_done = 0

            for epoch in range(num_epochs_real):
                running_loss = 0.0
                for batch_idx, (data, targets) in enumerate(train_loader):
                    data, targets = data.to(device), targets.to(device)
                    optimizer.zero_grad()
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()
                    batch_done += 1
                    progress_bar.progress(batch_done / total_batches)
                    status_text.text(f"Época {epoch+1}/{num_epochs_real} - Lote {batch_idx+1}/{len(train_loader)}")

                avg_loss = running_loss / len(train_loader)
                model.eval()
                correct = 0
                total = 0
                with torch.no_grad():
                    for images, labels in test_loader:
                        images, labels = images.to(device), labels.to(device)
                        outputs = model(images)
                        _, predicted = torch.max(outputs, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
                accuracy = 100.0 * correct / total
                model.train()
                status_text.text(f"Época {epoch+1} completada | Loss: {avg_loss:.4f} | Precisión: {accuracy:.2f}%")
                time.sleep(0.5)

            progress_bar.progress(1.0)
            st.success(f"✅ Entrenamiento completado. Precisión final: {accuracy:.2f}%")
            st.info("Ahora puedes volver a dibujar un dígito; el modelo se ha actualizado.")