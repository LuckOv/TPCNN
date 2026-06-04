import streamlit as st
import time
import torch
import torchvision.transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import gradcam
from PIL import Image
from streamlit_drawable_canvas import st_canvas
device = "cuda" if torch.cuda.is_available() else "cpu"
from modelo_cnn import CNN


st.set_page_config(page_title="Reconocedor de Dígitos con CNN", layout="wide")
st.title("🖍️ Reconocedor de dígitos con Red Neuronal Convolucional (CNN)")
st.markdown("Dibuja un dígito en el lienzo y la CNN lo identificará, mostrándote además **cómo está pensando**.")

# ------------------ CARGA DEL MODELO ------------------
@st.cache_resource
def load_model():
    model = CNN(in_channels=1, num_classes=10)
    model.load_state_dict(torch.load("MulticlassCNN.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# ------------------ SECCIÓN DIDÁCTICA: DIAGRAMA Y EXPLICACIÓN ------------------
st.markdown("---")
st.header("📖 ¿Cómo funciona esta CNN?")

# Diagrama de la arquitectura con matplotlib
def draw_architecture():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')
    # Posiciones (x, y) de los bloques
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
        ax.text(x, y, texto, ha='center', va='center', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7), fontsize=10)
        if x < 7:
            ax.annotate('', xy=(x+0.8, y), xytext=(x+0.2, y), arrowprops=dict(arrowstyle='->', lw=1.5))
    ax.set_xlim(-0.5, 8)
    ax.set_ylim(0, 1)
    return fig

st.pyplot(draw_architecture())
st.caption("Arquitectura de la CNN: capas convolucionales, pooling y una capa densa final.")

# Explicación paso a paso
with st.expander("🔍 Ver explicación detallada (sin tecnicismos)"):
    st.markdown("""
    **1. La CNN no mira píxel por píxel**  
    Usa pequeñas *ventanas* (filtros) que recorren la imagen buscando patrones simples: líneas, curvas, bordes.  
    En nuestro modelo, la primera capa tiene 8 filtros diferentes.

    **2. Se reduce el tamaño (MaxPooling)**  
    Después de cada capa convolucional, se aplica *MaxPooling*: se divide la imagen en cuadros de 2x2 y se conserva el valor más alto. Esto **resume la información** y hace que el modelo sea más robusto a pequeñas deformaciones.

    **3. Segunda capa convolucional**  
    Los patrones simples se combinan para formar rasgos más complejos (por ejemplo, un círculo o un ángulo). Ahora tenemos 16 filtros.

    **4. Capa completamente conectada**  
    Al final, toda la información se aplana y pasa por una capa que asigna puntuaciones a cada dígito (0-9). El dígito con mayor puntuación es la predicción.

    **5. El entrenamiento** (las épocas)  
    - Una **época** es una pasada completa por todos los ejemplos de entrenamiento (60.000 dígitos).  
    - Después de cada lote (grupo de imágenes), el modelo ajusta sus filtros para cometer menos errores.  
    - Nuestro modelo se entrenó durante **10 épocas**. La precisión final en pruebas fue del **98.57%**.
    """)

# ------------------ LADO IZQUIERDO: CANVAS Y PREDICCIÓN ------------------
col1, col2 = st.columns([1, 1.5])

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
        # Convertir canvas a imagen en escala de grises
        img = canvas_result.image_data.astype(np.uint8)
        img_pil = Image.fromarray(img[:, :, 0])  # canal único
        img_pil = img_pil.resize((28, 28))
        
        # Invertir porque la red espera fondo negro, dígito blanco (normalmente)
        # El canvas ya tiene fondo negro y trazo blanco, así que no invertimos
        # Pero si lo prefieres, puedes dar opción.
        invert = st.checkbox("Invertir colores (si l fondo es blanco)", value=False)
        if invert:
            img_pil = Image.eval(img_pil, lambda x: 255 - x)
        
        st.image(img_pil, caption="Imagen redimensionada a 28x28", width=100)
        
        # Transformar a tensor
        transform = torchvision.transforms.ToTensor()
        tensor = transform(img_pil).unsqueeze(0)  # [1,1,28,28]
        
        # Predicción
        with torch.no_grad():
            output = model(tensor)
            pred = torch.argmax(output, dim=1).item()
            probs = F.softmax(output[0], dim=0)
        
        st.success(f"### ✅ Dígito reconocido: **{pred}**")
        st.write("**Confianza por dígito:**")
        for i, p in enumerate(probs):
            st.progress(float(p), text=f"{i}: {p:.1%}")
            
         # ================= GRAD-CAM (opcional) =================
        st.markdown("---")
        show_gradcam = st.checkbox("🔍 Mostrar explicación visual (Grad-CAM) - ¿por qué eligió ese número?")
        
        if show_gradcam:
            with st.spinner("Generando mapa de calor explicativo..."):
                # IMPORTANTE: Necesitamos hacer un forward con gradientes
                # Clonamos el tensor y lo movemos al dispositivo (debería estar ya)
                tensor_grad = tensor.clone().detach().requires_grad_(True).to(device)
                
                # Activar el modo de entrenamiento temporalmente (para que guarde gradientes)
                model.train()  # Necesario para .backward()
                model.enable_gradcam()   # activamos hook
                
                model.zero_grad()
                
                # Forward y backward para la clase predicha
                output_grad = model(tensor_grad)
                output_grad[0, pred].backward()
                
                # Generar heatmap con funciones de gradcam.py
                heatmap = gradcam.generate_gradcam_heatmap(model)
                
                # Desactivar modo y hook
                model.eval()
                model.disable_gradcam()
                
                # Mostrar resultados
                col_g1, col_g2, col_g3 = st.columns(3)
                with col_g1:
                    st.image(img_pil, caption="Dígito original", width=120)
                with col_g2:
                    fig_heat = gradcam.plot_heatmap_only(heatmap)
                    st.pyplot(fig_heat)
                    plt.close(fig_heat)
                with col_g3:
                    overlay_img = gradcam.overlay_heatmap(img_pil, heatmap, alpha=0.6)
                    st.image(overlay_img, caption="Superposición Grad-CAM", width=120)
                
                with st.expander("📖 ¿Qué significa este mapa de calor?"):
                    st.markdown(f"""
                    - Las áreas **rojas/amarillas** fueron las más importantes para que la CNN decidiera que es un **{pred}**.
                    - Las áreas **azules** influyeron poco o nada.
                    - Esto te permite entender si la red se fijó en las partes correctas del dígito.
                    """)
                
    #else:
        #st.info("Dibuja algo en el lienzo de la izquierda...")
           

# ------------------ LADO DERECHO: VISUALIZACIÓN DE MAPAS DE ACTIVACIÓN ------------------
with col2:
    st.subheader("🧠 ¿Qué está viendo la CNN?")
    if canvas_result.image_data is not None:
        # Obtenemos los mapas de activación de conv1 y conv2 usando un hook o re-ejecutando forward
        # Como ya tenemos el modelo cargado, podemos hacer un forward especial que guarde las activaciones
        # Pero nuestro modelo ya las guarda en self.act_conv1 y self.act_conv2 durante el forward.
        # Así que volvemos a pasar el tensor (sin gradiente)
        with torch.no_grad():
            _ = model(tensor)
            act_conv1 = model.act_conv1[0]  # [8, 28, 28] después de ReLU pero antes de pooling
            act_conv2 = model.act_conv2[0]  # [16, 14, 14] después de ReLU pero antes de pooling
        
        # Normalizar para visualización
        def norm_act(act):
            act = act.cpu().numpy()
            act = (act - act.min()) / (act.max() - act.min() + 1e-8)
            return act
        
        act1_norm = norm_act(act_conv1)
        act2_norm = norm_act(act_conv2)
        
        # Mostrar primeras 8 características de conv1
        st.markdown("##### Capa convolucional 1 (8 filtros)")
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        for i, ax in enumerate(axes.flat):
            if i < act1_norm.shape[0]:
                ax.imshow(act1_norm[i], cmap='hot')
                ax.set_title(f'Filtro {i}')
            ax.axis('off')
        st.pyplot(fig)
        
        st.markdown("##### Capa convolucional 2 (16 filtros) - primeros 8")
        fig, axes = plt.subplots(2, 4, figsize=(8, 4))
        for i, ax in enumerate(axes.flat):
            if i < act2_norm.shape[0]:
                ax.imshow(act2_norm[i], cmap='hot')
                ax.set_title(f'Filtro {i}')
            ax.axis('off')
        st.pyplot(fig)
        
        st.caption("""
        **¿Qué significan estos mapas?**  
        Cada filtro resalta diferentes partes del número: bordes, curvas, ángulos.  
        La segunda capa combina esos patrones para formar partes más complejas (círculos, líneas largas).
        """)
    else:
        st.warning("Dibuja un dígito para ver cómo la CNN lo analiza paso a paso.")



# ------------------ ENTRENAMIENTO REAL (sobre un subconjunto) ------------------
st.subheader("⚙️ Entrenamiento interactivo de la CNN")

st.markdown("""
Entrena la red en tiempo real y observa cómo mejora:

- 📉 La pérdida (error) baja
- 📈 La precisión sube
- 🧠 La red aprende patrones visuales
""")

col_ctrl, col_chart, col_viz = st.columns([1, 1.5, 1.5])

with col_ctrl:
    num_epochs = st.slider("Épocas", 1, 5, 2)
    subset_size = st.slider("Datos de entrenamiento", 100, 2000, 500, step=100)
    batch_size = st.slider("Batch size", 16, 128, 64, step=16)

    start_train = st.button("🚀 Entrenar modelo", use_container_width=True)
    
with col_chart:
    st.markdown("### 📈 Aprendizaje en vivo")

    chart_data = {"loss": [], "accuracy": []}
    chart = st.empty()

    status = st.empty()
    
with col_viz:
    st.markdown("### 🧠 Qué está aprendiendo")

    img_placeholder = st.empty()
    pred_placeholder = st.empty()
    
if start_train:
    import torch.optim as optim
    from torch.utils.data import Subset, DataLoader
    import torchvision.datasets as datasets
    import torchvision.transforms as transforms
    import random

    transform = transforms.ToTensor()

    full_train = datasets.MNIST(root="dataset/", train=True, download=True, transform=transform)
    indices = np.random.choice(len(full_train), subset_size, replace=False)
    subset = Subset(full_train, indices)

    train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

    test_dataset = datasets.MNIST(root="dataset/", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=128)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        running_loss = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)

        # 📊 Evaluación
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

        accuracy = 100 * correct / total

        model.train()

        losses.append(avg_loss)
        accuracies.append(accuracy)

        # 📈 Actualizar gráfico
        chart_data["loss"].append(avg_loss)
        chart_data["accuracy"].append(accuracy)

        chart.line_chart(chart_data)

        status.markdown(f"""
        **Época {epoch+1}/{num_epochs}**  
        📉 Loss: `{avg_loss:.4f}`  
        📈 Accuracy: `{accuracy:.2f}%`
        """)

        # 🧠 Mostrar ejemplo real
        img, label = random.choice(test_dataset)
        img_tensor = img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = torch.argmax(model(img_tensor)).item()

        # Convertir tensor a imagen para Streamlit
        img_np = img.squeeze().cpu().numpy()  # shape (28,28), valores entre 0 y 1
        img_placeholder.image(img_np, width=120, clamp=True)  # clamp=True evita problemas de rango

        if pred == label:
            pred_placeholder.success(f"Predicción: {pred} ✅ (correcto)")
        else:
            pred_placeholder.error(f"Predicción: {pred} ❌ (era {label})")