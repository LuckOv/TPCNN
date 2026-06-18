import streamlit as st
import time
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from src import gradcam
from src.model import CNN


# ------------------ FUNCIÓN PARA ANIMACIÓN DE CONVOLUCIÓN ------------------
def mostrar_convolucion_interactiva():
    """Muestra una animación interactiva de cómo un filtro recorre la imagen."""
    st.markdown("### 🎬 ¿Cómo funciona la convolución?")
    st.markdown("""
    Un **filtro (kernel)** se desliza sobre la imagen realizando multiplicaciones y sumas.
    Cada posición genera un valor que forma el **mapa de características**.
    """)
    
    # Crear una imagen de ejemplo (un dígito 8 simple)
    img_ejemplo = np.zeros((28, 28))
    # Dibujar un 8 simple
    for i in range(28):
        for j in range(28):
            # Círculo superior
            dist_sup = np.sqrt((i-9)**2 + (j-14)**2)
            # Círculo inferior
            dist_inf = np.sqrt((i-19)**2 + (j-14)**2)
            if dist_sup < 5 or dist_inf < 5:
                img_ejemplo[i, j] = 1.0
            # Línea conectora
            if abs(i - 14) < 3 and j > 11 and j < 17:
                img_ejemplo[i, j] = 1.0
            # Pequeño ruido para más realismo
            if dist_sup < 6 or dist_inf < 6:
                img_ejemplo[i, j] = max(img_ejemplo[i, j], 0.9)
    
    # Crear un filtro simple (detector de bordes horizontales)
    filtro = np.array([
        [-1, -1, -1],
        [0, 0, 0],
        [1, 1, 1]
    ])
    
    col_anim1, col_anim2 = st.columns(2)
    
    with col_anim1:
        st.markdown("**Imagen original (dígito 8 de ejemplo)**")
        fig_img, ax_img = plt.subplots(figsize=(4, 4))
        ax_img.imshow(img_ejemplo, cmap='gray')
        ax_img.set_title("Imagen 28x28 píxeles")
        ax_img.axis('off')
        st.pyplot(fig_img)
        plt.close(fig_img)
        
        st.markdown("**Filtro (kernel) 3x3 - Detector de bordes horizontales**")
        fig_filtro, ax_filtro = plt.subplots(figsize=(3, 3))
        ax_filtro.imshow(filtro, cmap='RdBu', vmin=-1, vmax=1)
        for i in range(3):
            for j in range(3):
                ax_filtro.text(j, i, f"{filtro[i, j]:.0f}", ha='center', va='center', color='white', fontsize=12)
        ax_filtro.set_title("Kernel 3x3")
        ax_filtro.axis('off')
        st.pyplot(fig_filtro)
        plt.close(fig_filtro)
    
    with col_anim2:
        st.markdown("**🎮 Simula el movimiento del filtro**")
        st.markdown("El filtro recorre TODA la imagen, pero puedes explorar posiciones específicas:")
        
        # Posiciones disponibles: desde 0 hasta 25 (para imagen 28x28 con kernel 3x3, stride=1)
        pos_x = st.slider("Posición X (columna)", 0, 25, 12, key="conv_x")
        pos_y = st.slider("Posición Y (fila)", 0, 25, 12, key="conv_y")
        
        # Mostrar la posición del filtro en la imagen
        fig_pos, ax_pos = plt.subplots(figsize=(5, 5))
        ax_pos.imshow(img_ejemplo, cmap='gray')
        
        # Dibujar cuadrado del filtro
        from matplotlib.patches import Rectangle
        rect = Rectangle((pos_x-0.5, pos_y-0.5), 3, 3, 
                          fill=False, edgecolor='red', linewidth=2)
        ax_pos.add_patch(rect)
        
        # Marcar el centro
        ax_pos.plot(pos_x+1, pos_y+1, 'r+', markersize=12, linewidth=2)
        
        ax_pos.set_title(f"Filtro en posición ({pos_x}, {pos_y})")
        ax_pos.axis('off')
        st.pyplot(fig_pos)
        plt.close(fig_pos)
    
    # Calcular y mostrar la región y el resultado
    st.markdown("---")
    st.markdown("### 🔢 ¿Qué calcula el filtro en esta posición?")
    
    # Extraer la región de la imagen (asegurar límites)
    region = img_ejemplo[pos_y:pos_y+3, pos_x:pos_x+3]
    # Calcular la convolución (suma de elementos multiplicados)
    resultado = np.sum(region * filtro)
    
    col_calc1, col_calc2, col_calc3 = st.columns(3)
    
    with col_calc1:
        st.markdown("**Región de la imagen (3x3)**")
        fig_reg, ax_reg = plt.subplots(figsize=(3, 3))
        ax_reg.imshow(region, cmap='gray', vmin=0, vmax=1)
        for i in range(3):
            for j in range(3):
                ax_reg.text(j, i, f"{region[i, j]:.2f}", ha='center', va='center', 
                           color='white' if region[i, j] < 0.6 else 'black', fontsize=8)
        ax_reg.axis('off')
        st.pyplot(fig_reg)
        plt.close(fig_reg)
    
    with col_calc2:
        st.markdown("**Filtro (kernel)**")
        fig_fil2, ax_fil2 = plt.subplots(figsize=(3, 3))
        ax_fil2.imshow(filtro, cmap='RdBu', vmin=-1, vmax=1)
        for i in range(3):
            for j in range(3):
                ax_fil2.text(j, i, f"{filtro[i, j]:.0f}", ha='center', va='center', color='white', fontsize=12)
        ax_fil2.axis('off')
        st.pyplot(fig_fil2)
        plt.close(fig_fil2)
    
    with col_calc3:
        st.markdown("**Multiplicación elemento a elemento**")
        st.markdown("Región × Filtro:")
        resultado_matriz = region * filtro
        st.text("")
        for i in range(3):
            fila_texto = "  ".join([f"{resultado_matriz[i, j]:+5.2f}" for j in range(3)])
            st.code(fila_texto, language="text")
        st.markdown(f"**SUMA TOTAL = {resultado:+.3f}**")
    
    st.info(f"""
    💡 **Interpretación del resultado {resultado:+.3f}**:  
    - Si es **positivo alto (> 1.0)**: El filtro detecta un borde horizontal fuerte.  
    - Si es **cercano a 0**: No hay patrón coincidente.  
    - Si es **negativo**: Detecta el patrón inverso (borde en dirección opuesta).  
    
    El filtro completo recorre la imagen generando un **mapa de características** de tamaño 26x26.
    """)
    
    # Mostrar mapa de características completo
    st.markdown("---")
    st.markdown("### 🗺️ Mapa de características completo")
    st.markdown("El filtro se aplica en CADA posición de la imagen. Así se ve el resultado final:")
    
    # Calcular la convolución completa (sin usar scipy para no agregar dependencias)
    def convolucion_simple(imagen, kernel):
        h, w = imagen.shape
        kh, kw = kernel.shape
        output_h = h - kh + 1
        output_w = w - kw + 1
        resultado = np.zeros((output_h, output_w))
        for i in range(output_h):
            for j in range(output_w):
                resultado[i, j] = np.sum(imagen[i:i+kh, j:j+kw] * kernel)
        return resultado
    
    mapa_caracteristicas = convolucion_simple(img_ejemplo, filtro)
    
    col_map1, col_map2 = st.columns(2)
    with col_map1:
        st.markdown("**Mapa de características**")
        fig_map, ax_map = plt.subplots(figsize=(5, 5))
        im = ax_map.imshow(mapa_caracteristicas, cmap='hot')
        ax_map.set_title(f"Tamaño: {mapa_caracteristicas.shape[0]}x{mapa_caracteristicas.shape[1]}")
        ax_map.axis('off')
        plt.colorbar(im, ax=ax_map, shrink=0.7)
        st.pyplot(fig_map)
        plt.close(fig_map)
    
    with col_map2:
        st.markdown("**Interpretación**")
        st.markdown("""
        - **Colores cálidos (rojo/amarillo)**: El filtro encontró bordes horizontales en esa zona.
        - **Colores fríos (azul/negro)**: No se detectó el patrón.
        - **Tamaño reducido**: La imagen pasó de 28×28 a 26×26 píxeles.
        
        🧠 **Esto es exactamente lo que hace nuestra CNN**:  
        Aplica múltiples filtros (8 en la primera capa) que aprenden patrones como bordes, curvas y esquinas.  
        Cada filtro genera un mapa de características, y la red combina esa información para reconocer el dígito.
        """)
    
    # Explicación final
    st.markdown("---")
    st.markdown("""
    ### 📚 Resumen visual del proceso completo
    
    1. **Filtro** (kernel) pequeño se desliza sobre la imagen
    2. En cada posición: multiplica elemento a elemento
    3. Suma los resultados → un solo número
    4. Ese número va al mapa de características
    5. El resultado final es una imagen más pequeña que resalta donde aparece el patrón
    
    **¡Así es como las redes convolucionales "ven" las imágenes!**
    """)

# Configuración de la página
st.set_page_config(page_title="Reconocedor de Dígitos con CNN", layout="wide")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------ CARGA DEL MODELO (cache) ------------------
@st.cache_resource
def load_model() -> CNN:
    model = CNN(in_channels=1, num_classes=10)
    try:
        state = torch.load("MulticlassCNN.pth", map_location="cpu", weights_only=True)
        model.load_state_dict(state)
    except FileNotFoundError:
        st.error("""
        **Archivo `MulticlassCNN.pth` no encontrado.**
        Ejecutá `python train.py` en la terminal para entrenar el modelo primero.
        """)
        st.stop()
    except (RuntimeError, KeyError) as e:
        st.error(f"""
        **Error al cargar los pesos del modelo.**
        Posiblemente el archivo `MulticlassCNN.pth` corresponde a una versión anterior de la arquitectura.
        Ejecutá `python train.py` para reentrenar el modelo.
        Detalle: {e}
        """)
        st.stop()
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
    ### Definición de Inteligencia Artificial
    La Inteligencia Artificial (IA) es un programa de computadora que puede realizar tareas que normalmente requerirían inteligencia humana.
    
    Piénsenlo así: un ajedrecista humano analiza el tablero, reconoce patrones, anticipa movimientos del rival y toma decisiones. Una IA de ajedrez (como AlphaZero) hace exactamente lo mismo, pero procesando información a velocidad electrónica.
    
    ## Ejemplos cotidianos:
    - El asistente de voz que responde preguntas (Siri, Alexa)
    - El filtro de spam en tu email
    - La recomendación de películas en Netflix
    - Los autos que conducen solos

    *Lo clave: la IA automatiza decisiones y tareas inteligentes, no reemplaza la inteligencia humana.*

    ### ¿Cómo aprende una IA?
    Una IA no nace sabiendo cosas. Aprende de ejemplos, similar a cómo aprendemos ustedes.
    
    ## El proceso básico:
    1. Se le muestran muchos ejemplos → Un detector de correos spam ve miles de correos: algunos legítimos, otros basura.
    2. Identifica patrones → "Hmm, los correos spam suelen tener palabras como 'gana dinero', 'urgente', 'click aquí'..."
    3. Ajusta sus reglas internas → La IA dice: "Voy a ponderar más estas palabras como indicadores de spam".
    4. Prueba con ejemplos nuevos → Se le muestra un correo que nunca vio antes y tiene que decidir: ¿spam o no spam?
    5. Mejora con retroalimentación → Si se equivoca, aprende del error.

    *Analogía: Es como aprender a reconocer frutas. Al principio no sabes qué es una manzana. Pero si alguien te muestra 100 manzanas rojas, verdes, grandes y pequeñas, tu cerebro capta el patrón: "Ah, esto es una manzana". Después reconoces una manzana nueva aunque nunca la hayas visto.*
        
    ### Tipos de IA
    Existen varias maneras de clasificar las IA. Les doy dos enfoques útiles:
        
    ## Por nivel de capacidad:
    - IA Débil (Narrow AI): Especializada en una sola tarea. Hace UNA cosa muy bien. Un programa que solo juega ajedrez.
    - IA Fuerte (General AI): Podría hacer cualquier tarea inteligente, como un humano. No existe aún es teórica.
    - IA Super (ASI): Hipotética. Más inteligente que cualquier humano. Ciencia ficción (por ahora).
        
    *Realidad actual: Todas las IA que existen hoy son débiles. ChatGPT habla de muchos temas, pero sigue siendo especializada en tareas de lenguaje.*
        
    ## Por técnica de aprendizaje:
        
    - **Aprendizaje Supervisado**: Se entrena con ejemplos etiquetados ("esto es gato", "esto es perro"). Reconocimiento facial, diagnóstico médico.
    - **Aprendizaje No Supervisado**: Encuentra patrones en datos sin etiquetas previas. Agrupar clientes por comportamiento de compra.
    - **Aprendizaje por Refuerzo**: Aprende mediante ensayo-error, ganando "recompensas" por buenas acciones. AlphaGo jugando ajedrez, robots aprendiendo a caminar.

    **Nuestra aplicación usa una **Red Neuronal Convolucional (CNN)**, un tipo de IA débil especializada en visión por computadora.**
    """)

# =============================================================================
# SECCIÓN 2: QUÉ ES UNA CNN (con partes interactivas)
# =============================================================================
elif seccion == "🧠 ¿Qué es una CNN?":
    st.title("🧠 Red Neuronal Convolucional (CNN)")
    st.markdown("""

    ## ¿Qué son y para qué sirven?
    Las redes neuronales convolucionales, conocidas como CNN, son sistemas informáticos que buscan patrones en datos para clasificarlos. Su característica principal es que trabajan específicamente con imágenes.
    En esta aplicación, la red recibe miles de imágenes de entrenamiento del conjunto de datos MNIST. A partir de estas imágenes, la red aprende a identificar las características únicas de cada dígito (del 0 al 9) para poder reconocer correctamente cualquier número. Por ejemplo, al terminar su aprendizaje, la red puede distinguir cuando una imagen contiene el dígito 6.
    
    ## ¿Cómo funcionan?
    Las CNN imitan la forma en que el cerebro humano procesa las imágenes. Nuestro cerebro divide este proceso en varias capas especializadas:

    - **Primera capa**: detecta patrones simples como líneas y bordes
    - **Segunda capa**: conecta esos patrones para formar figuras
    - **Capas intermedias**: identifican características cada vez más complejas
    - **Capa final**: combina toda la información y determina qué objeto es

    **Las CNN funcionan de manera similar**: comienzan extrayendo líneas y bordes, y conforme avanzan, combinan estos elementos para identificar el objeto completo en la imagen.
    
    ## Cómo las computadoras representan las imágenes
    Las imágenes digitales se almacenan como matrices de píxeles. Cada píxel tiene un valor numérico que va de 0 (oscuro) a 255 (blanco), pasando por todos los tonos de gris.
        *Por ejemplo, una imagen de 28x28 píxeles se representa como una matriz de 28 filas y 28 columnas, donde cada elemento es un número que indica la intensidad del píxel. En nuestro caso, como trabajamos con imágenes en escala de grises, cada píxel tiene un solo valor (en lugar de tres para RGB).*
    
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Ejemplo de imagen MNIST")
        st.image("assets/8.png", caption="Ejemplo de imagen MNIST (28x28 píxeles)", width=300)
    with col2:
        st.markdown("### Representación numérica")
        st.image("assets/8M.png", caption="Matriz de píxeles (28x28)", width=300)
        
    st.markdown ("""
    ## Estructura de una Red Convolucional
    ### Una CNN sigue este proceso:

    - **Entrada**: se recibe la imagen digital
    - **Capas convolucionales**: extraen características mediante filtros especializados. Cada filtro se enfoca en detectar un patrón diferente (líneas, curvas, texturas, etc.)
    - **Capas de reducción (Max Pooling)**: reducen el tamaño de la imagen, conservando solo la información más importante
    - **Repetición**: este proceso se repite varias veces. Con cada iteración, la imagen se hace más pequeña, pero la red detecta características más complejas
    - **Capas finales (Fully Connected y Softmax)**: toman todas las características extraídas y las clasifican, decidiendo qué objeto o número contiene la imagen
    """)
    
    col1 = st.columns(1)
    with col1[0]:
         st.markdown("### Visualización de capas y filtros")
         st.image("assets/diagramaGeneralCNN.png", caption="Ejemplo de filtros aprendidos por una CNN", width=800)

    st.markdown("""
                
    ### Los filtros: el elemento clave
    Los filtros *(también llamados kernels)* son matrices de números que "barren" la imagen para extraer características. Durante el entrenamiento, estos números se ajustan automáticamente para capturar mejor los patrones importantes.
    La convolución es el proceso de aplicar estos filtros a la imagen. Por ejemplo, un filtro puede estar diseñado para detectar bordes, mientras que otro detecta esquinas. Al aplicar múltiples filtros, la red obtiene una visión completa de las características presentes en la imagen.

    """)

    st.markdown("""        
    ## Arquitectura de nuestra CNN
    #AGREGAR LEYENDAS CON LAS CAPAS Y FILTROS DE CADA CAPA COMO ESTABA EN EL GRAFICO ANTERIOR
    
    Nuestra CNN tiene la siguiente estructura:
    1. **Primer equipo de observadores (Capa convolucional 1)**: 
    Tenemos 8 "detectores" que buscan patrones muy simples en la imagen: 
    esquinas, líneas rectas, puntos. Es como tener 8 personas mirando la imagen 
    desde diferentes ángulos para buscar detalles básicos.

    2. **Amplificador (ReLU)**: 
    Cuando encontramos algo interesante, lo amplificamos. Ignoramos los detalles 
    débiles para enfocarnos solo en lo importante (como subir el volumen de una 
    conversación importante).

    3. **Comprimidor inteligente (MaxPooling)**: 
    La imagen es muy grande. Tomamos pequeños cuadrados y nos quedamos solo con 
    el detalle más importante de cada uno. Esto reduce el tamaño a la mitad, 
    haciendo todo más rápido sin perder lo esencial.

    4. **Segundo equipo de observadores (Capa convolucional 2)**: 
    Ahora tenemos 16 "detectores" más sofisticados que buscan patrones complejos: 
    curves, combinaciones de líneas, formas que parecen dígitos. Trabajamos con 
    los resultados del equipo anterior.

    5. **Amplificador (ReLU)**: 
    Nuevamente, amplificamos lo importante.

    6. **Comprimidor (MaxPooling)**: 
    Comprimimos una vez más hasta detalles muy pequeños.

    7. **Desenrollador (Aplanar)**: 
    Convertimos toda la información visual en una larga lista de números 
    (como convertir una foto en código de barras).

    8. **Tomador de decisión final (Capa densa)**: 
    Tenemos 10 "jueces", uno para cada dígito (0 al 9). Cada juez mira toda 
    la información y da su opinión: "Esto parece un 3" o "Podría ser un 8".

    9. **Conversor de opiniones (Softmax)**: 
    Convertimos las opiniones en probabilidades reales. Por ejemplo: 
    "80% seguro de que es un 3, 15% de que es un 8, 5% otras opciones".
    """)
    
    col1 = st.columns(1)
    with col1[0]:
         st.markdown("### Visualización de capas y filtros utilizados en nuetro modelo")
         st.image("assets/diagramaNuestraCNN.png", caption="Diagrama de capas y filtros de nuestra CNN", width=800)
         
        # ========== ANIMACIÓN INTERACTIVA DE CONVOLUCIÓN ==========
    # Llamar a la función que muestra la animación
    mostrar_convolucion_interactiva()
    
# =============================================================================
# SECCIÓN 3: CÓMO SE ENTRENA (gráficos y simulación)
# =============================================================================
elif seccion == "📊 ¿Cómo se entrena?":
    # ---------- EXPLICACIÓN TEÓRICA ----------
    st.title("📊 Entrenamiento de la CNN")
    st.markdown("El entrenamiento de una red neuronal convolucional implica ajustar los pesos y sesgos de la red para minimizar la diferencia entre las predicciones y las etiquetas reales.")
       
    # ---------- SIMULACIÓN INTERACTIVA ----------
    st.subheader("📈 Simulación interactiva del aprendizaje")
    st.markdown("Ajusta los parámetros y observa cómo mejora la red durante el entrenamiento.")
    st.markdown("""
        - **Número de épocas**: Cuántas veces la red ve todo el conjunto de entrenamiento. Más épocas = más aprendizaje, pero cuidado con el sobreajuste. Si sobreajustamos la red es incapaz de analizar correctamente nuevas imágenes, ya que se ha "memorizado" demasiado los ejemplos de entrenamiento. Ajustar este parámetro te permite ver cómo la red mejora con más práctica, pero también cómo puede empezar a fallar si se le da demasiada información.
        
        - **Tasa de aprendizaje (learning rate)**: Qué tan rápido la red ajustaa sus pesos. Un valor muy alto puede hacer que la red no aprenda bien, mientras que uno muy bajo puede hacer que el entrenamiento sea muy lento. El valor alto hace que la red aprenda rápido pero puede pasar por alto detalles importantes, mientras que un valor bajo hace que la red aprenda lentamente pero con más precisión. Ajustar este parámetro te permite ver cómo afecta la velocidad y calidad del aprendizaje.
        
        - **Ruido (variabilidad)**: Simula la incertidumbre y las oscilaciones reales del entrenamiento. Un poco de ruido hace que la simulación sea más realista, mostrando cómo la pérdida y precisión pueden fluctuar antes de estabilizarse. Un entrenamiento igual a 0 sería un laboratorio de experimentos perfecto, pero en la vida real siempre hay algo de ruido. Ajustar este parámetro te permite que el entrenamiento sea más similar a lo que ocurre en la vida real, con sus altibajos y sorpresas.
    """)

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

    st.markdown("""
    - **Pérdida (loss)**: Mide qué tan equivocada está la red. Queremos que baje.
    - **Precisión (accuracy)**: Porcentaje de aciertos. Queremos que suba.
    - **Precisión final**: Con un buen entrenamiento, la red puede alcanzar un porcentaje más alto de precisión en MNIST, lo que significa que reconoce correctamente la mayoría de los dígitos. Sin embargo, el resultado exacto puede variar según los parámetros elegidos y la cantidad de ruido en la simulación.
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
                    model.zero_grad()
                    with torch.enable_grad():
                        output_grad = model(tensor_grad)
                    output_grad[0, pred].backward()
                    heatmap = gradcam.generate_gradcam_heatmap(model)

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
                act_conv1 = model.act_conv1[0]
                act_conv2 = model.act_conv2[0]

            def norm_act(act: torch.Tensor) -> np.ndarray:
                act = act.cpu().numpy()
                return (act - act.min()) / (act.max() - act.min() + 1e-8)

            act1_norm = norm_act(act_conv1)
            act2_norm = norm_act(act_conv2)

            def plot_filters(filters: np.ndarray, title: str, max_show: int = 8) -> None:
                n = min(filters.shape[0], max_show)
                cols = min(n, 8)
                rows = (n + 7) // 8
                st.markdown(f"##### {title} ({filters.shape[0]} filtros, mostrando {n})")
                fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
                for i, ax in enumerate(axes.flat if n > 1 else [axes]):
                    if i < n:
                        ax.imshow(filters[i], cmap='hot')
                        ax.set_title(f'Filtro {i}')
                    ax.axis('off')
                st.pyplot(fig)

            plot_filters(act1_norm, "Capa 1 (conv1)")
            plot_filters(act2_norm, "Capa 2 (conv3)")

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