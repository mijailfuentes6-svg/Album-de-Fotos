import torch
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import os
import numpy as np

# IMPORTACIÓN CRÍTICA: Traemos la estructura de la red que tú mismo diseñaste.
# Esto asume que el archivo train_model.py está en la misma carpeta.
from train_model import NexusAutoencoder

# =====================================================================
# 1. CONFIGURACIÓN E INICIALIZACIÓN DEL CEREBRO (FASE 2)
# =====================================================================
# Detecta automáticamente el hardware (GPU en tu ASUS TUF, o CPU si no hay gráfica)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instanciamos la red neuronal vacía
model = NexusAutoencoder()

# CARGAMOS TU CONOCIMIENTO: En lugar de usar modelos de Google (ResNet),
# cargamos los pesos (.pth) que tu sistema aprendió en la Fase 1.
model_path = "mi_red_neuronal.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
else:
    print("Advertencia: No se encontró el cerebro '.pth'. Debes entrenar primero en la Fase 1.")

model.to(device)

# .eval() es crucial. Le dice a PyTorch: "Ya no estamos aprendiendo, congela 
# los engranajes porque solo vamos a extraer predicciones (Inferencia)".
model.eval()

# PREPROCESAMIENTO ESTRICTO: Debe ser idéntico al usado en train_model.py.
# Si el modelo aprendió viendo fotos de 128x128, debemos entregarle fotos de 128x128.
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# =====================================================================
# 2. EXTRACCIÓN DEL ESPACIO LATENTE
# =====================================================================
def extract_vector(image_path):
    """
    Convierte una imagen física en una 'huella matemática' de 256 dimensiones.
    """
    try:
        # Abrir y estandarizar la imagen
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device) # Añade dimensión de "Lote" (Batch)
        
        # torch.no_grad() desactiva el cálculo de gradientes. Como no estamos
        # entrenando, esto ahorra muchísima memoria RAM y hace el proceso ultra rápido.
        with torch.no_grad():
            # MAGIA DE INGENIERÍA: Fíjate que no llamamos a model(), sino a model.encoder().
            # Cortamos la red por la mitad. Entra la foto 128x128 y sale un vector [256].
            out = model.encoder(batch_t)
        
        # Lo aplanamos y lo devolvemos a la memoria normal (CPU) para que scikit-learn lo pueda leer
        return out.flatten().cpu().numpy()
        
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

# =====================================================================
# 3. AGRUPACIÓN SEMÁNTICA (FASE 3)
# =====================================================================
def cluster_images(directory, num_clusters=3):
    """
    Toma los vectores matemáticos de todas las imágenes y usa el algoritmo K-Means
    para agruparlos según su distancia en el Espacio Latente.
    """
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # 1. Escanea la carpeta del volumen de Docker y obtiene las rutas absolutas
    image_paths = [
        os.path.abspath(os.path.join(directory, f)) 
        for f in os.listdir(directory) 
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_paths:
        return {}

    # 2. Pasa por todas las fotos, activa el Encoder y guarda sus vectores
    features = []
    final_paths = []
    
    for path in image_paths:
        vector = extract_vector(path)
        if vector is not None:
            features.append(vector)
            final_paths.append(path)
    
    if not features:
        return {}

    # 3. EL ALGORITMO K-MEANS
    # Convierte la lista de vectores en una matriz matemática (Array de NumPy)
    features_matrix = np.array(features)
    
    # Ajuste de seguridad: Si pides 10 álbumes pero solo subiste 5 fotos, 
    # K-Means colapsaría. Esto fuerza a K-Means a hacer máximo 5 álbumes.
    actual_k = min(num_clusters, len(features))
    
    # n_init='auto' optimiza la inicialización de los centroides
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init='auto')
    
    # .fit_predict() hace el trabajo pesado: mide distancias y le asigna una etiqueta (0, 1, 2...) a cada foto
    labels = kmeans.fit_predict(features_matrix)
    
    # 4. TRADUCCIÓN PARA EL FRONTEND (Vue.js)
    # Convertimos los números fríos en un diccionario ordenado que la página web pueda dibujar
    clusters = {}
    for path, label in zip(final_paths, labels):
        cluster_name = f"Clúster {label + 1}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(path)
        
    return clusters

