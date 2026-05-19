import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.cluster import KMeans
import os
import numpy as np

# 1. Configuración de Hardware (Universal)
# Detecta si tienes una GPU NVIDIA (común en laptops ASUS TUF) para ir más rápido
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Inicialización del Modelo
# NOTA: Esto sigue usando ResNet18 (Pre-entrenado). 
# Si tu docente lo prohíbe, aquí deberás cargar tu archivo .pth del Autoencoder.
weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)
model = torch.nn.Sequential(*(list(model.children())[:-1]))
model.to(device) # Movemos el modelo al hardware detectado
model.eval()

# Transformaciones estándar
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def extract_vector(image_path):
    """Extrae la huella matemática de una imagen con manejo de errores."""
    try:
        img = Image.open(image_path).convert('RGB')
        img_t = preprocess(img)
        batch_t = torch.unsqueeze(img_t, 0).to(device) # Enviamos la imagen al mismo hardware
        
        with torch.no_grad():
            out = model(batch_t)
        
        return out.flatten().cpu().numpy() # Devolvemos a CPU para que KMeans (Sklearn) lo entienda
    except Exception as e:
        print(f"Error procesando {image_path}: {e}")
        return None

def cluster_images(directory, num_clusters=3):
    """Agrupa las imágenes del directorio usando K-Means."""
    valid_extensions = ('.jpg', '.jpeg', '.png', '.webp')
    
    # Listar archivos con rutas absolutas compatibles (Universal)
    image_paths = [
        os.path.abspath(os.path.join(directory, f)) 
        for f in os.listdir(directory) 
        if f.lower().endswith(valid_extensions)
    ]
    
    if not image_paths:
        return {}

    # Extraer vectores filtrando los que dieron error
    features = []
    final_paths = []
    
    for path in image_paths:
        vector = extract_vector(path)
        if vector is not None:
            features.append(vector)
            final_paths.append(path)
    
    if not features:
        return {}

    # Ejecutar K-Means
    # Ajustamos n_clusters por si hay menos imágenes que el K solicitado
    actual_k = min(num_clusters, len(features))
    kmeans = KMeans(n_clusters=actual_k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(features)
    
    # Organizar resultados
    clusters = {}
    for path, label in zip(final_paths, labels):
        cluster_name = f"Álbum {label + 1}"
        if cluster_name not in clusters:
            clusters[cluster_name] = []
        clusters[cluster_name].append(path)
        
    return clusters