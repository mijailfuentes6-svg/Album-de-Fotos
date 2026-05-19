import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# =====================================================================
# 1. ARQUITECTURA NEURONAL: EL AUTOENCODER
# =====================================================================
# Un Autoencoder tiene forma de "reloj de arena". Su objetivo no es clasificar,
# sino aprender a comprimir una imagen en un código matemático pequeño y luego 
# intentar reconstruirla con la menor pérdida de calidad posible.
class NexusAutoencoder(nn.Module):
    def __init__(self):
        super(NexusAutoencoder, self).__init__()
        
        # -------------------------------------------------------------
        # ENCODER (El Compresor)
        # -------------------------------------------------------------
        # Usa convoluciones (Conv2d) para extraer patrones visuales (bordes, colores).
        # El parámetro 'stride=2' reduce el tamaño de la imagen a la mitad en cada paso.
        self.encoder = nn.Sequential(
            # Entrada: Imagen RGB de 3 canales (128x128 píxeles)
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),   # Salida: 16 mapas de 64x64
            nn.ReLU(), # Función de activación: apaga los valores negativos
            
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # Salida: 32 mapas de 32x32
            nn.ReLU(),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # Salida: 64 mapas de 16x16
            nn.ReLU(),
            
            # Aplana los mapas 2D en una sola lista unidimensional (un vector largo)
            nn.Flatten(),
            
            # Capa Densa (Linear): Comprime esa lista gigante en exactamente 256 números.
            # ESTE ES EL ESPACIO LATENTE (La huella dactilar de la imagen).
            nn.Linear(64 * 16 * 16, 256) 
        )
        
        # -------------------------------------------------------------
        # DECODER (El Reconstructor)
        # -------------------------------------------------------------
        # Hace exactamente el proceso inverso. Toma los 256 números e intenta
        # redibujar la imagen original usando Deconvoluciones (ConvTranspose2d).
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 16 * 16), # Expande los 256 números al tamaño aplanado
            nn.Unflatten(1, (64, 16, 16)), # Lo vuelve a convertir en bloques 2D (64 mapas de 16x16)
            
            # Sube la resolución de 16x16 a 32x32
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Sube la resolución de 32x32 a 64x64
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            
            # Sube la resolución de 64x64 a 128x128 y reduce a 3 canales (R, G, B)
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            
            # Sigmoid asegura que los valores de los píxeles de salida estén estrictamente entre 0 y 1.
            nn.Sigmoid() 
        )

    # El método 'forward' define cómo viaja la información desde que entra hasta que sale.
    def forward(self, x):
        encoded = self.encoder(x)       # 1. Comprime a 256 dimensiones
        decoded = self.decoder(encoded) # 2. Descomprime a 128x128 RGB
        return decoded


# =====================================================================
# 2. INGESTA DE DATOS (DATASET)
# =====================================================================
# PyTorch necesita saber cómo leer los archivos binarios de tu disco duro
# y convertirlos en tensores (matrices matemáticas que la GPU/CPU puede operar).
class FlatImageFolder(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        # Escanea la carpeta y guarda solo los nombres de los archivos que sean imágenes
        self.images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    # Indica cuántas imágenes hay en total
    def __len__(self):
        return len(self.images)

    # Este método es llamado automáticamente por PyTorch para cargar las imágenes una por una
    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('RGB') # Fuerza el formato a color RGB
        
        # Aplica recortes o conversiones matemáticas si fueron definidas
        if self.transform:
            image = self.transform(image)
        return image


# =====================================================================
# 3. PIPELINE DE ENTRENAMIENTO (BACKPROPAGATION)
# =====================================================================
def train():
    # Detecta si hay tarjeta gráfica Nvidia (CUDA); si no, usa el procesador (CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Pipeline de preprocesamiento: 
    # 1. Escala todas las fotos al mismo tamaño (128x128) sin importar su tamaño original.
    # 2. ToTensor() las convierte en matrices matemáticas y normaliza los píxeles de 0-255 a 0.0-1.0.
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # Instancia la clase de lectura apuntando a la carpeta de Docker
    dataset = FlatImageFolder(directory="/data/pictures", transform=transform)
    
    if len(dataset) == 0:
        print("No se encontraron imágenes en /data/pictures para entrenar.")
        return

    # DataLoader agrupa las imágenes en "lotes" (batches) de 32 en 32.
    # Esto es mucho más eficiente para la memoria que procesar de a una.
    # shuffle=True mezcla las fotos para que la IA no memorice el orden.
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Carga la red neuronal en la memoria de procesamiento
    model = NexusAutoencoder().to(device)
    
    # Función de Pérdida: Error Cuadrático Medio (MSE Loss). 
    # Mide matemáticamente la diferencia entre el píxel original y el píxel reconstruido.
    criterion = nn.MSELoss() 
    
    # Optimizador: El algoritmo que ajusta los "engranajes" (pesos) de la red para reducir el error.
    # Adam es el estándar actual de la industria. lr=0.001 es la "velocidad de aprendizaje".
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20 # Una "época" es una vuelta completa a todo tu dataset de fotos.
    
    print("Iniciando entrenamiento desde cero...")
    for epoch in range(epochs):
        total_loss = 0
        
        # Bucle interno: recorre los lotes de 32 imágenes
        for data in dataloader:
            img = data.to(device) # Envía el lote a la CPU/GPU
            
            # 1. FORWARD PASS (Predicción)
            # Pasamos la imagen por la red. Sale la imagen reconstruida.
            output = model(img)
            
            # 2. CALCULAR EL ERROR
            # Compara la imagen reconstruida (output) vs la original (img).
            loss = criterion(output, img)
            
            # 3. BACKWARD PASS (Retropropagación)
            optimizer.zero_grad() # Limpia la memoria de errores del ciclo anterior
            loss.backward()       # Calcula en qué dirección deben girar los engranajes para mejorar
            optimizer.step()      # Aplica el giro/ajuste a los pesos de la red
            
            total_loss += loss.item()
            
        # Al final de cada época, muestra el promedio de error. Debería ir bajando con cada ciclo.
        print(f"Época [{epoch+1}/{epochs}], Error de reconstrucción: {total_loss/len(dataloader):.4f}")

    # Cuando termina, extrae el "cerebro" (los pesos entrenados) y lo guarda en disco duro.
    torch.save(model.state_dict(), "mi_red_neuronal.pth")
    print("¡Entrenamiento finalizado! Modelo guardado como 'mi_red_neuronal.pth'")

if __name__ == "__main__":
    train()