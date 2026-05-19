import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

# 1. Definir tu propia Arquitectura Neuronal (Autoencoder)
class NexusAutoencoder(nn.Module):
    def __init__(self):
        super(NexusAutoencoder, self).__init__()
        
        # ENCODER: Comprime la imagen (128x128) a un vector de 256 dimensiones
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1), # Salida: 64x64
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), # Salida: 32x32
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # Salida: 16x16
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 16 * 16, 256) # TU VECTOR DE CARACTERÍSTICAS
        )
        
        # DECODER: Reconstruye la imagen a partir del vector
        self.decoder = nn.Sequential(
            nn.Linear(256, 64 * 16 * 16),
            nn.Unflatten(1, (64, 16, 16)),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid() # Valores entre 0 y 1 para los píxeles
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# 2. Lector de tus propias imágenes
class FlatImageFolder(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# 3. Función principal de entrenamiento
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Reducimos a 128x128 para que entrene rápido en tu laptop
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # La ruta virtual donde Docker mapea tus fotos
    dataset = FlatImageFolder(directory="/data/pictures", transform=transform)
    
    if len(dataset) == 0:
        print("No se encontraron imágenes en /data/pictures para entrenar.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = NexusAutoencoder().to(device)
    criterion = nn.MSELoss() # Mide qué tan parecida es la reconstrucción a la original
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 20 # Veces que verá toda tu galería
    
    print("Iniciando entrenamiento desde cero...")
    for epoch in range(epochs):
        total_loss = 0
        for data in dataloader:
            img = data.to(device)
            
            # Forward
            output = model(img)
            loss = criterion(output, img)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        print(f"Época [{epoch+1}/{epochs}], Error de reconstrucción: {total_loss/len(dataloader):.4f}")

    # Guardar los pesos generados
    torch.save(model.state_dict(), "mi_red_neuronal.pth")
    print("¡Entrenamiento finalizado! Modelo guardado como 'mi_red_neuronal.pth'")

if __name__ == "__main__":
    train()