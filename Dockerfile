# Usamos una imagen ligera de Python 3.12
FROM python:3.12-slim

# Evita que Python genere archivos .pyc y permite ver logs en tiempo real
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Instalamos dependencias del sistema necesarias para procesamiento de imágenes
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiamos el archivo de requerimientos primero para aprovechar el cache de capas
COPY requirements.txt .

# Instalamos PyTorch primero (es muy pesado, necesita más tiempo)
RUN pip install --no-cache-dir --timeout=1000 --retries=10 torch torchvision

# Instalamos el resto de dependencias
RUN pip install --no-cache-dir --timeout=300 fastapi uvicorn scikit-learn pillow pydantic python-multipart

# Copiamos todo el código del proyecto (main.py, ml_engine.py, static/, etc.)
COPY . .

# Exponemos el puerto que usa FastAPI
EXPOSE 8000

# Comando para arrancar la aplicación
# Usamos 0.0.0.0 para que sea accesible desde fuera del contenedor
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]