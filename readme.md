# ORGANIZACIÓN INTELIGENTE DE ÁLBUMES DE FOTOS

**Proyecto Académico - Ingeniería de Sistemas**
**Universidad Mayor de San Simón (UMSS)**

## Descripción

Este repositorio contiene el código fuente de un sistema web para la organización automática de álbumes fotográficos mediante técnicas de aprendizaje no supervisado.

A diferencia de soluciones basadas en modelos preentrenados o etiquetado manual, este proyecto implementa una arquitectura propia entrenada desde cero para extraer características visuales y agrupar imágenes según su similitud semántica.

---

## Características Principales

### Autoencoder Convolucional Personalizado

Arquitectura desarrollada en PyTorch capaz de comprimir información visual y extraer características relevantes mediante un espacio latente de 256 dimensiones.

### Clustering Dinámico con K-Means

Agrupación automática de imágenes utilizando el algoritmo K-Means, basado en la similitud matemática de los vectores de características generados por la red neuronal.

### Monitoreo y Telemetría en Tiempo Real

Interfaz desarrollada con Vue.js que permite visualizar:

* Evolución de la pérdida (MSE Loss)
* Progreso del entrenamiento
* Proyección del espacio latente mediante PCA
* Métricas de rendimiento del modelo

### Optimización de Recursos

Generación de álbumes comprimidos en formato ZIP utilizando buffers en memoria (`io.BytesIO`), evitando escrituras innecesarias en disco y mejorando el rendimiento del servidor.

---

## Pila Tecnológica

### Backend e Inteligencia Artificial

* Python 3.x
* PyTorch
* Scikit-Learn
* FastAPI

### Frontend

* Vue.js 3
* Tailwind CSS
* Chart.js

### Infraestructura y Despliegue

* Docker
* Docker Compose

---

## Estructura del Proyecto

```text
.
├── main.py                # API REST y orquestación principal
├── train_model.py         # Entrenamiento del Autoencoder
├── ml_engine.py           # Inferencia y clustering K-Means
├── mi_red_neuronal.pth    # Pesos entrenados del modelo
├── requirements.txt       # Dependencias de Python
├── docker-compose.yml     # Orquestación de contenedores
├── Dockerfile             # Configuración del entorno
└── static/
    ├── index.html         # Interfaz principal
    ├── css/
    │   └── styles.css
    └── js/
        └── app.js
```

La arquitectura sigue el principio de **Separation of Concerns (SoC)**, facilitando el mantenimiento y la escalabilidad del sistema.

---

## Despliegue con Docker (Recomendado)

### Requisitos

* Docker
* Docker Compose

### Ejecución

Desde la raíz del proyecto:

```bash
docker-compose up --build
```

Una vez finalizada la construcción, acceder desde el navegador:

```text
http://localhost:8000
```

---

## Ejecución Local

### 1. Crear y activar un entorno virtual

```bash
python -m venv venv
```

#### Windows

```bash
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Iniciar el servidor

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Abrir la aplicación

```text
http://localhost:8000
```

---

## Guía de Uso

### Fase 1: Entrenamiento

1. Cargar un conjunto de imágenes.
2. Iniciar el entrenamiento del Autoencoder.
3. Observar la evolución del error de reconstrucción (MSE Loss) en tiempo real.

### Fase 2: Inferencia

1. Procesar nuevas imágenes.
2. Extraer automáticamente los vectores de características (embeddings).
3. Mantener los pesos del modelo sin modificaciones.

### Fase 3: Agrupación

1. Definir el número de grupos (K).
2. Ejecutar el algoritmo K-Means.
3. Revisar los álbumes generados automáticamente.
4. Descargar los resultados en formato ZIP.

---

## Objetivo Académico

Demostrar la aplicación práctica de técnicas de:

* Redes Neuronales Convolucionales
* Autoencoders
* Reducción de Dimensionalidad (PCA)
* Clustering No Supervisado (K-Means)
* Desarrollo Web con FastAPI y Vue.js
* Despliegue mediante Contenedores Docker

---

## Licencia

Proyecto desarrollado con fines académicos para la carrera de Ingeniería de Sistemas de la Universidad Mayor de San Simón (UMSS).
