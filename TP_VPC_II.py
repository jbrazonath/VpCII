#!/usr/bin/env python
# coding: utf-8

# # Universidad de Buenos Aires
# # Aprendizaje Profundo - TP3
# # Cohorte 22 - 5to bimestre 2025
# 

# Este tercer y último TP se debe entregar hasta las **23hs del viernes 12 de diciembre (hora de Argentina)**. La resolución del TP es **individual**. Pueden utilizar los contenidos vistos en clase y otra bibliografía. Si se toman ideas de fuentes externas deben ser correctamente citadas incluyendo el correspondiente link o página de libro.
# 
# ESTE TP3 EQUIVALE A UN TERCIO DE SU NOTA FINAL.
# 
# El formato de entrega debe ser un link a un notebook de google colab. Permitir acceso a gvilcamiza.ext@fi.uba.ar y **habilitar los comentarios, para poder darles el feedback**. Si no lo hacen así no se podrá dar el feedback respectivo por cada pregunta.
# 
# El envío **se realizará en el siguiente link de google forms: [link](https://forms.gle/kscHDArwzdvrTSG99)**. Tanto los resultados, gráficas, como el código y las explicaciones deben quedar guardados y visualizables en el colab.
# 
# **NO SE VALIDARÁN ENVÍOS POR CORREO, EL MÉTODO DE ENTREGA ES SOLO POR EL FORMS.**
# 
# **Consideraciones a tener en cuenta:**
# - Se entregará 1 solo colab para este TP3.
# - Renombrar el archivo de la siguiente manera: **APELLIDO-NOMBRE-DL-TP3-Co22.ipynb**
# - Los códigos deben poder ejecutarse.
# - Los resultados, cómo el código, los gráficos y las explicaciones deben quedar guardados y visualizables en el correspondiente notebook.
# - Prestar atención a las consignas, responder las preguntas cuando corresponda.
# - Solo se revisarán los trabajos que hayan sido enviados por el forms.

# #### Librerias

# In[ ]:


import gdown
import zipfile
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report

import cv2
from PIL import Image

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# # **CLASIFICADOR DE EMOCIONES**

# El objetivo de este trabajo es construir una red neuronal convolucional (CNN) utilizando Pytorch, capaz de clasificar emociones humanas a partir de imágenes faciales. El clasificador deberá identificar una de las 7 emociones básicas: alegría, tristeza, enojo, miedo, sorpresa, disgusto y seriedad. El dataset se encuentra en este link: https://drive.google.com/file/d/1aPHE00zkDhEV1waJKhaOJMdN6-lUc0iT/view?usp=sharing

# Les recomiendo usar el siguiente código para poder obtener las imágenes fácilmente desde ese link. Pero son libres de descargar las imágenes como mejor les parezca.

# In[ ]:


url = "https://drive.google.com/uc?id=1aPHE00zkDhEV1waJKhaOJMdN6-lUc0iT"
output = "archivo.zip"

gdown.download(url, output, quiet=False)

destino = "datos_zip"
os.makedirs(destino, exist_ok=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(destino)


# In[ ]:


url = "https://drive.google.com/uc?id=1iHmMN81lHFbimo0RqXlvjzyjqOMOXrbS"
output = "prueba.zip"

gdown.download(url, output, quiet=False)

destino = "prueba"
os.makedirs(destino, exist_ok=True)

with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall(destino)


# In[ ]:


url = "https://drive.google.com/uc?id=1ug9uZq_dG9_FUlFg5HI7WnUwrRaHuCjK"
output = "modelo_completo.pth"

gdown.download(url, output, quiet=False)


# ## 1. Preprocesamiento de Datos (2 puntos)
# 
# Antes de entrenar el modelo, se debe analizar qué tipo de preprocesamiento se debe aplicar a las imágenes. Para esto, se puede considerar uno o más aspectos como:
# 
# - Tamaño
# - Relación de aspecto
# - Color o escala de grises
# - Cambio de dimensionalidad
# - Normalización
# - Balanceo de datos
# - Data augmentation
# - etc.
# 
# Sean criteriosos y elijan solo las técnicas que consideren pertinentes para este caso de uso en específico.
# 
# Recomendación: usar `torchvision.transforms` para facilitar el preprocesamiento. Lean su documentación si tienen dudas: https://docs.pytorch.org/vision/0.14/transforms.html
# 
# 

# ### Transformaciones a aplicar
# 
# #### Preprocesamiento
# 
# * **Grayscale**: Conversión a escala de grises. El color de las imagenes no es informativo para este caso y reduce la complejidad del modelo.
# 
# * **Resize**: Conversión a 48x48 pixels. Es una resolución estándar que conserva la información facial mientras reduce complejidad.
# 
# #### Data augmentation
# 
# Se aplican solo en entrenamiento y de forma aleatoria, generando variaciones de la imagen original al solicitarla.
# 
# * **RandomHorizontalFlip(p=0.5)**: Refleja la imagen horizontalmente.
# 
# * **RandomRotation(15)**: Rota la imagen dentro de un rango de ±15°.
# 
# * **ColorJitter(brightness=0.1, contrast=0.1)**: Introduce variaciones leves de iluminación.
# 
# #### Transformación a tensores
# 
# * **ToTensor()**: Convierte la imagen a tensores usados por PyTorch.
# 
# #### Normalización
# 
# * **Normalize(mean=[0.5], std=[0.5])**: Normaliza la imagen con media y desviación estándar para estabilizar el entrenamiento.

# In[ ]:


# Transformaciones para train

train_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.RandomCrop(48, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Transformaciones para test

test_transforms = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((48,48)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Carga de los datos

train_data = datasets.ImageFolder('./datos_zip/dataset_emociones/train', transform=train_transforms)
test_data  = datasets.ImageFolder('./datos_zip/dataset_emociones/validation',  transform=test_transforms)

print(f'Mapeo de clases a índices: {train_data.class_to_idx}')

# Mostrar distribución real
labels = [label for _, label in train_data.samples]
conteo = Counter(labels)
print("\nCantidad de imágenes por clase:")
for idx, count in conteo.items():
    print(f"{train_data.classes[idx]} : {count}")


# In[ ]:


# Visualización de augmentations

fig, axes = plt.subplots(1, 5, figsize=(15, 4))

# Obtiene 5 veces la misma imagen con augmentations diferentes

for i in range(5):

    img, label = train_data[500]
    img = img * 0.5 + 0.5          # Desnormaliza la imagen
    img_np = img.squeeze().numpy() # Quitar canal si es 1 solo (grayscale)

    axes[i].imshow(img_np, cmap="gray")
    axes[i].set_title(f"Aug {i+1}")
    axes[i].axis("off")

plt.suptitle(f"Label: {train_data.classes[label]}")
plt.tight_layout()
plt.show()


# ## 2. Construcción y entrenamiento del Modelo CNN (3.5 puntos)
# 
# - Construir una red neuronal convolucional desde cero, sin usar modelos pre-entrenados.
# - Analizar correctamente qué funciones de activación se deben usar en cada etapa de la red, el learning rate a utilizar, la función de costo y el optimizador.
# - Cosas como el número de capas, neuronas, número y tamaño de los kernels, entre otros, queda a criterio de ustedes, pero deben estar justificadas.

# #### Bloque convolucional
# 
# **Capa Conv2d:**
# * Aplica filtros (kernels) que extraen patrones locales de la imagen.
#     * in_c: cantidad de canales de entrada.
#     * out_c: cantidad de canales de salida.
#     * k: tamaño del kernel (3x3 por defecto).
#     * padding='same': mantiene el tamaño espacial de la imagen.
#     * bias=False: se desactiva el bias porque se utiliza BatchNorm.
# 
# **Capa BatchNorm2d**
# * Normaliza la activación de cada canal para estabilizar el entrenamiento,
#     * out_c: cantidad de canales.
# 
# **Capa ReLU:**
# * Función de activación no lineal. Evita saturación que puede producirse con tanh.

# In[ ]:


# Bloque convolucional

def conv_block(in_c, out_c, k=3):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, kernel_size=k, padding='same', bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True)
    )


# #### Modelo
# 
# Se aplican tres bloques duplicando canales y reduciendo resolución a la mitas.  
# En cada bloque se aplican dos convoluciones seguidas (la segunda convolución mejoró mucho las métricas finales de evaluación).  
# La capa de Global Pooling Average convierte cada mapa final en un solo número.  
# Dropout evita overfiting.  
# Clasificador final a las 7 clases.  

# In[ ]:


# Modelo

class CNN(nn.Module):
    def __init__(self, n_channels=1, n_outputs=7):
        super().__init__()

        # Bloque 1 – salida: 32 canales, resolución 48x48 → 24x24
        self.b1 = nn.Sequential(
            conv_block(n_channels, 32),
            conv_block(32, 32),
            nn.MaxPool2d(2)   # Reduce resolución a la mitad
        )

        # Bloque 2 – salida: 64 canales, resolución 24x24 → 12x12
        self.b2 = nn.Sequential(
            conv_block(32, 64),
            conv_block(64, 64),
            nn.MaxPool2d(2)
        )

        # Bloque 3 – salida: 128 canales, resolución 12x12 → 6x6
        self.b3 = nn.Sequential(
            conv_block(64, 128),
            conv_block(128, 128),
            nn.MaxPool2d(2)
        )

        # Global Average Pooling: convierte 128 x 6 x 6 → 128 x 1 x 1
        # Resume toda la información espacial en un vector por canal.
        self.gap = nn.AdaptiveAvgPool2d(1)

        # Dropout para reducir overfitting.
        self.dropout = nn.Dropout(0.4)

        # Capa totalmente conectada (clasificador final)
        self.fc = nn.Linear(128, n_outputs)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)

        x = self.gap(x)         # GAP → 128x1x1
        x = x.flatten(1)        # Aplana → 128
        x = self.dropout(x)     # Dropout antes del clasificador

        return self.fc(x)


model = CNN()

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)


# ## 3. Evaluación del Modelo (2.5 puntos)
# 
# El modelo entrenado debe ser evaluado utilizando las siguientes métricas:
# 
# - **Accuracy**:
#   - Reportar el valor final en el conjunto de validación.
#   - Incluir una gráfica de evolución por época para entrenamiento y validación.
# 
# - **F1 Score**:
#   - Reportar el valor final en el conjunto de validación.
#   - Incluir una gráfica de evolución por época para entrenamiento y validación.
# 
# - **Costo (Loss)**:
#   - Mostrar una gráfica de evolución del costo por época para entrenamiento y validación.
# 
# - **Classification report**
#   - Mostrar la precisión, recall y F1 score por cada clase usando `classification_report`
# 
# - **Matriz de confusión**:
#   - Mostrar la matriz de confusión absoluta (valores enteros).
#   - Mostrar la matriz de confusión normalizada (valores entre 0 y 1 por fila).
# 
# Se recomienda utilizar `scikit-learn` para calcular métricas como accuracy, F1 score, el Classification report y las matrices de confusión. Las visualizaciones pueden realizarse con `matplotlib` o `seaborn`, separando claramente los datos de entrenamiento y validación en las gráficas.
# 

# #### Data Loaders
# 
# Se emplea un sampler balanceado en entrenaiento para compensar el desbalance de las clases minoritarias (disgusto, enojo y miedo).
# 
# COMENTARIO: Esto mejoró mucho las métricas de precisión y recall de estas clases minoritarias luego del entrenamiento.

# In[ ]:


# Etiquetas como array

labels_np = np.array(labels)

# Conteo por clase

class_counts = np.bincount(labels_np)

# Pesos por clase (menos muestras = más peso)

class_weights = 1.0 / class_counts

# Peso por muestra

sample_weights = class_weights[labels_np]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

# Creación de los dataloaders

train_loader = DataLoader(
    train_data,
    batch_size=64,
    sampler=sampler,
    num_workers=2,
    pin_memory=True
)

test_loader = DataLoader(
    test_data,
    batch_size=64,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)


# #### Entrenamiento
# 
# LOAD_MODEL = True / False permite cargar el modelo ya entrenado con sus métricas o bien entrenar de cero nuevamente.
# 
# Se aplica un scheduler para reducir el tiempo de entrenamiento iniciando con un learning rate mayor e ir reduciendolo cuando no mejora, reduciendo tambien el ruido de las métricas de evaluación.
# 
# La cantidad de épocas se ajusto al punto en el que ya no se observa mejora de las métricas.

# In[ ]:


# ------------------------------------------------------
# Permite cargar un modelo ya entrenado o entrenar nuevo
# -------------------------------------------------------

LOAD_MODEL = True
PATH = "modelo_completo.pth"

# -------------------------------------------
# Carga modelo entrenado y métricas si existe
# -------------------------------------------

if LOAD_MODEL and os.path.exists(PATH):

    checkpoint = torch.load(PATH, map_location=torch.device("cpu"), weights_only=False)

    model.load_state_dict(checkpoint["model_state"])

    train_loss_list = checkpoint["train_loss"]
    val_loss_list   = checkpoint["val_loss"]
    train_acc_list  = checkpoint["train_acc"]
    val_acc_list    = checkpoint["val_acc"]
    train_f1_list   = checkpoint["train_f1"]
    val_f1_list     = checkpoint["val_f1"]
    lr_list         = checkpoint["lr"]
    class_names     = checkpoint["class_names"]
    y_true_best = checkpoint["y_true_best"]
    y_pred_best = checkpoint["y_pred_best"]
    val_loss_best = checkpoint["val_loss_best"]
    val_acc_best  = checkpoint["val_acc_best"]
    val_f1_best   = checkpoint["val_f1_best"]

    print("Modelo cargado correctamente.")

# --------------
# Entrena modelo
# --------------

else:

    print("Entrenando modelo.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # Lleva el modelo al dispositivo

    model = model.to(device)

    # Épocas de entrenamiento

    num_epochs = 50

    # Listas de métrica

    train_loss_list = []
    val_loss_list   = []
    train_acc_list  = []
    val_acc_list    = []
    train_f1_list   = []
    val_f1_list     = []
    lr_list         = []

    # Mejor modelo

    best_val_loss = float("inf")
    best_model_state = None

    # Scheduler

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        min_lr=1e-5
    )

    # Loop de épocas

    for epoch in range(num_epochs):

        # Entrenamiento

        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for X, y in train_loader:
            X, y = X.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = outputs.argmax(dim=1)

            y_true_train.append(y)
            y_pred_train.append(preds)

        y_true_train = torch.cat(y_true_train).cpu().numpy()
        y_pred_train = torch.cat(y_pred_train).cpu().numpy()

        train_loss /= len(train_loader)
        train_acc = accuracy_score(y_true_train, y_pred_train)
        train_f1  = f1_score(y_true_train, y_pred_train, average="macro")

        # Validación

        model.eval()
        val_loss = 0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)

                outputs = model(X)
                loss = criterion(outputs, y)

                val_loss += loss.item()
                preds = outputs.argmax(dim=1)

                y_true_val.append(y)
                y_pred_val.append(preds)

        y_true_val = torch.cat(y_true_val).cpu().numpy()
        y_pred_val = torch.cat(y_pred_val).cpu().numpy()

        val_loss /= len(test_loader)
        val_acc = accuracy_score(y_true_val, y_pred_val)
        val_f1  = f1_score(y_true_val, y_pred_val, average="macro")

        # Paso del scheduler

        scheduler.step(val_loss)

        # Guarda el mejor modelo

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            y_true_best = y_true_val.copy()
            y_pred_best = y_pred_val.copy()
            val_loss_best = val_loss
            val_acc_best  = val_acc
            val_f1_best   = val_f1

        # Paso del scheduler

        scheduler.step(val_loss)

        # Guarda métricas

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        train_acc_list.append(train_acc)
        val_acc_list.append(val_acc)
        train_f1_list.append(train_f1)
        val_f1_list.append(val_f1)
        lr_list.append(optimizer.param_groups[0]["lr"])

        # Log

        print(
            f"Epoch {epoch+1}/{num_epochs} | "
            f"Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f} | "
            f"Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f} | "
            f"Train F1={train_f1:.4f}, Val F1={val_f1:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.6f}"
        )

    # Restaura el mejor modelo (mínimo loss de validación)

    model.load_state_dict(best_model_state)

    print("Modelo restaurado al menor val_loss.")

    # Guarda en disco el modelo entrenado y las métricas

    class_names = train_data.classes

    torch.save({
        "model_state": model.state_dict(),
        "train_loss": train_loss_list,
        "val_loss": val_loss_list,
        "train_acc": train_acc_list,
        "val_acc": val_acc_list,
        "train_f1": train_f1_list,
        "val_f1": val_f1_list,
        "lr": lr_list,
        "class_names": train_data.classes,
        "y_true_best": y_true_best,
        "y_pred_best": y_pred_best,
        "val_loss_best": val_loss_best,
        "val_acc_best": val_acc_best,
        "val_f1_best": val_f1_best
    }, "modelo_completo.pth")

    print("Modelo y métricas guardados como modelo_completo.pth")

# Imprimir métricas del mejor modelo

print("\nMétricas globales del Modelo:")
print(f"Accuracy: {val_acc_best:.4f}")
print(f"F1 Score:  {val_f1_best:.4f}")


# #### Gráficas de evolución de métricas y pérdida

# In[ ]:


epochs = range(1, len(train_loss_list)+1)

plt.figure(figsize=(16,4))

# Accuracy

plt.subplot(1, 3, 1)
plt.plot(epochs, train_acc_list, label="Train")
plt.plot(epochs, val_acc_list, label="Val")
plt.title("Accuracy por época")
plt.xlabel("Época")
plt.ylabel("Accuracy")
plt.legend()

# F1 Score

plt.subplot(1, 3, 2)
plt.plot(epochs, train_f1_list, label="Train")
plt.plot(epochs, val_f1_list, label="Val")
plt.title("F1 Score por época (macro)")
plt.xlabel("Época")
plt.ylabel("F1 Score")
plt.legend()

# Loss

plt.subplot(1, 3, 3)
plt.plot(epochs, train_loss_list, label="Train")
plt.plot(epochs, val_loss_list, label="Val")
plt.title("Loss por época")
plt.xlabel("Época")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()


# #### Reporte de clasificación
# 
# Las peores métricas se dan en las tres clases minoritarias, aunque mejoraron mucho luego de varios ajustes del modelo (dataloader balanceado en entrenamiento y segunda capa de convolución en cada bloque).

# In[ ]:


print("Reporte clasidicación (Validación)\n")
print(classification_report(y_true_best, y_pred_best, target_names=class_names))


# #### Matrices de Confusión

# In[ ]:


print("Matrices de confusión (Validación)\n")

cm_abs = confusion_matrix(y_true_best, y_pred_best)
cm_norm = cm_abs.astype("float") / cm_abs.sum(axis=1, keepdims=True)

plt.figure(figsize=(14,5))

# Absoluta

plt.subplot(1,2,1)
sns.heatmap(cm_abs, annot=True, fmt='d', cmap="Blues")
plt.title("Matriz de Confusión (Absoluta)")
plt.xlabel("Predicción")
plt.ylabel("Verdadera")

# Normalizada

plt.subplot(1,2,2)
sns.heatmap(cm_norm, annot=True, cmap="Blues")
plt.title("Matriz de Confusión (Normalizada)")
plt.xlabel("Predicción")
plt.ylabel("Verdadera")

plt.tight_layout()
plt.show()


#  ## 4. Prueba de Imágenes Nuevas (1 punto)
# Subir al menos 10 imágenes personales de cualquier relación de aspecto (pueden usar fotos del rostro de ustedes, rostros de personas generadas por IA o imágenes stock de internet), que no formen parte del dataset de entrenamiento ni de validación.
# 
# - Debe haber al menos una imagen para cada emoción.
# 
# - Aplicar el mismo pre-procesamiento que se usó para el dataset de validation durante el entrenamiento del modelo.
# 
# - Pasar las imágenes por el modelo entrenado y mostrar:
# 
#   - La imagen original
#   - La imagen pre-procesada (mismas transformaciones del entrenamiento)
#   - El score asignado a cada clase (normalizado de 0 a 1 o de 0% a 100%)
#   - La clase ganadora inferida por el modelo
# 
# - Redactar conclusiones preliminares

# In[ ]:


def predict_image(model, img_path, class_names):
    model.eval()

    device = next(model.parameters()).device

    # Imagen original

    img_orig = Image.open(img_path).convert("RGB")

    # Preprocesamiento

    img_tensor = test_transforms(img_orig)
    img_input = img_tensor.unsqueeze(0).to(device)

    # Predicción

    with torch.no_grad():
        logits = model(img_input)
        probs = F.softmax(logits, dim=1)[0].cpu()

    winner_idx = torch.argmax(probs).item()
    winner_class = class_names[winner_idx]

    # Mostrar imágenes

    fig, axes = plt.subplots(1, 2, figsize=(6,3.5))

    axes[0].imshow(img_orig)
    axes[0].set_title("Imagen Original")
    axes[0].axis("off")

    axes[1].imshow(img_tensor.squeeze().numpy(), cmap='gray')
    axes[1].set_title("Imagen Preprocesada")
    axes[1].axis("off")

    plt.suptitle(os.path.basename(img_path))
    plt.show()

    # Mostrar probabulidades

    print("Probabilidad por clase")
    for i, cls in enumerate(class_names):
        print(f"{cls:15s}: {probs[i].item():.4f}")

    print("\nClase predicha:")
    print(f"{winner_class}  ({probs[winner_idx].item():.4f})")

    return winner_class, probs

folder = "prueba/"
file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in file_list:
    img_path = os.path.join(folder, filename)
    predict_image(model, img_path, class_names)


# Es evidente que no tener el rostro bien recortado en la imagen afecta enormemente el desempeño del modelo.

#  ## 5. Prueba de Imágenes Nuevas con Pre-procesamiento Adicional (1 punto)
# Las 10 imágenes del punto 4, ahora serán pasadas y recortadas por un algoritmo de detección de rostros. Usen el siguiente código para realizar un pre-procesamiento inicial de la imagen y ya luego aplican el pre-procesamiento que usaron al momento de entrenar el modelo.
# 
# - Pasar las imágenes por el modelo entrenado y mostrar:
#   - La imagen original
#   - La imagen recortada por el algoritmo
#   - La imagen pre-procesada (mismas transformaciones del entrenamiento)
#   - El score asignado a cada clase (normalizado de 0 a 1 o de 0% a 100%)
#   - La clase ganadora inferida por el modelo
# 
# - Comparar los resultados con el punto 4 y redactar conclusiones finales.
# 
# NOTA: Pueden adaptar el código y modificar el `scaleFactor` y el `minNeighbors` según crean conveniente para obtener mejores resultados.

# In[ ]:


def predict_image2(model, img_path, class_names):

    model.eval()
    device = next(model.parameters()).device

    # Carga imagen original

    img_orig = Image.open(img_path).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

    # Detección de rostro

    haar_path = os.path.join(
        os.path.dirname(cv2.__file__), "data", "haarcascade_frontalface_default.xml"
    )
    face_cascade = cv2.CascadeClassifier(haar_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

    det_img = img_cv.copy()
    cropped_face_rgb = None
    cropped_face_pil = None

    if len(faces) > 0:
        (x, y, w, h) = faces[0]

        # Ajustes del recorte

        shrink_ratio = 0.80     # achicar
        shift_ratio  = 0.10    # bajar hacia la pera

        side = int(min(w, h) * shrink_ratio)

        # Centro del bounding box original

        cx, cy = x + w//2, y + h//2

        # Desplazar hacia abajo

        cy = cy + int(side * shift_ratio)

        half = side // 2

        # Coordenadas finales del recorte

        x1 = max(cx - half, 0)
        y1 = max(cy - half, 0)
        x2 = min(cx + half, img_cv.shape[1])
        y2 = min(cy + half, img_cv.shape[0])

        # Dibujar sel marco

        cv2.rectangle(det_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Recorte enviado al modelo

        cropped = img_cv[y1:y2, x1:x2]
        cropped_face_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # Convertir recorte a PIL

        cropped_face_pil = Image.fromarray(cropped_face_rgb)

    else:
        raise ValueError("No se detectó ningún rostro en la imagen.")

    det_img_rgb = cv2.cvtColor(det_img, cv2.COLOR_BGR2RGB)

    # Preprocesamiento

    img_tensor = test_transforms(cropped_face_pil)
    img_input = img_tensor.unsqueeze(0).to(device)

    # Predicción

    with torch.no_grad():
        logits = model(img_input)
        probs = F.softmax(logits, dim=1)[0].cpu()

    winner_idx = torch.argmax(probs).item()
    winner_class = class_names[winner_idx]

    # Visualización

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(det_img_rgb)
    axes[0].set_title("Imagen Original + Detección")
    axes[0].axis("off")

    axes[1].imshow(cropped_face_rgb)
    axes[1].set_title("Rostro Recortado (1:1)")
    axes[1].axis("off")

    axes[2].imshow(img_tensor.squeeze().numpy(), cmap="gray")
    axes[2].set_title("Rostro Preprocesado")
    axes[2].axis("off")

    plt.suptitle(os.path.basename(img_path))
    plt.show()

    # Probabilidades

    print("Probabilidad por clase:")
    for i, cls in enumerate(class_names):
        print(f"{cls:15s}: {probs[i].item():.4f}")

    print("\nClase predicha:")
    print(f"{winner_class}  ({probs[winner_idx].item():.4f})")

    return winner_class, probs


folder = "prueba/"
file_list = [f for f in os.listdir(folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for filename in file_list:
    img_path = os.path.join(folder, filename)
    predict_image2(model, img_path, class_names)


# Se observa una gran mejora de las predicciones con los recortes de rostros.     
# Se hicieron algunos ajustes de desplazamiento y tamaño del recorte propuesto por el algoritmo de detección para mejorar los resultados.    

# ## 6. Predicción de Sentimiento en Video Promedio

# In[ ]:


import cv2
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from collections import Counter

def analizar_sentimiento_video(video_path, model, class_names, fps_extraer=1):
    """
    Extrae fotogramas de un video y utiliza la CNN para emitir el sentimiento promedio general.
    """
    model.eval()
    device = next(model.parameters()).device

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return None, None

    fps_original = cap.get(cv2.CAP_PROP_FPS)
    if fps_original == 0: fps_original = 30
    print(f"► Procesando: {video_path.split('/')[-1]} | Fotogramas extraídos p/seg: {fps_extraer}")

    frame_interval = max(1, int(math.floor(fps_original / fps_extraer)))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    fotograma_actual = 0
    predicciones_totales = []

    while True:
        ret, frame = cap.read()
        if not ret: break

        if fotograma_actual % frame_interval == 0:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6)

            if len(faces) > 0:
                (x, y, w, h) = faces[0]
                shrink_ratio, shift_ratio = 0.80, 0.10
                side = int(min(w, h) * shrink_ratio)
                cx, cy = x + w//2, y + h//2
                cy += int(side * shift_ratio)
                half = side // 2

                x1, y1 = max(cx - half, 0), max(cy - half, 0)
                x2, y2 = min(cx + half, frame.shape[1]), min(cy + half, frame.shape[0])

                cropped = frame[y1:y2, x1:x2]
                cropped_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
                cropped_pil = Image.fromarray(cropped_rgb)

                img_tensor = test_transforms(cropped_pil)
                img_input = img_tensor.unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(img_input)
                    probs = F.softmax(logits, dim=1)[0].cpu()

                winner_idx = torch.argmax(probs).item()
                predicciones_totales.append(class_names[winner_idx])

        fotograma_actual += 1

    cap.release()

    if not predicciones_totales:
        print("No se detectaron rostros con claridad en el video.")
        return None, None

    conteo = Counter(predicciones_totales)

    print(f"\n--- [ RESULTADOS DEL ANÁLISIS DEL VIDEO ] ---")
    for sentimiento, frecuencia in conteo.most_common():
        print(f" - {sentimiento}: {frecuencia} cuadros ({(frecuencia/len(predicciones_totales))*100:.1f}%)")

    sentimiento_predominante = conteo.most_common(1)[0][0]
    print(f"\n⭐⭐ SENTIMIENTO DOMINANTE EN VIDEO: {sentimiento_predominante.upper()} ⭐⭐")

    return sentimiento_predominante, conteo

# ======= EJECUCIÓN =======
# Ya detecté el documento 'videoplayback.mp4' en tu carpeta, así que lo invoco a continuación:
video = "videoplayback.mp4"
sentimiento_final, estadisticas = analizar_sentimiento_video(video, model, class_names, fps_extraer=1)

