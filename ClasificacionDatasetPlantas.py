# # CLASIFICACIÓN DE IMÁGENES EN UN DATASET DE PLANTAS

# # INTEGRANTES:
# # DIEGO ARTEAGA MENDOZA
# # PATRICIA GONZALEZ LARENAS
# # SAMUEL PEREZ MANZANEDA

# # MÉTODO A UTILIZAR: RANDOM FOREST
# # DATASET A UTILIZAR: FLOWER 27

import os
import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

# Silenciar logs innecesarios de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Ruta a la carpeta principal que contiene las subcarpetas de flores
folder_path = 'flower_photosmod'

# Listas para almacenar las imágenes y sus etiquetas
images = []
labels = []

# Recorre todas las carpetas y subcarpetas
for root, dirs, files in os.walk(folder_path):
    label = os.path.basename(root)  # Nombre de la subcarpeta como etiqueta
    if label:  # Evita añadir etiquetas vacías
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                images.append(os.path.join(root, file))
                labels.append(label)

# Mapear etiquetas a índices
label_names = sorted(set(labels))
label_to_index = {name: idx for idx, name in enumerate(label_names)}
labels = [label_to_index[label] for label in labels]

# Función para redimensionar las imágenes y convertirlas a escala de grises


def procesar_imagen(file_path, label):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    # Redimensionar a un tamaño fijo (256,256)
    image = tf.image.resize(image, (256, 256))
    # Normalizar los valores entre 0 y 1
    image = tf.cast(image, tf.float32) / 255.0
    # Aplanar para convertirla a un vector 1D
    image = tf.reshape(image, (-1,))
    return image, label


# Crear un tf.data.Dataset
train_data = tf.data.Dataset.from_tensor_slices((images, labels))
train_data = train_data.map(
    procesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)

# Convertir el dataset a listas de NumPy
X, y = [], []
for image, label in train_data:
    X.append(image.numpy())
    y.append(label.numpy())

X = np.vstack(X)
y = np.hstack(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.4, random_state=42)

# Crear y entrenar el modelo Random Forest
rf = RandomForestClassifier(
    n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Predicción y generación de matriz de confusión
y_pred = rf.predict(X_test)

# Calcular y mostrar precisión
accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión del modelo: {accuracy:.2f}")

# Mostrar informe de clasificación
print("\nInforme de Clasificación:")
print(classification_report(y_test, y_pred, target_names=label_names))

# Crear la matriz de confusión
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión con etiquetas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot(cmap="Reds")

# Mostrar el gráfico
plt.title("Matriz de Confusión - Random Forest")
plt.show()

# Gráfica de la curva de aprendizaje usando `learning_curve`
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=20,
                           n_jobs=-1, random_state=42),  # Modelo con optimización
    # Cambios: Redujimos puntos de evaluación
    X_train, y_train, train_sizes=np.linspace(0.1, 0.9, 5), cv=5, n_jobs=-1
)

# Promediar las puntuaciones
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean,
         label="Precisión en Entrenamiento", marker='o')
plt.plot(train_sizes, test_scores_mean,
         label="Precisión en Prueba", marker='s')
plt.title("Curva de Aprendizaje - Random Forest")
plt.xlabel("Tamaño de Entrenamiento")
plt.ylabel("Precisión")
plt.legend()
plt.grid()
plt.show()


# Crear gráficos por cada tipo de flor
for idx, flor in enumerate(label_names):
    # Filtrar las muestras correspondientes a la flor actual
    # Índices donde la clase real es la flor actual
    indices = np.where(y_test == idx)
    X_flor = X_test[indices]  # Características de esta flor
    y_flor_real = y_test[indices]  # Etiquetas reales
    y_flor_pred = y_pred[indices]  # Predicciones del modelo

    # Calcular la precisión para esta flor
    accuracy = accuracy_score(y_flor_real, y_flor_pred)

    # Diagrama de dispersión para la flor actual
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(
        X_flor[:, 0], X_flor[:, 1],  # Usamos las dos primeras características
        c=y_flor_pred, cmap='viridis', alpha=0.6, edgecolor='k'
    )

    # Añadir una barra de colores que muestre los índices y nombres de las clases
    cbar = plt.colorbar(scatter)
    # Colocar ticks en la barra para cada índice
    cbar.set_ticks(range(len(label_names)))
    # Etiquetar los ticks con los nombres de las flores
    cbar.set_ticklabels(label_names)
    cbar.set_label("Clase Predicha")

    plt.title(f"Diagrama de Dispersión: {flor} (Precisión: {accuracy:.2f})")
    plt.xlabel("Primera característica")
    plt.ylabel("Segunda característica")
    plt.grid(alpha=0.3)
    plt.show()
