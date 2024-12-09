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
from sklearn.model_selection import learning_curve  # Nuevo: Usamos learning_curve para optimizar
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
train_data = train_data.map(procesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)

# Convertir el dataset a listas de NumPy
X, y = [], []
for image, label in train_data:
    X.append(image.numpy())
    y.append(label.numpy())

X = np.vstack(X)
y = np.hstack(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Crear y entrenar el modelo Random Forest
# Antes: rf = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42)  # Cambios: Reducimos n_estimators a 100 y añadimos n_jobs=-1
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
# Antes: Hacíamos un bucle manual para calcular la curva
train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=42),  # Modelo con optimización
    X_train, y_train, train_sizes=np.linspace(0.1, 0.9, 5), cv=5, n_jobs=-1  # Cambios: Redujimos puntos de evaluación
)

# Promediar las puntuaciones
train_scores_mean = np.mean(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)

plt.figure(figsize=(8, 6))
plt.plot(train_sizes, train_scores_mean, label="Precisión en Entrenamiento", marker='o')
plt.plot(train_sizes, test_scores_mean, label="Precisión en Prueba", marker='s')
plt.title("Curva de Aprendizaje - Random Forest")
plt.xlabel("Tamaño de Entrenamiento")
plt.ylabel("Precisión")
plt.legend()
plt.grid()
plt.show()




# import os
# import tensorflow_datasets as tfds
# import numpy as np
# import tensorflow as tf
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, classification_report
# import matplotlib.pyplot as plt

# # Silenciar logs innecesarios de TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# # Ruta a la carpeta principal que contiene las subcarpetas de flores
# # folder_path = 'D:\\Descargas\\flower_photosmod\\flower_photos'
# folder_path = 'flower_photosmod'

# # Listas para almacenar las imágenes y sus etiquetas
# images = []
# labels = []

# # Recorre todas las carpetas y subcarpetas
# for root, dirs, files in os.walk(folder_path):
#     label = os.path.basename(root)  # Nombre de la subcarpeta como etiqueta
#     if label:  # Evita añadir etiquetas vacías
#         for file in files:
#             if file.endswith(('jpg', 'jpeg', 'png')):
#                 images.append(os.path.join(root, file))
#                 labels.append(label)

# # Mapear etiquetas a índices
# label_names = sorted(set(labels))
# label_to_index = {name: idx for idx, name in enumerate(label_names)}
# labels = [label_to_index[label] for label in labels]

# # Función para redimensionar las imágenes y convertirlas a escala de grises
# def procesar_imagen(file_path, label):
#     image = tf.io.read_file(file_path)
#     image = tf.image.decode_jpeg(image, channels=3)
#     # Redimensionar a un tamaño fijo (256,256)
#     image = tf.image.resize(image, (256,256))
#     # Normalizar los valores entre 0 y 1
#     image = tf.cast(image, tf.float32) / 255.0
#     # Aplanar para convertirla a un vector 1D
#     image = tf.reshape(image, (-1,))
#     return image, label

# # Crear un tf.data.Dataset
# train_data = tf.data.Dataset.from_tensor_slices((images, labels))
# train_data = train_data.map(procesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)
# train_data = train_data.batch(32).prefetch(tf.data.AUTOTUNE)

# # Convertir el dataset a listas de NumPy
# X, y = [], []
# for image, label in train_data:
#     X.append(image.numpy())
#     y.append(label.numpy())

# X = np.vstack(X)
# y = np.hstack(y)


# # Dividir los datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# # Crear y entrenar el modelo Random Forest
# rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
# rf.fit(X_train, y_train)

# # Predicción y generación de matriz de confusión
# y_pred = rf.predict(X_test)


# # Calcular y mostrar precisión
# accuracy = accuracy_score(y_test, y_pred)
# print(f"Precisión del modelo: {accuracy:.2f}")

# # Mostrar informe de clasificación
# print("\nInforme de Clasificación:")
# print(classification_report(y_test, y_pred, target_names=label_names))

# # Crear la matriz de confusion---------------------

# cm = confusion_matrix(y_test, y_pred)

# # Mostrar la matriz de confusión con etiquetas
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
# disp.plot(cmap="Reds")

# # Mostrar el gráfico
# plt.title("Matriz de Confusión - Random Forest")
# plt.show()

# # Gráfica de la curva de aprendizaje---------------------

# train_sizes= np.linspace(0.1, 0.9, 9)  # Evitamos llegar a 1.0
# train_scores= []
# test_scores= []

# for train_size in train_sizes:
#  # Selección de subconjunto de datos de entrenamiento
#  X_train_partial, _, y_train_partial, _ = train_test_split(
#     X_train, y_train, train_size=train_size, random_state=42
#  )
#  rf.fit(X_train_partial, y_train_partial)
#  # Precisión en entrenamiento y prueba
#  y_train_pred = rf.predict(X_train_partial)
#  y_test_pred = rf.predict(X_test)
#  train_scores.append(accuracy_score(y_train_partial, y_train_pred))
#  test_scores.append(accuracy_score(y_test, y_test_pred))

# plt.figure(figsize=(8, 6))
# plt.plot(train_sizes * 100, train_scores, label="Precisión en Entrenamiento", marker='o')
# plt.plot(train_sizes * 100, test_scores, label="Precisión en Prueba", marker='s')
# plt.title("Curva de Aprendizaje - Random Forest")
# plt.xlabel("Tamaño de Entrenamiento (%)")
# plt.ylabel("Precisión")
# plt.legend()
# plt.grid()
# plt.show()