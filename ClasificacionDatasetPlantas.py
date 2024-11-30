# # CLASIFICACIÓN DE IMÁGENES EN UN DATASET DE PLANTAS

# # INTEGRANTES:
# # DIEGO ARTEAGA MENDOZA
# # PATRICIA GONZALEZ LARENAS
# # SAMUEL PEREZ MANZANEDA

# # MÉTODO A UTILIZAR: RANDOM FOREST
# # DATASET A UTILIZAR: FLOWER 27

# import os
# import tensorflow_datasets as tfds
# import numpy as np
# import tensorflow as tf
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# import matplotlib.pyplot as plt

# # Silenciar logs innecesarios de TensorFlow
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # Carga del dataset
# dataset, dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
# train_data = dataset['train']

# # Función para redimensionar las imágenes y convertirlas a escala de grises
# def procesar_imagen(image, label):
#     # Redimensionar a un tamaño fijo (128x128)
#     image = tf.image.resize(image, (128, 128))
#     # Convertir a escala de grises
#     image = tf.image.rgb_to_grayscale(image)
#     # Normalizar los valores entre 0 y 1
#     image = tf.cast(image, tf.float32) / 255.0
#     # Aplanar para convertirla a un vector 1D
#     image = tf.reshape(image, (-1,))
#     return image, label

# # Aplicar la función de preprocesamiento al dataset
# train_data = train_data.map(procesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)

# # Convertir el dataset a listas de NumPy
# X, y = [], []
# for image, label in tfds.as_numpy(train_data):
#     X.append(image)
#     y.append(label)

# X = np.array(X)
# y = np.array(y)

# # Dividir los datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Crear y entrenar el modelo Random Forest
# rf = RandomForestClassifier(n_estimators=100, random_state=42)
# rf.fit(X_train, y_train)

# # Predicción y generación de matriz de confusión
# y_pred = rf.predict(X_test)
# cm = confusion_matrix(y_test, y_pred)

# # Mostrar la matriz de confusión
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(dataset_info.features['label'].num_classes))
# disp.plot()

# # Mostrar el gráfico
# plt.title("Matriz de Confusión - Random Forest")
# plt.show()


import os
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Silenciar logs innecesarios de TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Carga del dataset
dataset, dataset_info = tfds.load('tf_flowers', with_info=True, as_supervised=True)
train_data = dataset['train']

# Obtener nombres de las etiquetas
label_names = dataset_info.features['label'].names

# Función para redimensionar las imágenes y convertirlas a escala de grises
def procesar_imagen(image, label):
    # Redimensionar a un tamaño fijo (128x128)
    image = tf.image.resize(image, (128, 128))
    # Convertir a escala de grises
    image = tf.image.rgb_to_grayscale(image)
    # Normalizar los valores entre 0 y 1
    image = tf.cast(image, tf.float32) / 255.0
    # Aplanar para convertirla a un vector 1D
    image = tf.reshape(image, (-1,))
    return image, label

# Aplicar la función de preprocesamiento al dataset
train_data = train_data.map(procesar_imagen, num_parallel_calls=tf.data.AUTOTUNE)

# Convertir el dataset a listas de NumPy
X, y = [], []
for image, label in tfds.as_numpy(train_data):
    X.append(image)
    y.append(label)

X = np.array(X)
y = np.array(y)

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predicción y generación de matriz de confusión
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Mostrar la matriz de confusión con etiquetas
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_names)
disp.plot()

# Mostrar el gráfico
plt.title("Matriz de Confusión - Random Forest")
plt.show()
