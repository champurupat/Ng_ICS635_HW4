# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# %%
fashion_mnist = tf.keras.datasets.fashion_mnist

# %%
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# %%
# normalize images to the [0,1] range
train_images_normalized = train_images / 255.0
test_images_normalized = test_images / 255.0

# %%
# organize into typical variable names
X_train = train_images_normalized
y_train = train_labels
X_test = test_images_normalized
y_test = test_labels

# %%
# split original trainingset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# %%
# class names that map to labels
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# %%
# basic neural network, reference: https://www.tensorflow.org/tutorials/keras/classification
model_baseline = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
model_baseline.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_baseline = model_baseline.fit(X_train, y_train, epochs=10, 
                    validation_data=(X_val, y_val))

# %%
# CNN model, adapted from Lecture 19 slides for 28x28x1 input images
model_cnn1 = models.Sequential(
    [       
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)
model_cnn1.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_cnn1 = model_cnn1.fit(X_train, y_train, epochs=10, 
                              validation_data=(X_val, y_val))

# %%
model_cnn2 = models.Sequential(
    [       
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10)
    ]
)
model_cnn2.compile(optimizer='sgd',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history_cnn2 = model_cnn2.fit(X_train, y_train, epochs=10, 
                              validation_data=(X_val, y_val))

# %% Plot training histories
plt.figure(figsize=(15, 10))

# Loss curves
plt.subplot(3, 2, 1)
plt.plot(history_baseline.history['loss'], label='Training Loss')
plt.plot(history_baseline.history['val_loss'], label='Validation Loss')
plt.title('Baseline Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(history_cnn1.history['loss'], label='Training Loss')
plt.plot(history_cnn1.history['val_loss'], label='Validation Loss')
plt.title('CNN1 (Adam) Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(3, 2, 5)
plt.plot(history_cnn2.history['loss'], label='Training Loss')
plt.plot(history_cnn2.history['val_loss'], label='Validation Loss')
plt.title('CNN2 (SGD) Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy curves
plt.subplot(3, 2, 2)
plt.plot(history_baseline.history['accuracy'], label='Training Accuracy')
plt.plot(history_baseline.history['val_accuracy'], label='Validation Accuracy')
plt.title('Baseline Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(history_cnn1.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn1.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN1 (Adam) Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(3, 2, 6)
plt.plot(history_cnn2.history['accuracy'], label='Training Accuracy')
plt.plot(history_cnn2.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN2 (SGD) Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# %% Evaluate models on test set
import numpy as np

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    # Add value annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.show()
    return accuracy_score(y_true, y_pred)

# Reshape test data for CNN models
X_test_reshaped = X_test.reshape(-1, 28, 28, 1)

# Get predictions
baseline_pred = np.argmax(model_baseline.predict(X_test), axis=1)
cnn1_pred = np.argmax(model_cnn1.predict(X_test_reshaped), axis=1)
cnn2_pred = np.argmax(model_cnn2.predict(X_test_reshaped), axis=1)

# Plot confusion matrices and print accuracies
print("Baseline Model Test Accuracy:", 
      plot_confusion_matrix(y_test, baseline_pred, "Baseline Model Confusion Matrix"))
print("\nCNN1 (Adam) Model Test Accuracy:", 
      plot_confusion_matrix(y_test, cnn1_pred, "CNN1 (Adam) Model Confusion Matrix"))
print("\nCNN2 (SGD) Model Test Accuracy:", 
      plot_confusion_matrix(y_test, cnn2_pred, "CNN2 (SGD) Model Confusion Matrix"))