from keras import Sequential
from keras.api.layers import Dense, BatchNormalization, Flatten, MaxPool2D, Conv2D, LeakyReLU, Dropout
from keras.api.optimizers import Adam
from keras.api.losses import categorical_crossentropy
from keras.api.utils import to_categorical
from keras.api.layers import RandomZoom
import tensorflow_datasets as tfds
import numpy as np

# Load the MNIST dataset using TensorFlow Datasets
dataset, info = tfds.load('mnist', with_info=True)
train_data, test_data = dataset['train'], dataset['test']

# Convert the data to NumPy arrays
x_train = np.array([example['image'].numpy() for example in train_data])
y_train = np.array([example['label'].numpy() for example in train_data])
x_test = np.array([example['image'].numpy() for example in test_data])
y_test = np.array([example['label'].numpy() for example in test_data])

# plt.imshow(x_train[1000])
# plt.show()

# Normalize and preprocess the data
x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
x_test = x_test.reshape(-1, 28, 28, 1) / 255.0


# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

# Print label distribution for the training data
unique, counts = np.unique(y_train, return_counts=True)
print("Label Distribution:")
for label, count in zip(unique, counts):
    print(f"Class {label}: {count} samples")


# Define the CNN model
cnn_model = Sequential([
    RandomZoom(height_factor=(-0.5, 0.5), width_factor=(-0.5, 0.5)),
    Conv2D(32, (3, 3), input_shape=(28, 28, 1)),
    LeakyReLU(alpha=0.1),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.25),
    Conv2D(64, (3, 3)),
    LeakyReLU(alpha=0.1),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.25),
    Conv2D(128, (3, 3)),  # Added a third convolutional layer
    LeakyReLU(alpha=0.1),
    MaxPool2D(),
    BatchNormalization(),
    Dropout(0.3),
    Flatten(),
    Dense(128),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.5),
    Dense(64),
    LeakyReLU(alpha=0.1),
    BatchNormalization(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

# Compile the model
cnn_model.compile(optimizer=Adam(learning_rate=0.001), loss=categorical_crossentropy, metrics=['accuracy'])

# Train the model using augmented data
cnn_model.fit(x_train, y_train, epochs=10,validation_data=(x_test,y_test))

# Save the model
cnn_model.save('modelv3.h5')
