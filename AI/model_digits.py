import tensorflow as tf
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont
import random
from tensorflow.keras import layers
import matplotlib.pyplot as plt

def generate_dataset(font_family_folder, min_size, max_size, digits, num_samples):
    dataset = []

    font_files = os.listdir(font_family_folder)
    for _ in range(num_samples):
        font_file = font_files[0]
        font_path = os.path.join(font_family_folder, font_file)
        font_size = random.randint(min_size, max_size)
        digit = random.choice(digits)

        # Create a blank white image with a black background
        image = Image.new("RGB", (50, 50), "Black")
        draw = ImageDraw.Draw(image)

        # Set the font properties
        font = ImageFont.truetype(font_path, font_size)

        # Calculate the position to center the digit in the image
        digit_bbox = draw.textbbox((0, 0), str(digit), font=font)
        digit_width = digit_bbox[2] - digit_bbox[0]
        digit_height = digit_bbox[3] - digit_bbox[1]
        x = (image.width - digit_width) // 2
        y = (image.height - digit_height) // 2

        # Draw the digit on the image
        draw.text((x, y), str(digit), font=font, fill="white")

        # Resize the image to 28 x 28 pixels
        image = image.resize((28, 28))

        # Convert the image to grayscale
        image = image.convert("L")

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Normalize the pixel values to range between 0 and 1
        image_array = image_array / 255.0

        noisy_image = image_array.copy()

        # Get the white pixels
        white_pixels = noisy_image == 1.0

        # Add random black groups in random locations
        num_groups = random.randint(0, 2)  # Number of black groups to add
        for _ in range(num_groups):
            group_size = random.randint(2, 8)  # Size of each black group
            group_x = random.randint(0, 28 - group_size)
            group_y = random.randint(0, 28 - group_size)
            noisy_image[group_x : group_x + group_size, group_y : group_y + group_size][white_pixels[group_x : group_x + group_size, group_y : group_y + group_size]] = 0.0

        # Clip the pixel values to the range of 0.0 to 1.0
        noisy_image = np.clip(noisy_image, 0.0, 1.0)

        # Add the noisy image and label to the dataset
        dataset.append((noisy_image, digit))

    return dataset

def create_model(input_shape, num_classes):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(16, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(120, activation='relu'))
    model.add(layers.Dense(84, activation='relu'))
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

def main_ai():
    font_families_folder = "AI/Fonts"
    font_size_min = 54
    font_size_max = 62
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    num_samples = 75000

    dataset = generate_dataset(font_families_folder, font_size_min, font_size_max, digits, num_samples)

    plt.imshow(dataset[0][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[6][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[6544][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[4548][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[8787][0], cmap='gray')
    plt.show()
    
    plt.imshow(dataset[1000][0], cmap='gray')
    plt.show()
    
    plt.imshow(dataset[3300][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[5454][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[877][0], cmap='gray')
    plt.show()

    plt.imshow(dataset[2424][0], cmap='gray')
    plt.show()

    # Split the dataset into training and testing sets
    train_size = int(0.8 * len(dataset))
    train_data = dataset[:train_size]
    test_data = dataset[train_size:]

    # Prepare the training data
    train_images = np.array([data[0] for data in train_data])
    train_labels = np.array([data[1] for data in train_data])

    # Prepare the testing data
    test_images = np.array([data[0] for data in test_data])
    test_labels = np.array([data[1] for data in test_data])

    # Reshape the images to match the expected input shape of the CNN
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

    # Normalize the pixel values to range between 0 and 1
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0

    # Define the input shape and the number of classes
    input_shape = train_images[0].shape
    num_classes = len(digits)

    # Create the CNN model
    model = create_model(input_shape, num_classes)

    # Compile the model
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])

    # Train the model
    model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_images , test_labels)
    print(f'Test loss : {test_loss}')
    print(f'Test accuracy: {test_acc}')

    model.save('model.h5')

main_ai()