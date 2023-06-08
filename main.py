import sys
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the paths to use the main functions
sys.path.append('./Processing')

from processing import main_processing

def main():
    image = cv2.imread(r'Image_Cards/f.jpg')

    # Step 1 : Image processing
    digits = main_processing(image)

    if(digits is None):
        return
    
    card_number = []

    # Load the CNN model
    model = tf.keras.models.load_model('model.h5')

    # Step 2 : Recognizes the numbers using the CNN model
    for digit in digits:
        # Resize the image to 28 x 28 pixels
        image = cv2.resize(digit, (28, 28))

        # Expand dimensions to match the expected shape of the model
        image = np.expand_dims(image, axis=-1)

        # Normalize the pixel values to range between 0 and 1
        image = image / 255.0

        # Reshape the image to match the expected input shape of the model
        image = np.reshape(image, (1, 28, 28, 1))

        # Make the prediction using the model
        prediction = np.argmax(model.predict(image))
        
        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

        card_number.append(prediction)

    # Step 3 : Credit card analysis

if __name__ == "__main__":
    main()