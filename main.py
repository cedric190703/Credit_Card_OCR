import sys
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Import the paths to use the main functions
sys.path.append('./Processing')

from processing import main_processing

def main():
    image = cv2.imread(r'Image_Cards/c3.jpg')

    # Step 1 : Image processing
    digits = main_processing(image)

    if(digits is None):
        return
    
    card_number = []

    # Load the CNN model
    model = tf.keras.models.load_model('model.h5')

    # Step 2 : Recognizes the numbers using the CNN model
    for digit in digits:
        # Step 3: Preprocess the digit image
        #digit = preprocess_digit(digit)

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
        print(prediction)

        plt.imshow(image.reshape(28, 28), cmap='gray')
        plt.show()

        # Step 11: Add the predicted digit to the card number
        card_number.append(prediction)

    card_number.reverse()

    cpt = 0
    real_data = [4, 8, 4, 1, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4]

    print(len(real_data), "   ", len(card_number))
    for i in range(len(card_number)):
        if(real_data[i] != card_number[i]):
            cpt+=1
    
    print(card_number)
    print(cpt)

    # Step 3 : Extraction of digits


    # Step 4 : Credit card analysis

if __name__ == "__main__":
    main()