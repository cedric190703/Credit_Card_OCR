import sys
import cv2
import pytesseract

# Import the paths to use the main functions
sys.path.append('./Processing')

from processing import main_processing

def main():
    image = cv2.imread(r'C:\Users\cbrzy\OneDrive\Bureau\creditCard.jpg')

    # Step 1 : Image processing
    transformed = main_processing(image)

    if(transformed is None):
        return
    
    # Step 2 : Recognizes the numbers
    
    # Step 3 : Extraction of digits

    # Step 4 : Credit card analysis

if __name__ == "__main__":
    main()