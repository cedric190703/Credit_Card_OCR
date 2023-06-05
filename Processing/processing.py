import cv2
import numpy as np

def processing(resized):
    """Returns the image with binary colours"""
    # Convert the image in a Grayscale image
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    # Apply gaussian blur on the Grayscale image
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply the threshold on the blur image
    binary = cv2.adaptiveThreshold(blur, 255,
    cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 4)

    return binary

def find_card(binary):
        """"Returns the contours of the credit card or None otherwise"""
        # Find the contours on the binary image
        contours, _ = cv2.findContours(binary, 
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter the contours based on their area
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 1000]

        # Sort the contours by area in descending order
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        # Find the largest contour with 4 sides
        for cnt in contours:
            perimeter = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * perimeter, True)
            if len(approx) == 4:
                return approx.reshape(4, 2)

        # If no suitable contour is found, return None
        return None

def normalized_NP(contours):
    """Normalized the points of the contours for the application
    of the perspective transform"""
    rect = np.zeros((4, 2))
    rect = np.float32(rect)

    # Get the sum of the contours
    s = contours.sum(axis=1)

    # Get the min and max value of the sum of the array
    rect[0] = contours[np.argmin(s)]
    rect[2] = contours[np.argmax(s)]

    # Get the diff in the sum of the array
    diff = np.diff(contours, axis=1)

    # Get the min and max value of the diff of the sum of the original array
    rect[1] = contours[np.argmin(diff)]
    rect[3] = contours[np.argmax(diff)]
    return rect

def perspective_transform(points, image):
    """Returns the image with the perspective transform using
    the normalized points"""
    grid_size = 400

    # Represents the desired points to the corners destination
    destination_corners = np.float32([[0, 0], [grid_size, 0],
    [grid_size, grid_size], [0, grid_size]])
    
    # Ensure the source points are of type np.float32
    points = np.float32(points)
    
    normalized = normalized_NP(points)

    # Calculates the transformation matrix needed to perform the perspective transform
    transformation_matrix = cv2.getPerspectiveTransform(normalized, destination_corners)
    
    # Apply the perspective transform on the binary image
    transformed = cv2.warpPerspective(image,
    transformation_matrix, (grid_size, grid_size))

    return transformed

def main_processing(image):
    # Define the new size of the image
    new_width = 500
    new_height = 450
    new_size = (new_width, new_height)

    resized = cv2.resize(image, new_size)

    binary = processing(resized)

    cv2.imshow("result", binary)
    contours = find_card(binary)

    if(contours is not None):
        # Apply perspective transform
        rect = normalized_NP(contours)

        transformed = perspective_transform(rect, binary)
        cv2.imshow("transformed", transformed)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        return transformed
    
    return None