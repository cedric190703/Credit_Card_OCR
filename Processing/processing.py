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
    grid_size_height = 350
    grid_size_weight = 225

    # Represents the desired points to the corners destination
    destination_corners = np.float32([[0, 0], [grid_size_height, 0],
    [grid_size_height, grid_size_weight], [0, grid_size_weight]])
    
    # Ensure the source points are of type np.float32
    points = np.float32(points)
    
    normalized = normalized_NP(points)

    # Calculates the transformation matrix needed to perform the perspective transform
    transformation_matrix = cv2.getPerspectiveTransform(normalized, destination_corners)
    
    # Apply the perspective transform on the binary image
    transformed = cv2.warpPerspective(image,
    transformation_matrix, (grid_size_height, grid_size_weight))

    return transformed

def opening_process(gray):
    """Apply filters to have digits group patterns"""
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))

    # Apply errosion and dilatation
    opening = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)

    # Compute the gradient
    gradX = cv2.Sobel(opening, ddepth=cv2.CV_32F, dx=1, dy=0,
	ksize=-1)
    
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
    gradX = gradX.astype("uint8")

    # Errode the contours in the image
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)

    thresh = cv2.threshold(gradX, 0, 255,
    cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    return thresh

def get_digits(areas, transformed):
    """"Returns images of all the numbers in the areas"""
    digits_list = []

    # Get each digits contours in the area
    for (x, y, w, h) in areas:

        # Group of digits
        roi = transformed[y:y+h, x:x+w]

        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        _, thresh = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        contours_digit, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get each digit
        for ctn in contours_digit:
            (x, y, w, h) = cv2.boundingRect(ctn)
            digit = thresh[y-2:y + h+2, x-2:x + w+2]

            digits_list.append(digit)
    
    return digits_list

def perspective_filter(transformed):
    """Apply filters to detect numbers in the trasnformed image
    and returns the groups of digits in the credit card"""
    gray = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)

    thresh = opening_process(gray)

    contours, _ = cv2.findContours(thresh,
    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = []

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        area = w / h

        if 2 <= area <= 4 and 40 < w < 70 and 12 < h < 25:
                areas.append((x-5, y-5, w+10, h+10))
    
    return areas

def main_processing(image):
    # Define the new size of the image
    new_width = 500
    new_height = 450
    new_size = (new_width, new_height)

    resized = cv2.resize(image, new_size)
    
    binary = processing(resized)

    contours = find_card(binary)

    if(contours is not None):
        # Apply perspective transform
        rect = normalized_NP(contours)

        transformed = perspective_transform(rect, resized)

        areas = perspective_filter(transformed)

        digits_images = get_digits(areas, transformed)

        digits = digits_images if len(digits_images) == 16 else None
        cv2.destroyAllWindows()

        return digits
        
    return None