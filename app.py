import cv2
import re
import numpy as np
import pytesseract
import json
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'

# Preprocessing part ->
# Load the image
img = cv2.imread(r'C:\Users\cbrzy\OneDrive\Bureau\creditCard.jpg')

# Define the new size of the image
new_width = 500
new_height = 450
new_size = (new_width, new_height)

# Resize the image
resized = cv2.resize(img, new_size)

# Get the copy of the resized image to crop it without drawings
image_copy = resized.copy()

# Use this image to draw all the squares
allRectangles = resized.copy()

# Use the cvtColor() function to grayscale the image
gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

# Apply a blur filter to remove noise
blurred = cv2.GaussianBlur(gray,(3,3),0)

# Use the bilateral filter to have a better image
bilateral = cv2.bilateralFilter(gray, 11, 17, 17)

# Use erosion and dilation to increase the white shape in the image
kernel = np.ones((5,5),np.uint8)
erosion = cv2.erode(gray,kernel,iterations = 2)
kernel = np.ones((4,4),np.uint8)
dilation = cv2.dilate(erosion,kernel,iterations = 2)

# Use canny filter for having the contours of the image
canny = cv2.Canny(dilation,100,300,apertureSize = 3)

# Get the contours
contours,hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# To have the largest contour
largestArea = 0
largestContour = 0

# Iterate through each contour
for cnt in contours:
    # Find the convex hull of the contour
    hull = cv2.convexHull(cnt)
    # Simplify the contour using approxPolyDP
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    
    # Calculate the length of each line segment in the contour
    lines = cv2.HoughLinesP(canny, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
    lengths = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)
    
    # Sort the line segments in descending order of length
    sorted_lines = [line for _, line in sorted(zip(lengths, lines), reverse=True)]
    
    # Select the four longest line segments
    longest_lines = sorted_lines[:4]
    
    # Compute the intersections between the selected line segments to obtain the four corners of the quadrilateral
    points = []
    for i in range(4):
        for j in range(i+1, 4):
            x1, y1, x2, y2 = longest_lines[i][0]
            x3, y3, x4, y4 = longest_lines[j][0]
            det = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
            if det != 0:
                px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / det
                py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / det
                points.append((px, py))
    
    # Draw the quadrilateral on the image
    pts = np.array(points, np.int32)

    # Compute the bounding box of the quadrilateral
    x, y, w, h = cv2.boundingRect(pts)

    # Draw the rectangle on the image
    cv2.rectangle(allRectangles, (x, y), (x+w, y+h), (0, 255, 0), 2)

    areaRectangle = w*h
    if(areaRectangle > largestArea):
        largestArea = areaRectangle
        largestContour = cnt

# Draw the largest rectangle on the image
x, y, w, h = cv2.boundingRect(largestContour)
cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Draw the rectangle on the image
cv2.rectangle(resized, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Desired points value from the largest rectangle
p1 = [x,y]
p2 = [x+w,y]
p3 = [x,y+h]
p4 = [x+w,y+h]

# Create point matrix
point_matrix = np.float32([p1,p2,p3,p4])

width = 450
height = 305

# Desired points value in output images
converted_p1 = [0,0]
converted_p2 = [width,0]
converted_p3 = [0,height]
converted_p4 = [width,height]

# Convert points
converted_points = np.float32([converted_p1,converted_p2,
                               converted_p3,converted_p4])

# perspective transform
perspective_transform = cv2.getPerspectiveTransform(point_matrix,converted_points)
warped = cv2.warpPerspective(image_copy,perspective_transform,(width,height))

# apply the bilateral filter with diameter 9, sigmaColor 75 and sigmaSpace 75
bilateral = cv2.bilateralFilter(warped, 9, 75, 75)

# apply a thresholding filter to the filtered image
gray_img = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
result = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# Recognition part with Tesseract ->
# Perform OCR using Tesseract
text = pytesseract.image_to_string(result)

# Split the text into lines for the regex part
lines = text.splitlines()

# Extract data part with regex ->
# Regex to get the range of 16 numbers for the code
regex_code = r"\b\d{16}\b"

code = None
for line in lines:
    line = line.replace(" ", "")
    if(re.match(regex_code,line)):
        code = line
        break

# Convert the code in integer for the fraud part detection
code = int(code)

# Try to get the user name without a null value
# Check also a no number value and different from the first line for the bank name
last_idx = len(lines)-1
user_name = None
for i in range(last_idx, 1, -1):
    if(lines[i] != "" and not any(char.isnumeric() for char in lines[i])):
        user_name = lines[i]

# Create a data for all information about the credit card
data = {
    'bank name': lines[0] if len(lines) > 0 else None,
    'code': code,
    'user name': user_name    
}

# Create a JSON file to put all information in it
file_output = "data.json"

# Open the file in write mode
with open(file_output, "w") as f:
    # Use the json.dump() function to write the data to the file
    json.dump(data, f)

# Fraud detection part ->
