In this part we clean the image and also find the credit card and apply the perspective transform using Numpy and openCV2.

### The processing part includes different parts :

1-Resize the image to have a smaller image.


2-Get the grayscale image.
<img src="images/gray1.png" alt="Grayscale" width="500" height="300">


3-Apply blur filter on the image.
<img src="images/blur1.png" alt="Blur" width="500" height="300">


4-Apply a threshold filter on the image to have an image with binary colors.
<img src="images/binary.png" alt="Binary" width="500" height="300">


5-Get the contours of the credit card.


6-Normalize the contours and get the perspective transform.
<img src="images/transformed1.png" alt="Perspective transform" width="500" height="300">


7-Get the opening process on the image to have the main patterns of the digits.
<img src="images/opening.png" alt="Opening" width="500" height="300">


8-Crop the image to have the groups of numbers.
<img src="images/transformed.png" alt="Perspective transform" width="500" height="300">


9-Get the contours of the 4 digits in each group.


10-Get each digit image in a list and return the list.
<img src="images/digit.png" alt="Digit">