In this part we clean the image and also find the credit card and apply the perspective transform using Numpy and openCV2.

### The processing part includes different parts :
1-Resize the image to have a smaller image.


2-Get the grayscale image.
<img src="images/gray1.png" alt="Grayscale">


3-Apply blur filter on the image.
<img src="images/blur1.png" alt="Blur">


4-Apply a threshold filter on the image to have an image with binary colors.
<img src="images/binary.png" alt="Binary">


5-Get the contours of the credit card.


6-Normalize the contours and get the perspective transform.
<img src="images/trasnformed1.png" alt="Perspective transform">


7-Get the opening process on the image to have the main patterns of the digits.
<img src="images/opening.png" alt="Opening">

8-Crop the image to have the groups of numbers.
<img src="images/trasnformed.png" alt="Perspective transform">


9-Get the contours of the 4 digits in each group.


10-Get each digit image in a list and return the list.
<img src="images/digit.png" alt="Digit">