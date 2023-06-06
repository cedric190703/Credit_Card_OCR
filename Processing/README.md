In this part we clean the image and also find the credit card and apply the perspective transform using Numpy and openCV2.

### The processing part includes different parts :

1-Resize the image to have a smaller image.


2-Get the grayscale image.
<p align="center">
<img src="images/gray1.png" alt="Grayscale" width="500" height="350">
</p>

3-Apply blur filter on the image.
<p align="center">
<img src="images/blur1.png" alt="Blur" width="500" height="350">
</p>

4-Apply a threshold filter on the image to have an image with binary colors.
<p align="center">
<img src="images/binary.png" alt="Binary" width="500" height="350">
</p>

5-Get the contours of the credit card.


6-Normalize the contours and get the perspective transform.
<p align="center">
<img src="images/transformed1.png" alt="Perspective transform" width="500" height="350">
</p>

7-Get the opening process on the image to have the main patterns of the digits.
<p align="center">
<img src="images/opening.png" alt="Opening" width="500" height="350">
</p>

8-Crop the image to have the groups of numbers.
<p align="center">
<img src="images/transformed.png" alt="Perspective transform" width="500" height="350">
</p>

9-Get the contours of the 4 digits in each group.


10-Get each digit image in a list and return the list.
<p align="center">
<img src="images/digit.png" alt="Digit" width="200" height="125">
</p>