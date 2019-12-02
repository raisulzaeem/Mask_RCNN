import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt



class Mask:

    def __init__(self, input_mask):
        image = input_mask
        if len(input_mask.shape) == 3:
            image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)
        self.contour, _ = cv.findContours(image, 0, 3) # Get contours from mask
        epsilon = 0.02*cv.arcLength(self.contour[0], True)
        polygon_pts = cv.approxPolyDP(self.contour[0], epsilon, True)
        polygon_pts = order_points(np.squeeze(polygon_pts, axis=1))

        self.mask = image
        self.polygon_points = np.int32(polygon_pts)

    def drawContour(self, image):
        """This method draws the contour(from the mask) in the given image
        Note: The image and the mask should be of the same size"""

        cv.drawContours(image, self.contour, -1, (100, 255, 100), 10)
    
    def drawPolygon(self, image):
        """This method draws the polygon(from the mask) in the given image
        Note: The image and the mask should be of the same size"""

        cv.polylines(image, [self.polygon_points], 1, (255, 255, 0), 10)


def order_points(pts):
	# initialzie a list of coordinates that will be ordered
	# such that the first entry in the list is the top-left,
	# the second entry is the top-right, the third is the
	# bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
 
	# the top-left point will have the smallest sum, whereas
	# the bottom-right point will have the largest sum
    summe = pts.sum(axis=1)
    rect[0] = pts[np.argmin(summe)]
    rect[2] = pts[np.argmax(summe)]
 
	# now, compute the difference between the points, the
	# top-right point will have the smallest difference,
	# whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
	# return the ordered coordinates
    return rect

if __name__ == "__main__":
    
    masknames = os.listdir('masks')
    mask_dict = {}
    my_img = cv.imread('dataset/test/20190702_105926.jpg')
    my_img = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)

    for name in masknames:

        mask = cv.imread('masks/'+name, 0)
        mask_dict[name] = Mask(mask)
        mask_dict[name].drawContour(my_img)
        mask_dict[name].drawPolygon(my_img)

    plt.imshow(my_img)
    plt.show()







