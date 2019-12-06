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
        cv.polylines(image, [self.polygon_points], 1, (255, 255, 0), 3)

    def crop_image(self, image):

        x_points = self.polygon_points[:, 0]
        y_points = self.polygon_points[:, 1]

        col_max = max(x_points)
        row_max = max(y_points)

        col_min = min(x_points)
        row_min = min(y_points)

        offset = int((row_max - row_min)/10)

        col_max += offset
        row_max += offset
        self.col_min = max(0, col_min-offset)   # if col_min/row_min < 0, cropping error.
        self.row_min = max(0, row_min -offset)  # To uncrop the images in crop2original_coord method

        cropped_image = image[self.row_min:row_max, self.col_min:col_max, :]
        # copy_image = cropped_image.copy() >>> When necessary!
        return cropped_image


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


def hough_line_transformation(image):
# Create a CLAHE object for histogram equalization. 
#(CLAHE: Contrast Limited Adaptive Histogram Equalization)"""

    if len(image.shape) == 3:
        image = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

    clahe = cv.createCLAHE(clipLimit=1.0)
    image_he = clahe.apply(image)


    # We haven't set a constant kernel size. It depends on the size of the image.
    # In our case we got better result when we divided the no of pixels by 150.

    k = int((max(image.shape)/150))

    # Kernel size should be an odd integer

    if k <= 1:
        kernel_size = 3
        k = 1
    elif  k%2 == 1:
        kernel_size = k
    else:
        kernel_size = k+1

    # image_blur = cv.medianBlur(image_he,kernel_size)

    # Applying Binary inverse adaptive threshold, to keep the forground in white and background in black. 
    # Here we also used the kernel size from the size of the image

    image_thresh = cv.adaptiveThreshold(image_he, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, kernel_size, 4)

    # Now we will use Hough Transformation to get the lines in the image in 'Hesse Normal Form' (rho,theta).
    # Output vector of lines. Each line is represented by a 2 or 3 element vector (ρ,θ) or (ρ,θ,votes) . 
    # ρ is the distance from the coordinate origin (0,0) (top-left corner of the image). θ is the line rotation angle in radians.

    hough_lines = cv.HoughLines(image_thresh,1,np.pi/180,k*70)

    return hough_lines,image_thresh

def hough_lines_show(hough_lines, img):

    for line in hough_lines:
        rho, theta = line[0]
        # Drawing the lines in the image: 
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 2000*(-b))
        y1 = int(y0 + 2000*(a))
        x2 = int(x0 - 2000*(-b))
        y2 = int(y0 - 2000*(a))
        cv.line(img,(x1,y1),(x2,y2),(0,0,255),2)
    plt.imshow(img)
    plt.show()


def points2parametric(points):
    """ Takes end points of lines and calculates the
    parametric representation(ρ,θ) of each line.
    Points = array([[x1,y1],[x2,y2],[x3,y3],[x4,y4]])
    Returns [[ρ1,θ1],[ρ2,θ2],[ρ3,θ3],[ρ4,θ4]]
    """
    points_ext = np.append(points, [points[0]], axis=0)
    rho_theta = []


    for i in range(len(points_ext)-1):
        x1 = points_ext[i][0]
        y1 = points_ext[i][1]
        x2 = points_ext[i+1][0]
        y2 = points_ext[i+1][1]

        if y1 == y2:
            theta = np.pi/2
        else:
            theta = np.arctan((x2-x1)/(y1-y2))
        
        rho = x1 * np.cos(theta) + y1 * np.sin(theta)
        # We want our Rho to be positive. 
        # Theta is in the range from >> 0 to 2*pi <<
        if theta < 0:
            theta = np.pi + theta
            rho = -rho
        
        if rho < 0:
            theta = np.pi + theta
            rho = -rho
        
        rho_theta.append([rho, theta])
    
    rho_theta = np.array(rho_theta)[:, np.newaxis]
    
    return rho_theta

def scale_rho_theta(houghlines):
    
    rho_theta = np.array([[[-rho, theta+np.pi]] if rho < 0 else 
                         [[rho, theta]] for [[rho, theta]] in houghlines])
    
    return rho_theta


def hough_lines_split(lines):
    
    vertical_lines = [] # A list to separate the vertical lines with small theta(<30) from hough_lines

    horizontal_lines = [] # A list to separate the horizontal lines with large theta(45 to 135) from hough_lines


    for line in lines:
        rho, theta = line[0]

        if 0.7854 < theta < 2.356 or 3.927 < theta < 5.4978:
            horizontal_lines.append(line[0])
        else:
            vertical_lines.append(line[0])
    
    return horizontal_lines, vertical_lines

def combine_mask_houghline(masklines, _houghlines, norm):
    best_lines = []
    for mline in masklines:
        print("\n mline: ", mline)
        best_line = mline
        for hline in _houghlines:
            print("hline: ", hline)
            mse = (((mline/norm) - (hline/norm))**2).mean()
            print("mse: ", mse)
            if mse < 0.0005:
                best_line = hline
        
        best_lines.append(best_line)

    return best_lines

def lines2corners(small_theta_points,large_theta_points):

    # Sorting the points on rho, ascending

    small_theta_points_sorted = sorted(small_theta_points, key=lambda x: x[0])
    large_theta_points_sorted = sorted(large_theta_points, key=lambda x: x[0])

    # Taking only the two points with maximum and minimum values of rho from each theta group

    horizontal_lines = [small_theta_points_sorted[0], small_theta_points_sorted[-1]]
    vertical_lines = [large_theta_points_sorted[0], large_theta_points_sorted[-1]]

    coordinates= [] # A list to store the intersecting coordinates

    for i in horizontal_lines:
        for j in vertical_lines:

            rho1 = i[0]
            theta1 = i[1]

            rho2 = j[0]
            theta2 = j[1]

            coeff = np.array([[np.cos(theta1), np.sin(theta1)],\
                            [np.cos(theta2), np.sin(theta2)]])

            rhos = np.array([rho1, rho2])
            x = np.linalg.solve(coeff,rhos)

            coordinates.append(x)



    # print(Coordinates)

    coordinates = np.array(coordinates)

    ordered_coordinates = order_points(coordinates)

    ordered_coordinates = np.int32([ordered_coordinates])

    return ordered_coordinates



if __name__ == "__main__":
    
    masknames = os.listdir('masks')
    mask_dict = {}
    my_img = cv.imread('dataset/test/20190702_105415.jpg')
    my_img = cv.cvtColor(my_img,cv.COLOR_BGR2RGB)
    zero_img = np.zeros(my_img.shape[:3],dtype=np.int32)

    for name in masknames:

        mask = cv.imread('masks/'+name, 0)
        mask_dict[name] = Mask(mask)
        mask_image = mask_dict[name].mask

        print("\n#######\nMaskname: ", name)

        image_cropped = mask_dict[name].crop_image(my_img)
        houghlines, img = hough_line_transformation(image_cropped)
        # hough_lines_show(houghlines, image_cropped)
        houghlines = scale_rho_theta(houghlines)

        # convert polygon points into the crop coordinate
        polygon_pts = mask_dict[name].polygon_points - [mask_dict[name].col_min, mask_dict[name].row_min]
        mask_houghlines = points2parametric(polygon_pts)

        hline_vertical, hline_horizontal = hough_lines_split(houghlines)
        mline_vertical, mline_horizontal = hough_lines_split(mask_houghlines)

        norm_parameter = np.array([max(image_cropped.shape), 2*np.pi])
        vertical_lines = combine_mask_houghline(mline_vertical, hline_vertical, norm_parameter)
        horizontal_lines = combine_mask_houghline(mline_horizontal, hline_vertical, norm_parameter)

        corners = lines2corners(vertical_lines, horizontal_lines)

        corners = corners + [mask_dict[name].col_min, mask_dict[name].row_min]

        cv.polylines(my_img, corners, True, (255, 0, 0), thickness=5)

        mask_dict[name].drawContour(my_img)
        mask_dict[name].drawPolygon(my_img)

plt.imshow(my_img)
plt.show()












