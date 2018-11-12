from __future__ import division
import cv2 as cv
# to show the image
from matplotlib import pyplot as plt
import numpy as np
from math import cos, sin

green = (0, 255, 0)


def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))
    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')


def overlay_mask(mask, image):
    # make the mask rgb
    rgb_mask = cv.cvtColor(mask, cv.COLOR_GRAY2RGB)
    # calculates the weightes sum of two arrays. in our case image arrays
    # input, how much to weight each.
    # optional depth value set to 0 no need
    img = cv.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img


def find_biggest_contour(image):
    # Copy
    image = image.copy()
    # input, gives all the contours, contour approximation compresses
    # horizontal, vertical, and diagonal segments and leaves only their end
    # points. For example, an up-right rectangular contour is encoded with 4
    # points. Optional output vector, containing information about the image
    # topology. It has as many elements as the number of contours. we
    # dont need it
    _, contours, _ = cv.findContours(
        image, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv.contourArea(contour), contour)
                     for contour in contours]
    biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

    mask = np.zeros(image.shape, np.uint8)
    cv.drawContours(mask, [biggest_contour], -1, 255, -1)
    return biggest_contour, mask


def box_contour(image, contour):
    # Bounding box
    image_with_box = image.copy()
    # easy function
    x, y, w, h = cv.boundingRect(contour)
    cv.rectangle(image_with_box, (x, y), (x + w, y + h), green, 2)
    return image_with_box


def find_strawberry(image):
    # RGB stands for Red Green Blue. Most often, an RGB color is stored
    # in a structure or unsigned integer with Blue occupying the least
    # significant “area” (a byte in 32-bit and 24-bit formats), Green the
    # second least, and Red the third least. BGR is the same, except the
    # order of areas is reversed. Red occupies the least significant area,
    # Green the second (still), and Blue the third.
    # we'll be manipulating pixels directly
    # most compatible for the transofrmations we're about to do
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

    # Make a consistent size
    # get largest dimension
    max_dimension = max(image.shape)
    # The maximum window size is 700 by 660 pixels. make it fit in that
    scale = 700/max_dimension
    # resize it. same width and hieght none since output is 'image'.
    image = cv.resize(image, None, fx=scale, fy=scale)

    # we want to eliminate noise from our image. clean. smooth colors without
    # dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to
    # filter, empty)
    image_blur = cv.GaussianBlur(image, (7, 7), 0)
    # t unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    # just want to focus on color, segmentation
    image_blur_hsv = cv.cvtColor(image_blur, cv.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    # minimum red amount, max red amount
    min_red = np.array([0, 100, 80])
    max_red = np.array([10, 256, 256])
    # layer
    mask1 = cv.inRange(image_blur_hsv, min_red, max_red)

    # birghtness of a color is hue
    # 170-180 hue
    min_red_lum = np.array([170, 100, 80])
    max_red_lum = np.array([180, 256, 256])
    mask2 = cv.inRange(image_blur_hsv, min_red_lum, max_red_lum)

    # looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2

    # Clean up
    # we want to circle our strawberry so we'll circle it with an box
    # with a shape of 15x15
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (15, 15))
    # morph the image. closing operation Dilation followed by Erosion.
    # It is useful in closing small holes inside the foreground objects,
    # or small black points on the object.
    mask_closed = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # erosion followed by dilation. It is useful in removing noise
    mask_clean = cv.morphologyEx(mask_closed, cv.MORPH_OPEN, kernel)

    # Find biggest strawberry
    # get back list of segmented strawberries and an outline for the biggest
    # one
    big_strawberry_contour, mask_strawberries = find_biggest_contour(
        mask_clean)

    # Overlay cleaned mask on image
    # overlay mask on image, strawberry now segmented
    overlay = overlay_mask(mask_clean, image)

    # Circle biggest strawberry
    # circle the biggest one
    boxed = box_contour(overlay, big_strawberry_contour)
    show(boxed)

    # we're done, convert back to original color scheme
    bgr = cv.cvtColor(boxed, cv.COLOR_RGB2BGR)

    return bgr


# read the image
image = cv.imread('berry.jpg')
# detect it
result = find_strawberry(image)
# display new image
cv.imshow('strawberry', result)
cv.waitKey(0)
cv.destroyAllWindows()
