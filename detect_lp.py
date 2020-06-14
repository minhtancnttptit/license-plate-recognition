import cv2
import imutils
import pytesseract
import numpy as np

# Param
from PIL import Image

max_size = 15000
min_size = 100
im = cv2.imread("xemay1.jpg")
im_gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
noise_removal = cv2.bilateralFilter(im_gray,9,75,75)
equal_histogram = cv2.equalizeHist(noise_removal)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
morph_image = cv2.morphologyEx(equal_histogram,cv2.MORPH_OPEN,kernel,iterations=20)
sub_morp_image = cv2.subtract(equal_histogram,morph_image)
ret,thresh_image = cv2.threshold(sub_morp_image, 0,255,cv2.THRESH_OTSU)
canny_image = cv2.Canny(thresh_image,250,255)
kernel = np.ones((3,3), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)
cv2.imshow('dilated image', dilated_image)

cnts = cv2.findContours(dilated_image.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:10]
screenCnt = None

# loop over our contours
for c in cnts:
    # approximate the contour
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.05 * peri, True)
    rect = cv2.boundingRect(approx)
    width = rect[0] - rect[2]
    height = rect[1] - rect[3]
    # print(c)
    # if our approximated contour has four points, then
    # we can assume that we have found our screen
    if len(approx) == 4 and 0.6 < width / height < 0.8:
        screenCnt = approx
        break

if screenCnt is None:
    detected = 0
    print ("No plate detected")
else:
    detected = 1

if detected == 1:
    images = []
    cv2.drawContours(im, [screenCnt], -1, (0, 255, 0), 3)

    # Masking the part other than the number plate
    mask = np.zeros(im_gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(im, im, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    cropped = im_gray[topx:bottomx + 1, topy:bottomy + 1]
    cropped2 = thresh_image[topx:bottomx + 1, topy:bottomy + 1]
    cv2.imshow('License plate', cropped)
    cnts = cv2.findContours(cropped2.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:9]

    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.05 * peri, True)
        rect = cv2.boundingRect(c)
        x = rect[2]
        y = rect[3]
        if x * y < 2000:
            images.append(rect)
            cv2.rectangle(cropped, rect, (0, 255, 0))
            cv2.imshow('test', cropped)
    imagesSorted = sorted(images, key=lambda x: (x[1] // 20, x[0]));
    i = 1
    result = ''
    for image in imagesSorted:
        charImg = cropped2[image[1] - 1: image[1] + image[3] + 1, image[0] - 1: image[0] + image[2] + 1]
        cv2.imshow('char ' + str(i), charImg)
        result += pytesseract.image_to_string(Image.fromarray(charImg), config='--psm 8')
        i +=1
    print(result)
#
#     # Display image
cv2.imshow('Input image', im)
cv2.waitKey(0)
cv2.destroyAllWindows()



