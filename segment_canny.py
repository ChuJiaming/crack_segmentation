import numpy as np
import cv2
import matplotlib.pyplot as plt


def seg(img):
    # insert gussian blur
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # canny edge detection
    edges = cv2.Canny(img, 70, 170)
    # morphological operations
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
    # Find contours
    cnt, hierarchy = cv2.findContours(
        edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # filter contours
    cnt_filtered = [c for c in cnt if cv2.contourArea(
        c) > 300 and cv2.contourArea(c) < 10000 and cv2.arcLength(c, True) > 300 and (cv2.boundingRect(c)[2] / cv2.boundingRect(c)[3] > 2)]
    return edges, cnt_filtered


def compare_and_color(masks, mask_f):
    # Create an empty color image
    color_img = np.zeros((*masks.shape, 3), dtype=np.uint8)

    # Find the same parts and color them blue
    same = cv2.bitwise_and(masks, mask_f)
    color_img[same > 0] = [255, 0, 0]  # BGR color

    # Find the different parts and color them red
    diff = cv2.bitwise_xor(masks, mask_f)
    color_img[diff > 0] = [0, 255, 0]  # BGR color
    return color_img


if __name__ == '__main__':
    img = cv2.imread('dataset/images/CFD_001.jpg')
    masks = cv2.imread('dataset/masks/CFD_001.jpg')
    res, contours = seg(img)
    # create black background
    black = np.zeros_like(img)
    mask_f = cv2.drawContours(
        black.copy(), contours, -1, (255, 255, 255), -1)
    masks = cv2.cvtColor(masks, cv2.COLOR_BGR2GRAY)
    masks = cv2.threshold(masks, 100, 255, cv2.THRESH_BINARY)[1]
    mask_f = cv2.cvtColor(mask_f, cv2.COLOR_BGR2GRAY)
    # compare the results
    color_img = compare_and_color(masks, mask_f)
    plt.subplot(2, 3, 2)
    plt.imshow(img)
    plt.title('Original Image')
    plt.subplot(2, 3, 4)
    plt.imshow(masks, cmap='gray')
    plt.title('test mask')
    plt.subplot(2, 3, 5)
    plt.imshow(mask_f, cmap='gray')
    plt.title('Segmented mask')
    plt.subplot(2, 3, 6)
    plt.imshow(color_img)
    plt.title('Comparison')
    plt.show()
