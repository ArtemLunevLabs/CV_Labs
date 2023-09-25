from typing import List, Tuple
from time import time

import cv2
import numpy as np


def get_foreground_mask(image_path: str) -> Tuple[List[tuple], float]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    pixel_count = img.shape[0] * img.shape[1]
    start = time()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blured = cv2.GaussianBlur(gray_img, (3, 3), 0)

    img_canny = cv2.Canny(gray_blured, 15, 55)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=15)
    markers = cv2.erode(img_dilate, kernel, iterations=10)

    markers[markers != 255] = 0
    markers[markers == 255] = 1
    end = time()

    return np.argwhere(markers), (end - start) / (pixel_count / 1e6)


def get_foreground_mask_with_watershed(image_path: str) -> Tuple[List[tuple], float]:
    """
    Метод для вычисления маски переднего плана на фото
    :param image_path - путь до фото
    :return массив в формате [(x_1, y_1), (x_2, y_2), (x_3, y_3)], в котором перечислены все точки, относящиеся к маске
    """

    img = cv2.imread(image_path, cv2.IMREAD_COLOR)

    pixel_count = img.shape[0] * img.shape[1] * img.shape[2]
    start = time()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blured = cv2.GaussianBlur(gray_img, (3, 3), 0)

    img_canny = cv2.Canny(gray_blured, 15, 55)
    kernel = np.ones((3, 3))
    img_dilate = cv2.dilate(img_canny, kernel, iterations=5)

    kernel = np.ones((40, 40))
    img_erode = cv2.erode(img_dilate, kernel)
    markers = np.int32(img_erode)

    markers = cv2.circle(markers, (10, 10), 10, 1, -1)
    markers = cv2.circle(markers, (markers.shape[1] - 10, markers.shape[0] - 10), 10, 2, -1)
    markers = cv2.watershed(img, markers)

    # colors = []
    # found_labels = list(np.unique(markers))
    # for i in range(len(np.unique(markers))):
    #     b = np.random.uniform(0, 256)
    #     g = np.random.uniform(0, 256)
    #     r = np.random.uniform(0, 256)
    #     colors.append((b, g, r))
    #
    # dst_colored = np.zeros((markers.shape[0], markers.shape[1], 3), np.uint8)
    # for i in range(markers.shape[0]):
    #     for j in range(markers.shape[1]):
    #         if markers[i][j] > 0:
    #             dst_colored[i][j] = colors[found_labels.index(markers[i][j])]
    #
    # cv2.imshow('dst_colored', dst_colored)
    # cv2.waitKey()

    markers[markers != 255] = 0
    markers[markers == 255] = 1
    markers = np.uint8(markers)
    kernel = np.ones((3, 3))
    markers = cv2.dilate(markers, kernel, iterations=8)
    end = time()

    return np.argwhere(markers), (end - start) / (pixel_count / 1e6)
