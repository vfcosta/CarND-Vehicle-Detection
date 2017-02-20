import matplotlib.pyplot as plt
import glob
import os.path
import cv2
import numpy as np
import time
from vehicledetection.sliding_window import slide_window
from vehicledetection.features import color_hist, bin_spatial, get_hog_features
from vehicledetection.heatmap import Heatmap
from vehicledetection.pipeline import process_image
from vehicledetection.draw import draw_labeled_bboxes, draw_heatmap_image, draw_boxes


def process_sample_images(process):
    images = glob.glob(os.path.join('..', 'test_images', '*'))
    for image_name in images:
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process(image, image_name)


def write_result(image, image_name, suffix='', cmap=None):
    name, extension = os.path.splitext(os.path.basename(image_name))
    if suffix:
        suffix = '_' + suffix
    plt.imsave(os.path.join('..', 'output_images', name + suffix + extension), image, cmap=cmap)


def sliding_windows():
    image = cv2.imread('../test_images/test1.jpg')
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    write_result(window_img, 'test1.jpg', suffix='windows')


def color_hist(image):
    hist = color_hist(image)
    plt.plot(hist)
    plt.show()


def bin_spatial(image):
    features = bin_spatial(image)
    plt.plot(features)
    plt.show()


def hog(image, image_name, display=False):
    features = get_hog_features(image[:, :, 0], vis=True, orient=12, pix_per_cell=16, cell_per_block=2)
    result = features[1]
    if display:
        plt.imshow(result, cmap='gray')
        plt.show()

    write_result(result, image_name, suffix='hog', cmap='gray')
    return result


def search():
    for image_name in glob.glob('../test_images/test*.jpg'):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_image = np.copy(image)
        # Check the prediction time for a single sample
        t = time.time()
        hot_windows = process_image(image)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to search for cars...')
        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
        write_result(window_img, image_name, suffix='search')


def heatmap():
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hot_windows = process_image(image)
    heatmap = Heatmap(image.shape, threshold=1)
    heatmap.add_heat(hot_windows)
    plt.imshow(heatmap.detections, cmap='hot')
    plt.show()

    cars, n = heatmap.detect()
    print(n, 'cars found')
    plt.imshow(cars, cmap='gray')
    plt.show()

    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), cars, n)
    # Display the image
    plt.imshow(cv2.cvtColor(draw_img, cv2.COLOR_BGR2RGB))
    plt.show()

    result = draw_heatmap_image(draw_img, heatmap.detections)
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.show()

if __name__ == "__main__":
    # sliding_windows()
    # display_color_hist()
    # display_bin_spatial()
    # process_sample_images(hog)
    # search()
    heatmap()
