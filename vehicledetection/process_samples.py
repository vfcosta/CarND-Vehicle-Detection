import matplotlib.pyplot as plt
import glob
import os.path
import cv2
import numpy as np
import time
from vehicledetection.sliding_window import slide_window
from vehicledetection.features import color_hist, bin_spatial, get_hog_features
from vehicledetection.heatmap import Heatmap
from vehicledetection.pipeline import process_image, generate_result_image
from vehicledetection.draw import draw_labeled_bboxes, draw_heatmap_image, draw_boxes


def process_sample_images(process):
    print('process images', process.__name__)
    images = glob.glob(os.path.join('..', 'test_images', '*'))
    for image_name in images:
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        process(image, image_name)


def write_result(image, image_name, suffix='', cmap=None, save_fig=False):
    name, extension = os.path.splitext(os.path.basename(image_name))
    if suffix:
        suffix = '_' + suffix
    if save_fig:
        plt.savefig(os.path.join('..', 'output_images', name + suffix + extension))
    else:
        plt.imsave(os.path.join('..', 'output_images', name + suffix + extension), image, cmap=cmap)


def sliding_windows():
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    windows_settings = [((64, 64), [400, 500]),
                        ((84, 84), [400, 600]),
                        ((104, 104), [400, None]),
                        ((124, 124), [400, None]),
                        ((144, 144), [400, None])]

    windows = []
    for xy_window, y_start_stop in windows_settings:
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=(0.5, 0.5))
    print(len(windows))
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    write_result(window_img, 'test1.jpg', suffix='windows')


def display_color_hist(image, image_name):
    hist = color_hist(image)
    plt.clf()
    plt.plot(hist)
    write_result(None, image_name, suffix='histogram', save_fig=True)


def display_bin_spatial(image, image_name):
    features = bin_spatial(image, size=(16, 16))
    spatial_img = np.array(features)
    spatial_img = np.reshape(spatial_img, (16, 16, 3))
    write_result(spatial_img, image_name, suffix='spatial')


def hog(image, image_name, display=False):
    features = get_hog_features(image[:, :, 0], vis=True, orient=12, pix_per_cell=8, cell_per_block=2)
    result = features[1]
    if display:
        plt.imshow(result, cmap='gray')
        plt.show()

    write_result(result, image_name, suffix='hog', cmap='gray')
    return result


def search():
    print('search vehicles')
    for image_name in glob.glob('../test_images/test*.jpg'):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_image = np.copy(image)
        # Check the prediction time for a single sample
        t = time.time()
        hot_windows = process_image(image)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to search for cars...')
        window_img = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)
        write_result(window_img, image_name, suffix='search')


def heatmap():
    print('generating samples for heatmap')
    image = cv2.imread('../test_images/test1.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hot_windows = process_image(image)
    heatmap = Heatmap(image.shape, threshold=1)
    heatmap.add_heat(hot_windows)
    write_result(heatmap.detections, 'test1.jpg', suffix='heatmap_detections', cmap='hot')

    cars, n = heatmap.detect()
    print(n, 'cars found')
    write_result(cars, 'test1.jpg', suffix='heatmap_thresh', cmap='gray')

    # Draw bounding boxes on a copy of the image
    draw_img = draw_labeled_bboxes(np.copy(image), cars, n)
    write_result(draw_img, 'test1.jpg', suffix='heatmap_labels')

    result = draw_heatmap_image(draw_img, heatmap.detections)
    write_result(result, 'test1.jpg', suffix='result')


def full_pipeline():
    print('generating samples for full pipeline')
    for image_name in glob.glob('../test_images/test*.jpg'):
        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draw_image = generate_result_image(image)
        write_result(draw_image, image_name, suffix='pipeline')


if __name__ == "__main__":
    # sliding_windows()
    # process_sample_images(display_color_hist)
    process_sample_images(display_bin_spatial)
    # process_sample_images(hog)
    # search()
    # heatmap()
    # full_pipeline()
