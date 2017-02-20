import numpy as np
from vehicledetection.search import search_windows
from vehicledetection.sliding_window import slide_window
from vehicledetection.classifier import Classifier
from vehicledetection.draw import draw_boxes, draw_labeled_bboxes, draw_heatmap_image
from vehicledetection.heatmap import Heatmap


classifier = Classifier.load()
heatmap = None


def generate_result_image(image, display_all_boxes=True, display_heatmap_boxes=True, display_heatmap=True):
    global heatmap
    if heatmap == None:
        print("initializing heatmap class")
        heatmap = Heatmap(image.shape, threshold=2)

    hot_windows = process_image(image)
    heatmap.add_heat(hot_windows)

    result_image = np.copy(image)
    if display_all_boxes:
        result_image = draw_boxes(result_image, hot_windows, color=(255, 0, 0), thick=6)

    if display_heatmap_boxes:
        cars, n = heatmap.detect()
        result_image = draw_labeled_bboxes(result_image, cars, n)

    if display_heatmap:
        result_image = draw_heatmap_image(result_image, heatmap.detections)

    return result_image


def process_image(image):
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=classifier.y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, classifier, classifier.scaler, color_space=classifier.color_space,
                                 spatial_size=classifier.spatial_size, hist_bins=classifier.hist_bins,
                                 orient=classifier.orient, pix_per_cell=classifier.pix_per_cell,
                                 cell_per_block=classifier.cell_per_block,
                                 hog_channel=classifier.hog_channel, spatial_feat=classifier.spatial_feat,
                                 hist_feat=classifier.hist_feat, hog_feat=classifier.hog_feat)
    return hot_windows
