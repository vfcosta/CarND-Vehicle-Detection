import numpy as np
from vehicledetection.search import search_windows
from vehicledetection.sliding_window import slide_window
from vehicledetection.classifier import Classifier
from vehicledetection.draw import draw_boxes, draw_labeled_bboxes, draw_heatmap_image
from vehicledetection.heatmap import Heatmap


classifier = Classifier.load()
heatmap = None
windows = None


def generate_result_image(image, display_all_boxes=False, display_heatmap_boxes=True, display_heatmap=False):
    """Generate an image with bounding boxes representing vehicle detections"""
    global heatmap
    if heatmap is None:
        print("initializing heatmap class")
        heatmap = Heatmap(image.shape)

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


def slide_windows(image):
    """Store custom windows for image search using five sizes"""
    global windows
    if windows is not None:
        return windows
    windows = []
    windows_settings = [((64, 64), [400, 500], (0.6, 0.6)),
                        ((84, 84), [400, 600], (0.5, 0.5)),
                        ((104, 104), [400, None], (0.5, 0.5)),
                        ((124, 124), [400, None], (0.5, 0.5)),
                        ((144, 144), [400, None], (0.5, 0.5))]
    for xy_window, y_start_stop, xy_overlap in windows_settings:
        windows += slide_window(image, x_start_stop=[None, None], y_start_stop=y_start_stop,
                           xy_window=xy_window, xy_overlap=xy_overlap)
    return windows


def process_image(image):
    """Process an input image to return possible windows with detections"""
    windows = slide_windows(image)
    hot_windows = search_windows(image, windows, classifier, classifier.scaler, color_space=classifier.color_space,
                                 spatial_size=classifier.spatial_size, hist_bins=classifier.hist_bins,
                                 orient=classifier.orient, pix_per_cell=classifier.pix_per_cell,
                                 cell_per_block=classifier.cell_per_block,
                                 hog_channel=classifier.hog_channel, spatial_feat=classifier.spatial_feat,
                                 hist_feat=classifier.hist_feat, hog_feat=classifier.hog_feat)
    return hot_windows
