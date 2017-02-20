from vehicledetection.search import search_windows
from vehicledetection.sliding_window import slide_window
from vehicledetection.classifier import Classifier


classifier = Classifier.load()


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
