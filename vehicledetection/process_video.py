from moviepy.editor import VideoFileClip
import os.path
import numpy as np
from vehicledetection.search import search_windows
from vehicledetection.sliding_window import slide_window, draw_boxes
from vehicledetection.classifier import Classifier


base_dir = os.path.dirname(__file__)
classifier = Classifier.load()


def process_frame(image):
    """Use pipeline to process a single image frame and return an image with lane lines drawn on top"""
    draw_image = np.copy(image)

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=classifier.y_start_stop,
                           xy_window=(96, 96), xy_overlap=(0.5, 0.5))

    hot_windows = search_windows(image, windows, classifier, classifier.scaler, color_space=classifier.color_space,
                                 spatial_size=classifier.spatial_size, hist_bins=classifier.hist_bins,
                                 orient=classifier.orient, pix_per_cell=classifier.pix_per_cell,
                                 cell_per_block=classifier.cell_per_block,
                                 hog_channel=classifier.hog_channel, spatial_feat=classifier.spatial_feat,
                                 hist_feat=classifier.hist_feat, hog_feat=classifier.hog_feat)

    window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)
    return window_img


def process_video(filename):
    """Process a video using the lane finding pipeline."""
    white_output = os.path.join(base_dir, '..', filename+'_processed.mp4')
    clip1 = VideoFileClip(os.path.join(base_dir, '..', filename+'.mp4'))
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    # process_video('project_video')
    process_video('test_video')
