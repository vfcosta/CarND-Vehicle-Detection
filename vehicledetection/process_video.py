from moviepy.editor import VideoFileClip
import os.path
import numpy as np
from vehicledetection.pipeline import process_image
from vehicledetection.draw import draw_boxes


base_dir = os.path.dirname(__file__)


def process_frame(image):
    """Use pipeline to process a single image frame and return an image with lane lines drawn on top"""
    draw_image = np.copy(image)
    hot_windows = process_image(image)
    return draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)


def process_video(filename):
    """Process a video using the lane finding pipeline."""
    white_output = os.path.join(base_dir, '..', filename+'_processed.mp4')
    clip1 = VideoFileClip(os.path.join(base_dir, '..', filename+'.mp4'))
    white_clip = clip1.fl_image(process_frame)
    white_clip.write_videofile(white_output, audio=False)


if __name__ == '__main__':
    # process_video('project_video')
    process_video('test_video')
