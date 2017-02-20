from scipy.ndimage.measurements import label
import numpy as np


class Heatmap:

    def __init__(self, shape, threshold=2):
        self.shape = shape  # shape of the images that will be used in heatmap detection
        self.detections = np.zeros(shape[:2])  # array of shape to track detections
        self.threshold = threshold

    def add_heat(self, bbox_list):
        # Iterate through list of bboxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox
            # Assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.detections[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

        # Return updated heatmap detections
        return self.detections

    def apply_threshold(self):
        # Zero out pixels below the threshold
        self.detections[self.detections <= self.threshold] = 0

    def detect(self):
        self.apply_threshold()
        return label(self.detections)
