from scipy.ndimage.measurements import label
import numpy as np


class Heatmap:

    def __init__(self, shape, threshold=2):
        self.shape = shape  # shape of the images that will be used in heatmap detection
        self.detections = np.zeros(shape[:2])  # array of shape to track detections
        self.threshold = threshold
        self.decay = 0.5

    def add_heat(self, bbox_list):
        """Add a bounding box list to the detection heat map array"""
        # penalize old detections by a decay factor
        self.detections -= self.decay
        # Iterate through list of bounding boxes
        for box in bbox_list:
            # Add += 1 for all pixels inside each bbox assuming each "box" takes the form ((x1, y1), (x2, y2))
            self.detections[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
        self.detections = np.clip(self.detections, 0, 2 * self.threshold)
        # Return updated heatmap detections
        return self.detections

    def apply_threshold(self):
        """Zero out pixels below the threshold"""
        self.detections[self.detections <= self.threshold] = 0
        return self.detections

    def detect(self):
        """Apply the threshold and return the cars found in heat map"""
        return label(self.apply_threshold())
