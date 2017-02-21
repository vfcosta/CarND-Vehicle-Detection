import cv2
import numpy as np


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def draw_labeled_bboxes(img, cars, n_cars):
    # Iterate through all detected cars
    for car_number in range(1, n_cars + 1):
        # Find pixels with each car_number label value
        nonzero = (cars == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def draw_heatmap_image(image, heatmap, w=320, h=180, max_value=8):
    """Draw the heatmap on the top right corner of the image."""
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_resized = np.clip(heatmap_resized * 255/max_value, 0, 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_HOT)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    image[0:h, image.shape[1] - w:] = heatmap_color
    return image
