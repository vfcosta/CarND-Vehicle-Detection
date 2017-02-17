import numpy as np
import cv2


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop[0] = x_start_stop[0] or 0
    x_start_stop[1] = x_start_stop[1] or img.shape[1]
    y_start_stop[0] = y_start_stop[0] or 0
    y_start_stop[1] = y_start_stop[1] or img.shape[0]
    # Compute the span of the region to be searched
    span = (x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0])
    # Compute the number of pixels per step in x/y
    pixels_per_step = (np.array(xy_window) * (1 - np.array(xy_overlap))).astype(int)
    # Compute the number of windows in x/y
    n_windows = (span / pixels_per_step - 1).astype(int)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for ys in range(n_windows[1]):
        for xs in range(n_windows[0]):
            # Calculate window position
            startx = xs * pixels_per_step[0] + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * pixels_per_step[1] + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list
