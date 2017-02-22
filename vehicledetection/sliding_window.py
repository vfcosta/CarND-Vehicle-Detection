import numpy as np


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Define a function that takes an image, start and stop positions in both x and y,
    window size (x and y dimensions), and overlap fraction (for both x and y)
    """
    # Initialize a list to append window positions to
    window_list = []
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


def slide_windowaa(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    """Define a function that takes an image, start and stop positions in both x and y,
    window size (x and y dimensions), and overlap fraction (for both x and y)
    """
    # Initialize a list to append window positions to
    window_list = []
    # If x and/or y start/stop positions not defined, set to image size
    x_start_stop[0] = x_start_stop[0] or 0
    x_start_stop[1] = x_start_stop[1] or img.shape[1]
    y_start_stop[0] = y_start_stop[0] or 0
    y_start_stop[1] = y_start_stop[1] or img.shape[0]
    # Compute the span of the region to be searched
    span = (x_start_stop[1] - x_start_stop[0], y_start_stop[1] - y_start_stop[0])
    # Compute the number of pixels per step in x/y

    endy = 0
    i = 0
    while endy < y_start_stop[1]:
        i += 1
        size = max(int(xy_window[1] * i * (1 - xy_overlap[1])), xy_window[1])
        pixels_per_step = (np.array([size, size]) * (1 - np.array(xy_overlap))).astype(int)
        # Compute the number of windows in x/y
        n_windows = (span / pixels_per_step - 1).astype(int)
        # print(size, pixels_per_step)
        # Compute y positions
        starty = y_start_stop[0]
        endy = starty + size
        # print(starty, endy)

        # Loop through finding x window positions
        for xs in range(n_windows[0]):
            # Calculate window position
            startx = xs * pixels_per_step[0] + x_start_stop[0]
            endx = startx + size
            # Append window position to list
            # print(((startx, starty), (min(endx, x_start_stop[1]), min(endy, y_start_stop[1]))))
            window_list.append(((startx, starty), (min(endx, x_start_stop[1]), min(endy, y_start_stop[1]))))
    # Return the list of windows
    return window_list
