import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from vehicledetection.search import single_img_features
from vehicledetection.sliding_window import slide_window, draw_boxes
from vehicledetection.features import color_hist, bin_spatial, get_hog_features


def display_sliding_windows():
    image = mpimg.imread('../test_images/test1.jpg')
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    plt.imshow(window_img)
    plt.show()


def display_color_hist():
    img = mpimg.imread('../test_images/test1.jpg')
    hist = color_hist(img)
    plt.plot(hist)
    plt.show()


def display_bin_spatial():
    img = mpimg.imread('../test_images/test1.jpg')
    features = bin_spatial(img)
    plt.plot(features)
    plt.show()


def display_hog():
    img = mpimg.imread('../test_images/test1.jpg')
    features = get_hog_features(img[:, :, 0], vis=True, orient=12, pix_per_cell=16, cell_per_block=2)
    plt.imshow(features[1], cmap='gray')
    plt.show()


if __name__ == "__main__":
    # display_sliding_windows()
    # display_color_hist()
    # display_bin_spatial()
    display_hog()
