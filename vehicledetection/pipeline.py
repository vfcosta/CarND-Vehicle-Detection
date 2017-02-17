import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import os.path
from vehicledetection.sliding_window import slide_window, draw_boxes
from vehicledetection.features import color_hist, bin_spatial, get_hog_features


def process_sample_images(process):
    images = glob.glob(os.path.join('..', 'test_images', '*'))
    for image_name in images:
        image = mpimg.imread(image_name)
        process(image, image_name)


def write_result(image, image_name, suffix='', cmap=None):
    name, extension = os.path.splitext(os.path.basename(image_name))
    if suffix:
        suffix = '_' + suffix
    plt.imsave(os.path.join('..', 'output_images', name + suffix + extension), image, cmap=cmap)


def sliding_windows():
    image = mpimg.imread('../test_images/test1.jpg')
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(128, 128), xy_overlap=(0.5, 0.5))
    window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
    write_result(window_img, 'test1.jpg', suffix='windows')


def color_hist(image):
    hist = color_hist(image)
    plt.plot(hist)
    plt.show()


def bin_spatial(image):
    features = bin_spatial(image)
    plt.plot(features)
    plt.show()


def hog(image, image_name, display=False):
    features = get_hog_features(image[:, :, 0], vis=True, orient=12, pix_per_cell=16, cell_per_block=2)
    result = features[1]
    if display:
        plt.imshow(result, cmap='gray')
        plt.show()

    write_result(result, image_name, suffix='hog', cmap='gray')
    return result


if __name__ == "__main__":
    sliding_windows()
    # display_color_hist()
    # display_bin_spatial()
    process_sample_images(hog)
