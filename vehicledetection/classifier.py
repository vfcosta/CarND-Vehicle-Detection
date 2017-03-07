import os.path
import numpy as np
import glob
import time
import pickle
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from vehicledetection.features import extract_features
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


class Classifier:

    def __init__(self):
        self.svc = None
        self.scaler = None
        self.rand_state = np.random.randint(0, 100)
        # initialize parameters to use in feature extraction
        self.color_space = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        self.orient = 12  # HOG orientations
        self.pix_per_cell = 16  # HOG pixels per cell
        self.cell_per_block = 2  # HOG cells per block
        self.hog_channel = "ALL"  # Can be 0, 1, 2, or "ALL"
        self.spatial_size = (16, 16)  # Spatial binning dimensions
        self.hist_bins = 32  # Number of histogram bins
        self.spatial_feat = True  # Spatial features on or off
        self.hist_feat = True  # Histogram features on or off
        self.hog_feat = True  # HOG features on or off
        self.y_start_stop = [400, None]  # Min and max in y to search in slide_window()
        self.hist_range = (0, 256)

    def train(self):
        images = glob.glob(os.path.join('../data/non-vehicles', '**/*.png'))
        notcars = [image for image in images]

        images = glob.glob(os.path.join('../data/vehicles', '**/*.png'))
        cars = [image for image in images]

        car_features = extract_features(cars, color_space=self.color_space,
                                        spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                        orient=self.orient, pix_per_cell=self.pix_per_cell,
                                        cell_per_block=self.cell_per_block, hist_range=self.hist_range,
                                        hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                        hist_feat=self.hist_feat, hog_feat=self.hog_feat)
        notcar_features = extract_features(notcars, color_space=self.color_space,
                                           spatial_size=self.spatial_size, hist_bins=self.hist_bins,
                                           orient=self.orient, pix_per_cell=self.pix_per_cell,
                                           cell_per_block=self.cell_per_block,
                                           hog_channel=self.hog_channel, spatial_feat=self.spatial_feat,
                                           hist_feat=self.hist_feat, hog_feat=self.hog_feat)

        X = np.vstack((car_features, notcar_features)).astype(np.float64)
        # Fit a per-column scaler
        self.scaler = StandardScaler().fit(X)
        # Apply the scaler to X
        scaled_X = self.scaler.transform(X)

        # Define the labels vector
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

        # Split up data into randomized training and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=self.rand_state)

        print('Using:', self.orient, 'orientations', self.pix_per_cell,
              'pixels per cell and', self.cell_per_block, 'cells per block')
        print('Feature vector length:', len(X_train[0]))
        # Use a linear SVC
        self.svc = LinearSVC(verbose=1)
        # Check the training time for the SVC
        t = time.time()
        self.svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2 - t, 2), 'Seconds to train SVC...')

        # Check the score of the SVC
        print('Train Accuracy of SVC = ', round(self.svc.score(X_train, y_train), 4))
        print('Test Accuracy of SVC = ', round(self.svc.score(X_test, y_test), 4))
        train_cf_matrix = confusion_matrix(y_train, self.svc.predict(X_train))
        y_pred = self.svc.predict(X_test)
        test_cf_matrix = confusion_matrix(y_test, y_pred)
        print(train_cf_matrix)
        print(test_cf_matrix)
        print(classification_report(y_test, y_pred))

    def predict(self, features):
        return self.svc.predict(features)

    def save(self, filename='classifier.p'):
        """Save parameters from Classifier class in file"""
        pickle.dump(vars(self), open(filename, "wb"))

    def save_confusion_matrix(self, cm):
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # normalize confusion matrix
        plt.matshow(cm, cmap='gray')
        plt.show()

    @staticmethod
    def load(filename='classifier.p'):
        """Load classifier parameters from file"""
        classifier = Classifier()
        attributes = pickle.load(open(filename, "rb"))
        for k, v in attributes.items():
            setattr(classifier, k, v)
        return classifier


if __name__ == "__main__":
    classifier = Classifier()
    classifier.train()
    classifier.save()
