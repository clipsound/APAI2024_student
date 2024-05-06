import cv2
import numpy as np
import matplotlib.pyplot as plt



class ImageProcessingTask:
    def __init__(self, image_path, enable_debug_show=False):
        self.enable_debug_show = enable_debug_show
        self.image = cv2.imread(image_path)
        if self.enable_debug_show:
            self.display_image(image=self.image, title="Color Image")

    def display_image(self, image, title="Image"):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def convert_to_grayscale(self):
        self.grayscale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if self.enable_debug_show:
            self.display_image(self.grayscale_image, "Grayscale Image")

    def apply_gaussian_blur(self):
        self.blurred_image = cv2.GaussianBlur(self.grayscale_image, (5, 5), 0)
        if self.enable_debug_show:
            self.display_image(self.blurred_image, "Blurred Image")

    def rotate_image(self, angle):
        rows, cols = self.image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        self.rotated_image = cv2.warpAffine(self.image, rotation_matrix, (cols, rows))
        if self.enable_debug_show:
            self.display_image(self.rotated_image, "Rotated Image")

    def apply_binary_threshold(self, threshold_value=128, max_value=255, threshold_type=cv2.THRESH_BINARY):
        _, self.thresholded_image = cv2.threshold(self.grayscale_image, threshold_value, max_value, threshold_type)
        if self.enable_debug_show:
            self.display_image(self.thresholded_image, "Thresholded Image")

class LineDetectionAndCrossIdentificationTask:
    def __init__(self, grayscale_image):
        self.grayscale_image = grayscale_image

    def display_image(self, image, title="Image"):
        cv2.imshow(title, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def detect_lines(self):
        # TODO IMPLEMENT ...

        self.display_image(result_img2, "result Image lines after")



class ImageFeaturesTask:
    def __init__(self, grayscale_image):
        self.grayscale_image = grayscale_image

    def corner_detection(self):
        # TODO IMPLEMENT ...
        cv2.imshow("Image with Corners", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate_and_plot_histogram(self):
        # TODO IMPLEMENT ...

        plt.plot(histogram)
        plt.title('Grayscale Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

# Example Usage
if __name__ == "__main__":
    image_processor = ImageProcessingTask("Chess_board.jpg")
    image_processor.convert_to_grayscale()
    image_processor.apply_gaussian_blur()
    image_processor.rotate_image(45)
    image_processor.apply_binary_threshold()

    line_detection_task = LineDetectionAndCrossIdentificationTask(image_processor.grayscale_image)
    line_detection_task.detect_lines()

    image_features_task = ImageFeaturesTask(image_processor.grayscale_image)
    image_features_task.corner_detection()
    image_features_task.calculate_and_plot_histogram()
