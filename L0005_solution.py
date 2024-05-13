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
        self.blurred_image = cv2.GaussianBlur(self.grayscale_image, (51, 51), 0)
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


    def detect_lines2(self):
        median = cv2.medianBlur(self.grayscale_image, 3)
        kernel = np.ones((5, 5), np.uint8)
        kernel2 = np.ones((7, 7), np.uint8)
        edges = cv2.Canny(median, threshold1=1, threshold2=255, edges=5)

        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel2)
        edges = cv2.dilate(edges, kernel, iterations=1)

        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=200, minLineLength=200, maxLineGap=5)
        result_img = cv2.cvtColor(self.grayscale_image, cv2.COLOR_GRAY2RGB)

        print("Lines: " + str(len(lines)))
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(result_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # result_img[canny_img == 255] = [0, 0, 255]
        self.display_image(result_img, "result Image lines")
        self.detect_crosses()

    def merge_lines(self, lines):
        merged_lines = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            new_line = True

            for merged_line in merged_lines:
                xm1, ym1, xm2, ym2 = merged_line

                # Calculate distance between endpoints
                dist1 = np.sqrt((x1 - xm1) ** 2 + (y1 - ym1) ** 2)
                dist2 = np.sqrt((x2 - xm2) ** 2 + (y2 - ym2) ** 2)

                # If both endpoints are close enough to an existing line, merge them
                if dist1 < 10 and dist2 < 10:
                    merged_line[0] = min(x1, xm1)
                    merged_line[1] = min(y1, ym1)
                    merged_line[2] = max(x2, xm2)
                    merged_line[3] = max(y2, ym2)
                    new_line = False
                    break

            if new_line:
                merged_lines.append([x1, y1, x2, y2])

        return merged_lines

    def detect_lines(self):
        median = cv2.medianBlur(self.grayscale_image, 7)
        edges = cv2.Canny(median, threshold1=1, threshold2=255, edges=4)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)

        lines = cv2.HoughLinesP(image=edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=5)

        result_img2 = cv2.cvtColor(self.grayscale_image, cv2.COLOR_GRAY2RGB)

        if lines is not None:
            merged_lines = self.merge_lines(lines)

            for line in merged_lines:
                x1, y1, x2, y2 = line

                # Fit a line using fitLine
                vx, vy, x0, y0 = cv2.fitLine(np.array([[x1, y1], [x2, y2]]), cv2.DIST_L2, 0, 0.01, 0.01)

                # Get the direction vector
                line_direction = np.array([vx, vy])

                # Get the center point
                line_center = np.array([x0, y0])

                # Define a scale factor to extend the line
                scale_factor = 100.0

                # Calculate the endpoints of the extended line
                # Calcola i punti di estremità della linea estesa
                # Calcola i punti di estremità della linea estesa
                point1 = (int(line_center[0][0] - scale_factor * line_direction[0][0]),
                          int(line_center[1][0] - scale_factor * line_direction[1][0]))
                point2 = (int(line_center[0][0] + scale_factor * line_direction[0][0]),
                          int(line_center[1][0] + scale_factor * line_direction[1][0]))

                # Draw the extended line on the image
                cv2.line(result_img2, point1, point2, (0, 0, 255), 2)

        self.display_image(edges, "edges")

        self.display_image(result_img2, "result Image lines after")

        '''
        for line in merged_lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result_img1, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Calcola il vettore direzione della linea
            line_direction = np.array([x2 - x1, y2 - y1], dtype=np.float32)

            # Normalizza il vettore direzione
            line_direction /= np.linalg.norm(line_direction)

            # Calcola il centro della linea
            line_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2], dtype=np.float32)

            # Calcola la lunghezza della linea
            line_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

            # Definisci un fattore di scala per estendere la linea
            scale_factor = 100.0  # Puoi regolare questo valore in base alle tue esigenze

            # Calcola i punti di estremità della linea estesa
            point1 = tuple((line_center - scale_factor * line_direction).astype(int))
            point2 = tuple((line_center + scale_factor * line_direction).astype(int))

            # Disegna la linea estesa sull'immagine originale
            cv2.line(result_img2, point1, point2, (0, 0, 255), 2)
            '''

    def detect_crosses(self):
        # Implement cross identification logic here
        # Draw circles or rectangles around crosses
        # Example:
        pass



class ImageFeaturesTask:
    def __init__(self, grayscale_image):
        self.grayscale_image = grayscale_image

    def corner_detection(self):
        kernel2 = np.ones((7, 7), np.uint8)
        median = cv2.medianBlur(self.grayscale_image, 5)

        edges = cv2.Canny(median, threshold1=1, threshold2=255, edges=3)

        corners = cv2.goodFeaturesToTrack(edges, 100, 0.01, 80)
        corners = np.intp(corners)
        result_img = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result_img, (x, y), 3, (0, 0, 255), 2)
        cv2.imshow("Image with Corners", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def calculate_and_plot_histogram(self):
        histogram = cv2.calcHist([self.grayscale_image], [0], None, [256], [0, 256])
        plt.plot(histogram)
        plt.title('Grayscale Image Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.show()

# Example Usage
if __name__ == "__main__":
    image_processor = ImageProcessingTask("Chess_board.jpg", enable_debug_show=False)
    image_processor.convert_to_grayscale()
    image_processor.apply_gaussian_blur()
    image_processor.rotate_image(45)
    image_processor.apply_binary_threshold()

    line_detection_task = LineDetectionAndCrossIdentificationTask(image_processor.grayscale_image)
    line_detection_task.detect_lines()

    image_features_task = ImageFeaturesTask(image_processor.grayscale_image)
    image_features_task.corner_detection()
    image_features_task.calculate_and_plot_histogram()
