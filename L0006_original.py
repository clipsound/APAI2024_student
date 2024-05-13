import cv2
import numpy as np

def task1(path_image):
    image = cv2.imread(path_image)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    image_result = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)
    image_result2 = cv2.cvtColor(image_hsv, cv2.COLOR_HSV2BGR)

    lower_green = np.array([40, 50, 50])
    upper_green = np.array([80, 255, 255])

    mask = cv2.inRange(image_hsv, lower_green, upper_green)

    green_areas = cv2.bitwise_and(image_hsv, image_hsv, mask=mask)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) > 100:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_result, (x, y), (x + w, y + h), (255, 255, 255), 5)

    height, width, _ = image.shape
    view_factor = 3
    resized_height = height // view_factor
    resized_width = width // view_factor

    mask_color = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_color[mask > 0] = [255, 255, 255]

    mosaic = np.concatenate((
        cv2.resize(image_hsv, (resized_width, resized_height)),
        cv2.resize(mask_color, (resized_width, resized_height)),
        cv2.resize(green_areas, (resized_width, resized_height)),
        cv2.resize(image_result, (resized_width, resized_height)),
        cv2.resize(image_result2, (resized_width, resized_height)),
    ), axis=1)

    cv2.imshow('ColorSpaces (rgb, hsv, lab, ycrcb)', mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def task0(path_image):
    image = cv2.imread(path_image)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    height, width, _ = image.shape
    view_factor = 3
    resized_height = height // view_factor
    resized_width = width // view_factor

    mosaic = np.concatenate((
        cv2.resize(image_rgb, (resized_width, resized_height)),
        cv2.resize(image_hsv, (resized_width, resized_height)),
        cv2.resize(image_lab, (resized_width, resized_height)),
        cv2.resize(image_ycrcb, (resized_width, resized_height))
    ), axis=1)

    cv2.imshow('ColorSpaces (rgb, hsv, lab, ycrcb)', mosaic)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return

def task2(path_image):
    #TODO
    return

def task3_1(inputimage_domain, result_image):
    blur_img = cv2.GaussianBlur(inputimage_domain, (3, 3), 0)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(blur_img, scaleFactor=1.2, minNeighbors=10, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    return result_image

def task3(path_image):
    #TODO
    return


def draw_rectangle(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param['drawing'] = True
        param['x_start'], param['y_start'] = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if param['drawing']:
            param['x_end'], param['y_end'] = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        param['drawing'] = False
        param['x_end'], param['y_end'] = x, y


if __name__ == "__main__":
    #task0(path_image='_color.jpg')
    #task1(path_image='_color.jpg')
    task2(path_image="_low_color.webp")
    task3(path_image="_faces.jpg")
