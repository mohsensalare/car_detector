import uuid
import os
import cv2
from datetime import datetime, timedelta

# config section
START_POINT_LINT = (130, 250)
END_POINT_LINE = (700, 250)
LINE_COLOR = (255, 255, 255)
LINE_SIZE = 3
MIN_SIZE_CAR = (100, 100)

# output directory
directory = r'./output'

allow_ride = False

TRAFFIC_LIGHT_TIME = 3


def crop_image(image, x, y, w, h):
    """
    crop car area in image and return image
    :param image:
    :param x:
    :param y:
    :param w:
    :param h:
    :return:
    """
    # round pixel points
    x = (int(x / 10) * 10) - 10
    y = (int(y / 10) * 10) - 10
    w = (int(w / 10) * 10) + 10
    h = (int(h / 10) * 10) + 10
    if x < 0:
        x = 0
    if y < 0:
        x = 0
    return image[y:y + h, x:x + w]


# capture frames from a video
cap = cv2.VideoCapture('crop2.mp4')

# Trained XML classifiers describes some features of some object we want to detect
car_cascade = cv2.CascadeClassifier('haarcascade_car.xml')

# loop runs if capturing has been initialized.
start = datetime.now()


# to specified directory
os.chdir(directory)

while True:
    """
    main of the application
    """
    ret, frames = cap.read()

    if cv2.waitKey(30) == 27 or frames is None:
        break
    # control time traffic light and add traffic light color to frame
    if datetime.now() - start > timedelta(seconds=TRAFFIC_LIGHT_TIME):
        start = datetime.now()
        allow_ride = not allow_ride

    # find car if traffic light color is red
    if not allow_ride:
        # find cars whit haarcascade_car model
        cars = car_cascade.detectMultiScale(frames, scaleFactor=1.2, minNeighbors=2, minSize=MIN_SIZE_CAR,
                                            flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in cars:
            # create a rectangle around bad car's
            if y > START_POINT_LINT[1]:
                frames = cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cropped_image = crop_image(frames, x, y, w, h)
                if len(cropped_image) > 100:
                    filename = f'{uuid.uuid4()}.jpg'
                    cv2.imwrite(filename, cropped_image)

    if allow_ride:
        frames = cv2.circle(img=frames, center=(50, 50), radius=20, color=(0, 255, 0), thickness=-1)
    else:
        frames = cv2.circle(img=frames, center=(50, 50), radius=20, color=(0, 0, 255), thickness=-1)

    frames = cv2.line(frames, START_POINT_LINT, END_POINT_LINE, LINE_COLOR, LINE_SIZE)

    # show finally output
    # cv2.imshow('CAR DETECTION', frames[START_POINT_LINT[0]:START_POINT_LINT[1], :])
    # cv2.imshow('CAR DETECTION', frames[:, :])
    # cv2.waitKey(0)
    # cv2.imshow('CAR DETECTION', frames[250:, :])
    # cv2.waitKey(0)
    cv2.imshow('CAR DETECTION', frames)

# De-allocate any associated memory usage
cv2.destroyAllWindows()
