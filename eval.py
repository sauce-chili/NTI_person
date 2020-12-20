import cv2
import numpy as np
# модуль для детектирования

# если вы используте модель для детектирования добавьте ее здесь:
#################################################################

CLASSES = {15: "person"}

protxt = "MobileNetSSD_deploy.prototxt.txt"
model = "MobileNetSSD_deploy.caffemodel"
net = cv2.dnn.readNetFromCaffe(protxt, model)


#################################################################


def detector(rgb_image):
    """Детектирование пешеходов на изображении:
    Для каждого изображения формируется массив, состоящий из координат углов объектов(xmin,ymin, xmax, ymax) - координаты левого верхнего угла и
    правого нижнего угла объекта
    rects = [[335, 184, 384, 267]]
    """
    # height = int(rgb_image.shape[0])
    # frame = cv2.resize(rgb_image, (400, height))
    # frame = imutils.resize(rgb_image, width=400)
    # using a greyscale picture, also for faster detection
    frame = rgb_image
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()
    rects = []
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.32:
            idx = int(detections[0, 0, i, 1])
            if idx == 15:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (xMin, yMin, xMax, yMax) = box.astype("int")
                rects.append([xMin, yMin, xMax, yMax])

    return rects
