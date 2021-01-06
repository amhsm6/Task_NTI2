import helpers
import cv2
import random
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('model.h5')

def get_standatr_signs():
    """Функция, позволяющая получить стандартные изображения знаков для сравнения
    с оригинальными изображениями. Стандартные изображения знаков хранятся во внутреннем каталоге."""
    # стандартные изображения дорожных знаков
    a_unevenness = cv2.imread("data/standards/a_unevenness.jpg")
    a_unevenness = cv2.inRange(a_unevenness, (89, 91, 149), (255, 255, 255))
    a_unevenness = cv2.resize(a_unevenness, (64, 64))

    no_drive = cv2.imread("data/standards/no_drive.png")
    no_drive = cv2.inRange(no_drive, (89, 91, 149), (255, 255, 255))
    no_drive = cv2.resize(no_drive, (64, 64))

    no_entry = cv2.imread("data/standards/no_entry.jpg")
    no_entry = cv2.inRange(no_entry, (89, 91, 149), (255, 255, 255))
    no_entry = cv2.resize(no_entry, (64, 64))

    parking = cv2.imread("data/standards/parking.jpg")
    parking = cv2.inRange(parking, (0, 0, 0), (255, 0, 255))
    parking = cv2.resize(parking, (64, 64))

    pedistrain = cv2.imread("data/standards/pedistrain.jpg")
    pedistrain = cv2.inRange(pedistrain, (89, 91, 149), (255, 255, 255))
    pedistrain = cv2.resize(pedistrain, (64, 64))


    road_works = cv2.imread("data/standards/road_works.jpg")
    road_works = cv2.inRange(road_works, (89, 91, 149), (255, 255, 255))
    road_works = cv2.resize(road_works, (64, 64))

    stop = cv2.imread("data/standards/stop.jpg")
    stop = cv2.inRange(stop, (89, 91, 149), (255, 255, 255))
    stop = cv2.resize(stop, (64, 64))

    way_out = cv2.imread("data/standards/way_out.jpg")
    way_out = cv2.inRange(way_out, (89, 91, 149), (255, 255, 255))
    way_out = cv2.resize(way_out, (64, 64))

    standart_signs = {
        "a_unevenness": a_unevenness,
        "no_drive": no_drive,
        "no_entry": no_entry,
        "parking": parking,
        "pedistrain": pedistrain,
        "road_works": road_works,
        "stop": stop,
        "way_out": way_out
    }
    return standart_signs

def one_hot_encode(label):

    one_hot_encoded = []
    if label == "none":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 0]
    elif label == "pedistrain":
        one_hot_encoded = [1, 0, 0, 0, 0, 0, 0, 0]
    elif label == "no_drive":
        one_hot_encoded = [0, 1, 0, 0, 0, 0, 0, 0]
    elif label == "stop":
        one_hot_encoded = [0, 0, 1, 0, 0, 0, 0, 0]
    elif label == "way_out":
        one_hot_encoded = [0, 0, 0, 1, 0, 0, 0, 0]
    elif label == "no_entry":
        one_hot_encoded = [0, 0, 0, 0, 1, 0, 0, 0]
    elif label == "road_works":
        one_hot_encoded = [0, 0, 0, 0, 0, 1, 0, 0]
    elif label == "parking":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 1, 0]
    elif label == "a_unevenness":
        one_hot_encoded = [0, 0, 0, 0, 0, 0, 0, 1]

    return one_hot_encoded

def predict_label(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(1, 64, 64, 1)
    image = image.astype('float16')

    ans = []

    for x in model.predict(image)[0]:
        ans.append(round(x))
    
    image = image.reshape(64, 64)

    return ans