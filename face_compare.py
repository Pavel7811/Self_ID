import base64
from typing import Dict, Any
from urllib.parse import quote
from mtcnn.mtcnn import MTCNN
from numpy import asarray
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import cv2


# работа с моделью
def get_model_scores(faces):
    # преобразование в массив
    samples = asarray(faces, 'float32')
    # подготовка лица для модели
    samples = preprocess_input(samples, version=2)
    # создание модели
    # модель принимает на вход цветные изображения лиц размером 244×244
    model = VGGFace(model='vgg16',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    # получение оценки модели
    return model.predict(samples)


# функция извлечения лиц с фотографии
def extract_face_from_image(image, required_size=(224, 224)):
    detector = MTCNN()
    # обнаружение лиц с помощью детектора из библиотеки MTCNN
    faces = detector.detect_faces(image)
    face_images = []
    for face in faces:
        # вычисление начальных координат "рамки" лица
        x1, y1, width, height = face['box']
        # вычисление конечных координат "рамки" лица
        x2, y2 = x1 + width, y1 + height
        # получение изображений лиц в данных рамках
        face_boundary = image[y1:y2, x1:x2]
        # Форматирование изображений в формат PIL
        face_image = Image.fromarray(face_boundary)
        # изменение размера изображений лиц
        face_image = face_image.resize(required_size)
        # преобразование в массив
        face_array = asarray(face_image)
        # Добавление в лист изображения лиц
        face_images.append(face_array)
    # возврат листа с изображениями лиц обрезанными до 224, 224
    # и координаты лиц
    return face_images, faces


# построение рамки на лицах
def highlight_faces(image, faces):
    for face in faces:
        # извлечение координат лиц
        x, y, w, h = face['box']
        # построение рамок
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return image


# преобразование изображения в текст
def image_to_text(image: np.array):
    retval, buffer = cv2.imencode('.jpg', image)
    image_as_text = base64.b64encode(buffer)
    return image_as_text


# декодировка полученного изображения в RGB
def decode_and_resize_image(image):
    np_image = np.fromfile(image, np.uint8)
    image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
    return image


def faces_compare(filestorage_image):
    # подготовка изображения
    image = decode_and_resize_image(filestorage_image)
    # поиск лиц на фото
    extracted_face, faces = extract_face_from_image(image)
    # отрисовка рамок
    image = highlight_faces(image, faces=faces)
    # обрезка отображаемого изображения
    scale_percent = 20
    width = int(image.shape[1] * scale_percent/100)
    height = int(image.shape[0] * scale_percent/100)
    dsize = (width, height)
    image = cv2.resize(image, dsize)
    # преобразование изображения в текст для отображения
    image_as_text = image_to_text(image)
    # если на фото найдено два лица
    if len(extracted_face) == 2:
        # создание числовых характеристик для каждого лица
        model_scores = [get_model_scores([extracted_face[0]]), get_model_scores([extracted_face[1]])]
        # сравнение косинусного расстояния
        # возвращает оценку, пороговое значение
        recognition_confidence = 1 - cosine(model_scores[0], model_scores[1])
        print(recognition_confidence)
        # вывод данных в json формате
        if recognition_confidence >= 0.65:
            image_report: Dict[str, Any] = {'Лица идентичны:': 'Да'}
            image_report.update({'result_image': f'data:image/png;base64,{quote(image_as_text)}'})
            return image_report
        else:
            image_report: Dict[str, Any] = {'Лица идентичны:': 'Нет'}
            image_report.update({'result_image': f'data:image/png;base64,{quote(image_as_text)}'})
            return image_report
    else:
        image_report: Dict[str, Any] = {'Лица идентичны:': 'Не найдено'}
        return image_report
