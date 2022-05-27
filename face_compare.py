import base64
import uuid
from typing import Dict, Any, Tuple
from urllib.parse import quote
from matplotlib import pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from numpy import asarray
from PIL import Image
import numpy as np
from keras_vggface.utils import preprocess_input
from keras_vggface.vggface import VGGFace
from scipy.spatial.distance import cosine
import cv2
from werkzeug import datastructures


def get_model_scores(faces):
    samples = asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    model = VGGFace(model='resnet50',
                    include_top=False,
                    input_shape=(224, 224, 3),
                    pooling='avg')
    return model.predict(samples)


def extract_face_from_image(image, required_size=(224, 224)):
    detector = MTCNN()
    faces = detector.detect_faces(image)
    face_images = []
    for face in faces:
        x1, y1, width, height = face['box']
        x2, y2 = x1 + width, y1 + height

        face_boundary = image[y1:y2, x1:x2]

        face_image = Image.fromarray(face_boundary)
        face_image = face_image.resize(required_size)
        face_array = asarray(face_image)
        face_images.append(face_array)
    return face_images


def highlight_faces(image, faces):
    plt.imshow(image)
    ax = plt.gca()
    for face in faces:
        x, y, width, height = face['box']
        face_border = Rectangle((x, y), width, height,
                                fill=False, color='red')
        ax.add_patch(face_border)
    plt.show()


def image_to_text(image: np.array):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    retval, buffer = cv2.imencode('.jpg', image)
    image_as_text = base64.b64encode(buffer)

    return image_as_text


def decode_and_resize_image(image: datastructures.FileStorage) -> Tuple[np.array, np.array]:
    image: np.array = np.fromstring(image.read(), np.uint8)
    image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def faces_compare(filestorage_image):
    image_uuid = uuid.uuid4().hex
    image = decode_and_resize_image(filestorage_image)
    image_as_text = image_to_text(image)

    extracted_face = extract_face_from_image(image)
    if len(extracted_face) == 2:
        model_scores = [get_model_scores([extracted_face[0]]), get_model_scores([extracted_face[1]])]
        recognition_confidence = cosine(model_scores[0], model_scores[1])
        image = cv2.resize(image, (20, 20))
        percent_recognition_confidence = 100 - recognition_confidence * 100

        if recognition_confidence <= 0.5:
            image_report: Dict[str, Any] = {'Лица идентичны:': 'Да'}
            image_report.update({'result_image': f'data:image/png;base64,{quote(image_as_text)}',
                                 'Идентификатор файла': image_uuid})
            return image_report
        else:
            image_report: Dict[str, Any] = {'Лица идентичны:': 'Нет'}
            image_report.update({'result_image': f'data:image/png;base64,{quote(image_as_text)}',
                                 'Идентификатор файла': image_uuid})
            return image_report, image