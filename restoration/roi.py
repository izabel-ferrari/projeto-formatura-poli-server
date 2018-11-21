import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

def identify_face(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    try:
        for minNeighbors in range(0, 50, 10):
            # Itera até encontrar apenas um rosto ou até chegar no limite dos vizinhos
            image_faces = face_cascade.detectMultiScale(image_gray, 1.05, minNeighbors)
            if len(image_faces) == 1:
                break

        if len(image_faces) == 0:
            # Itera até encontrar apenas dois rostos
            for minNeighbors in range(0, 50, 10):
                image_faces = face_cascade.detectMultiScale(image_gray, 1.05, minNeighbors)
                if len(image_faces) == 2:
                    break

        if len(image_faces) > 1:
            curr_face = 0
            best_face = [[0, 0, 0, 0]]

            for (x,y,w,h) in image_faces:
                if (w**2 + h**2)**0.5 > curr_face:
                    best_face[0] = [x, y, w, h]
                    curr_face = (w**2 + h**2)**0.5

            image_faces = best_face
        grid = 8
        x, y, w, h = image_faces[0]
        w = w//grid*grid
        h = h//grid*grid
        return [x, y, w, h]
    except:
        x = 0
        y = 0
        h, w, p = image.shape

        if w != h:
            min_size = min(w,h)
            if w == min_size:
                y = int(y + (h-w)/2)
                h = w
            elif h == min_size:
                x = int(x + (w-h)/2)
                w = h
        return [x, y, w, h]

def detect_eyes(face):
    face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face_eyes = eye_cascade.detectMultiScale(face_gray)

    if len(face_eyes) > 2:
        true_eyes = [[0, 0, 0, 0], [0, 0, 0, 0]]
        dim_eyes = [0, 0]

        for face_eye in face_eyes:
            curr_eye = int(((face_eye[2])**2 + (face_eye[3])**2)**(0.5))

            if curr_eye > max(dim_eyes):
                dim_eyes[dim_eyes.index(min(dim_eyes))] = dim_eyes[dim_eyes.index(max(dim_eyes))]
                dim_eyes[dim_eyes.index(max(dim_eyes))] = curr_eye
                true_eyes[1] = true_eyes[0]
                true_eyes[0] = face_eye

            elif curr_eye > min(dim_eyes):
                dim_eyes[dim_eyes.index(min(dim_eyes))] = curr_eye
                true_eyes[1] = face_eye

    else:
        true_eyes = face_eyes

    return true_eyes
