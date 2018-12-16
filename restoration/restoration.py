import os
import time

import shutil

import boto3
from io import BytesIO
import matplotlib.image as mpimg

import numpy as np
from matplotlib import pyplot as plt

import cv2

import restoration.test as test
import restoration.mask as mask
import restoration.roi as roi
import restoration.utils as utils

def run_restoration(img_filepath, img_filename):
    BUCKET = 'restauracao'

    # Imagem de entrada (sem extensão)
    img_name = img_filename[:-4]
    # Extensão da imagem de entrada
    img_extension = img_filename[-4:]

    # Diretórios da imagem original, arquivos temp. e restauração final
    images_dir = img_filepath
    interm_dir = './interm_files/'
    inpaint_dir = './inpaints/'
    neural_gym_logs = './neuralgym_logs'
    tf_logs = './tf_logs'
    checkpoint_dir = './model_logs/release_celeba_256/'

    # Verifica se existem os diretórios e cria os que não existem
    if not os.path.exists(images_dir):
        os.mkdir(images_dir)
    if not os.path.exists(interm_dir):
        os.mkdir(interm_dir)
    if not os.path.exists(inpaint_dir):
        os.mkdir(inpaint_dir)

    print('Validando a imagem de entrada...', end = ' ')
    resource = boto3.resource('s3')
    bucket = resource.Bucket(BUCKET)
    image_object = bucket.Object(img_filename)
    image = mpimg.imread(BytesIO(image_object.get()['Body'].read()), 'jpg')
    cv2.imwrite(os.path.join(img_filepath, img_filename), cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image = cv2.cvtColor(cv2.imread(os.path.join(img_filepath, img_filename)), cv2.COLOR_BGR2RGB)
    # image = utils.validate_input_image(images_dir + img_name + img_extension, img_extension)
    print('OK')

    print('Identificando o rosto na imagem...', end = ' ')
    x, y, w, h = roi.identify_face(image)
    # Corta a região do rosto
    face = image[y:y+h, x:x+w]
    # Redimensiona o rosto
    face = utils.resize_face(w, h, face)
    print('OK')

    print('Identificando os olhos na imagem...', end = ' ')
    true_eyes = roi.detect_eyes(face)
    print('OK')

    print('Criando a máscara para o rosto...', end = ' ')
    face_rect_mask, mixed = mask.get_rect_mask(face)
    face_mask = mask.get_mask(face)
    mask.remove_eyes_from_mask(face_mask, true_eyes)
    mask.remove_eyes_from_mask(mixed, true_eyes)
    mask.remove_eyes_from_mask(face_rect_mask, true_eyes)
    mask.remove_border_from_mask(face_rect_mask)
    # Salva a máscara do rosto em disco ##
    cv2.imwrite(interm_dir + img_name + '_face_rect_mask' + img_extension, face_rect_mask)
    print('OK')

    print('Restaurando o rosto com OpenCV...', end = ' ')
    face_inpaint = cv2.inpaint(face, mixed, 3, cv2.INPAINT_NS)
    # Salvando a restauração obtida em disco ##
    cv2.imwrite(interm_dir + img_name + '_face_mixed_opencv' + img_extension, cv2.cvtColor(face_inpaint, cv2.COLOR_BGR2RGB))
    print('OK')

    print('Restaurando o rosto com Generative Inpaint with Contextual Attention...', end = ' ')
    face_inpaint = test.run_inpaint(image = interm_dir + img_name + '_face_mixed_opencv' + img_extension,
                                     mask = interm_dir + img_name + '_face_rect_mask' + img_extension,
                                     output = interm_dir + img_name + '_face_rect_generative' + img_extension,
                                     checkpoint_dir = checkpoint_dir)

    print('OK')

    print('Restaurando o rosto com OpenCV com máscara de ridges...', end = ' ')
    # Gera a máscara de ridges para o rosto
    face_ridge_mask = mask.get_ridge_mask(face_inpaint)
    mask.remove_eyes_from_mask(face_ridge_mask, true_eyes)
    face_ridge_mask = face_ridge_mask & ~face_mask
    # Redimensiona o rosto restaurado para o tamanho original
    face_inpaint_redim = cv2.resize(face_inpaint, (w, h))
    face_mask_total = mixed | face_rect_mask | face_ridge_mask
    face_mask_total = cv2.resize(face_mask_total, (w, h))
    face_ridge_mask = cv2.resize(face_ridge_mask, (w, h))
    print('OK')

    print('Criando a máscara do fundo da imagem...', end = ' ')
    image_mask = mask.get_mask(image)
    mask.remove_face_from_mask(image_mask, x, y, w, h)
    print('OK')

    print('Restaurando o fundo da imagem com OpenCV...', end = ' ')
    image_inpaint = cv2.inpaint(image, image_mask, 3, cv2.INPAINT_NS)
    print('OK')

    print('Reconstruindo a imagem completa...', end = ' ')
    image_restored = np.copy(image_inpaint)
    image_restored[y:y+h, x:x+w][face_mask_total > 0] = face_inpaint_redim[face_mask_total > 0]
    print('OK')

    print('Criando a máscara de ridges para a imagem completa...', end = ' ')
    #Gera a máscara de ridges
    ridge_mask = mask.get_ridge_mask(image_inpaint)
    ridge_mask = ridge_mask & ~image_mask
    ridge_mask[y:y+h, x:x+w] = face_ridge_mask
    print('OK')

    print('Restauração da imagem completa com OpenCV com máscara de ridges...', end = ' ')
    image_final = cv2.inpaint(image_restored, ridge_mask, 3, cv2.INPAINT_NS)
    cv2.imwrite(img_filepath + 'cv2_' + img_filename, cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB))
    print('OK')

    # Exclui os diretórios de logs
    if os.path.exists(interm_dir):
        shutil.rmtree(interm_dir)
    if os.path.exists(inpaint_dir):
        shutil.rmtree(inpaint_dir)
    if os.path.exists(neural_gym_logs):
        shutil.rmtree(neural_gym_logs)
    if os.path.exists(tf_logs):
        shutil.rmtree(tf_logs)

    print('Fazendo o upload da imagem para o S3...')
    s3 = boto3.client('s3')
    s3.upload_file(img_filepath + 'cv2_' + img_filename, 'restauracao', 'cv2_' + img_filename)

    return img_filename
