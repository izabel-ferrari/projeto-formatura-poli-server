import os
import shutil

import numpy as np
from matplotlib import pyplot as plt
import cv2

import restoration.test as test
import restoration.mask as mask
import restoration.roi as roi
import restoration.utils as utils

import time

class Restoration:

    def run_restoration(self, img_filepath, img_filename):
        # %%time
        # Imagem de entrada (sem extensão)
        img_name = img_filename[:-4] #'IMG_6621 (1)'

        # Extensão da imagem de entrada
        img_extension = img_filename[-4:] #'.jpg'

        # %%time
        # Diretórios da imagem original, arquivos temp. e restauração final
        images_dir = img_filepath
        interm_dir = './interm_files/'
        inpaint_dir = './inpaints/'

        neural_gym_logs = './neuralgym_logs'
        tf_logs = './tf_logs'

        checkpoint_dir = './model_logs/release_celeba_256/'

        # Verifica se existem os diretórios e cria os que não existem
        if not os.path.exists(interm_dir):
            os.mkdir(interm_dir)
        if not os.path.exists(inpaint_dir):
            os.mkdir(inpaint_dir)

        # %%time
        print('Validando a imagem de entrada...', end = ' ')
        image = utils.validate_input_image(images_dir + img_name + img_extension, img_extension)

        # Salva a imagem original redimensionada em disco
        cv2.imwrite(images_dir + img_name + img_extension, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        print('OK')

        # plt.figure()
        # plt.imshow(image)

        print('Identificando o rosto na imagem...', end = ' ')
        x, y, w, h = roi.identify_face(image)
        print('OK')

        # Cria o rosto identificado
        image_copy = np.copy(image)
        cv2.rectangle(image_copy, (x,y), (x+w,y+h), (255,0,0), 4)

        # Corta a região do rosto
        image_cropped = image[y:y+h, x:x+w]

        # Salva a região do rosto em disco
        cv2.imwrite(interm_dir + img_name + '_cropped' + img_extension, cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB))

        # Redimensiona o rosto
        face = utils.resize_face(w, h, image_cropped)

        # Salva o rosto redimensionado em disco
        cv2.imwrite(interm_dir + img_name + '_cropped_redim' + img_extension, cv2.cvtColor(face, cv2.COLOR_BGR2RGB))

        # Exibe as imagens salvas
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (10, 5))
        # imgs[0].set_title("Rosto identificado")
        # imgs[0].imshow(image_copy)

        # imgs[1].set_title("Rosto recortado")
        # imgs[1].imshow(image_cropped)


        # imgs[2].set_title("Rosto redimensionado")
        # imgs[2].imshow(face)
        # plt.show()

        print('Identificando os olhos...', end = ' ')
        true_eyes = roi.detect_eyes(face)
        print('OK')

        # Exibe os olhos identificados no rosto
        face_copy = np.copy(face)
        for (ex, ey, ew, eh) in true_eyes:
            cv2.ellipse(face_copy,(int(ex+0.5*ew), int(ey+0.5*eh)),(int(ew/2),int(eh/4)),0,0,360,(0, 255, 0),2)
        # plt.figure()
        # plt.title("Olhos identificados")
        # plt.imshow(face_copy)

        print('Criando a máscara para o rosto...', end = ' ')

        face_rect_mask, mixed = mask.get_rect_mask(face)
        face_mask = mask.get_mask(face)

        mask.remove_eyes_from_mask(face_mask, true_eyes)
        mask.remove_eyes_from_mask(mixed, true_eyes)
        mask.remove_eyes_from_mask(face_rect_mask, true_eyes)

        mask.remove_border_from_mask(face_rect_mask)

        print('OK')

        # Salva a máscara do rosto em disco
        cv2.imwrite(interm_dir + img_name + '_face_rect_mask' + img_extension, face_rect_mask)

        # Salva a máscara do rosto em disco
        cv2.imwrite(interm_dir + img_name + '_face_mixed_mask' + img_extension, mixed)

        # Exibe as maścaras geradas
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (12, 4))
        #
        # imgs[0].imshow(face_mask, cmap='gray')
        # imgs[0].set_title("Máscara Inicial")
        #
        # imgs[1].imshow(face_rect_mask, cmap='gray')
        # imgs[1].set_title("Máscara retangularizada")
        #
        # imgs[2].imshow(mixed, cmap='gray')
        # imgs[2].set_title("Máscara isolada")
        #
        # plt.show()

        print('OK')

        print('Verificando a porcentagem de dano no rosto...', end = ' ')
        sem_dano = sum(sum(face_rect_mask == 0))
        porcentagem_dano = 1-sem_dano/utils.FACE_SIZE**2
        print(str(round(porcentagem_dano, 2)*100) + '% de dano no rosto.')

        print('Restaurando o rosto com OpenCV...', end = ' ')
        face_inpaint = cv2.inpaint(face, mixed, 3, cv2.INPAINT_NS)

        # Salvando a restauração obtida em disco
        cv2.imwrite(interm_dir + img_name + '_face_mixed_opencv' + img_extension, cv2.cvtColor(face_inpaint, cv2.COLOR_BGR2RGB))
        print('OK')

        # Exibe as etapas da restauração
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (12, 4))

        # imgs[0].imshow(face)
        # imgs[0].set_title("Rosto danificado")

        # imgs[1].imshow(mixed, cmap='gray')
        # imgs[1].set_title("Máscara isolada")
        #
        # imgs[2].imshow(face_inpaint)
        # imgs[2].set_title("Rosto restaurado com o opencv")
        #
        # plt.show()

        print('Restaurando o rosto com Generative Inpaint with Contextual Attention...', end = ' ')
        face_inpaint = test.run_inpaint(image = interm_dir + img_name + '_face_mixed_opencv' + img_extension,
                                         mask = interm_dir + img_name + '_face_rect_mask' + img_extension,
                                         output = interm_dir + img_name + '_face_rect_generative' + img_extension,
                                         checkpoint_dir = checkpoint_dir)

        # Salvando a restauração obtida em disco
        cv2.imwrite(interm_dir + img_name + '_face_rect_generative' + img_extension, cv2.cvtColor(face_inpaint, cv2.COLOR_BGR2RGB))

        # Salvando a restauração obtida em disco
        cv2.imwrite(interm_dir + img_name + '_face_inpainting' + img_extension, cv2.cvtColor(face_inpaint, cv2.COLOR_BGR2RGB))

        print('OK')


        #Exibe as etapas da restauração
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (12, 4))
        #
        # imgs[0].imshow(face_inpaint)
        # imgs[0].set_title("Rosto danificado")
        #
        # imgs[1].imshow(face_rect_mask, cmap='gray')
        # imgs[1].set_title("Máscara retangular")
        #
        # imgs[2].imshow(face_inpaint)
        # imgs[2].set_title("Rosto restaurado com a rede")
        #
        # plt.show()

        # Exibe a restauração obtida
        # plt.figure()
        # plt.title("Restauração do rosto")
        # plt.imshow(face_inpaint)

        #Gera a máscara de rigdges para o rosto
        face_ridge_mask = mask.get_ridge_mask(face_inpaint)
        mask.remove_eyes_from_mask(face_ridge_mask, true_eyes)
        face_ridge_mask = face_ridge_mask & ~face_mask

        # Redimensiona o rosto restaurado para o tamanho original
        face_inpaint_redim = cv2.resize(face_inpaint, (w, h))
        face_mask_total = mixed | face_rect_mask | face_ridge_mask

        face_mask_total = cv2.resize(face_mask_total, (w, h))
        face_ridge_mask = cv2.resize(face_ridge_mask, (w, h))

        print('Criando a máscara do fundo da imagem...', end = ' ')

        image_mask = mask.get_mask(image)
        mask.remove_face_from_mask(image_mask, x, y, w, h)
        print('OK')

        # Salva a máscara da imagem em disco
        cv2.imwrite(interm_dir + img_name + '_image_mask' + img_extension, image_mask)

        print('Restaurando o fundo da imagem...', end = ' ')

        image_inpaint = cv2.inpaint(image, image_mask, 3, cv2.INPAINT_NS)
        print('OK')

        #Gera a máscara de ridges
        ridge_mask = mask.get_ridge_mask(image_inpaint)
        ridge_mask = ridge_mask & ~image_mask
        ridge_mask[y:y+h, x:x+w] = face_ridge_mask

        # Salva a restauração do fundo da imagem em disco
        cv2.imwrite(interm_dir + img_name + '_image_inpainting' + img_extension, cv2.cvtColor(image_inpaint, cv2.COLOR_BGR2RGB))

        # Exibe as imagens geradas
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (12, 6))
        #
        # imgs[0].imshow(image)
        # imgs[0].set_title("Imagem original")
        #
        # imgs[1].imshow(image_mask, cmap='gray')
        # imgs[1].set_title("Máscara do fundo")
        #
        # imgs[2].imshow(image_inpaint)
        # imgs[2].set_title("Restauração do fundo")

        print('Reconstruindo a imagem completa...', end = ' ')
        image_restored = np.copy(image_inpaint)
        image_restored[y:y+h, x:x+w][face_mask_total > 0] = face_inpaint_redim[face_mask_total > 0]
        print('OK')

        # Exibe a restauração obtida
        # plt.figure()
        # plt.title("Restauração total")
        # plt.imshow(image_restored)

        #Restauração dos ridges
        print('Restauração final...', end=' ')
        image_final = cv2.inpaint(image_restored, ridge_mask, 3, cv2.INPAINT_NS)
        print('OK')

        # Exibe as imagens geradas
        # fig, imgs = plt.subplots(nrows=1, ncols=3, figsize = (12, 6))
        #
        # imgs[0].imshow(image_restored)
        # imgs[0].set_title("Imagem restaurada")
        #
        # imgs[1].imshow(ridge_mask, cmap='gray')
        # imgs[1].set_title("Máscara de ridges")
        #
        # imgs[2].imshow(image_final)
        # imgs[2].set_title("Restauração final")

        # Salva a restauração da imagem em disco
        cv2.imwrite(inpaint_dir + img_name + '_inpainting' + img_extension, cv2.cvtColor(image_final, cv2.COLOR_BGR2RGB))

        # Exibe a imagem restaurada
        # plt.figure(figsize=(10,10))
        # plt.imshow(image_final)
        # plt.figure(figsize=(10,10))
        # plt.imshow(image)

        # Exclui os diretórios de logs

        if os.path.exists(interm_dir):
            shutil.rmtree(interm_dir)

        if os.path.exists(inpaint_dir):
            shutil.rmtree(inpaint_dir)

        if os.path.exists(neural_gym_logs):
            shutil.rmtree(neural_gym_logs)

        if os.path.exists(tf_logs):
            shutil.rmtree(tf_logs)

        return(image_final)
