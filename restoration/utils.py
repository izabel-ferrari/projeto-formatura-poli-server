import cv2

FACE_SIZE = 256

def validate_input_image(IMG_FILEPATH, IMG_EXTENSION):
	try:
		image = cv2.cvtColor(cv2.imread(IMG_FILEPATH), cv2.COLOR_BGR2RGB)
		height, width, channels = image.shape

		# Verifica se a imagem está no formato JPG
		if IMG_EXTENSION not in ['.JPG', '.jpg', '.JPEG', '.jpeg']:
			raise TypeError('A imagem de entrada deve estar no formato .jpg, encontrado {}'.format(IMG_EXTENSION))

		# Verifica se a imagem tem tamanho mínimo de 256 x 256
		elif (height or width) < 256:
			raise TypeError('A imagem de entrada deve ter pelo menos 256 x 256 px, encontrado {} x {} px'.format(height, width))

		return image
	except:
		raise

def resize_face(w, h, image_cropped):
	if (w != h):
		# print('Não temos um quadrado')
		# A menor dimensão será substituída pela maior dimensão
		if w < h:
			w = h
		else:
			h = w

	if (w == FACE_SIZE) and (h == FACE_SIZE):
			# print('Rosto no tamanho adequado')
			return image_cropped

	if (w < FACE_SIZE) and (h < FACE_SIZE):
		# print('Redimensionando para cima com INTER_CUBIC')
		return cv2.resize(image_cropped, (FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_CUBIC)

	if (w > FACE_SIZE) and (h > FACE_SIZE):
		# print('Redimensionando para baixo com INTER_AREA')
		return cv2.resize(image_cropped, (FACE_SIZE, FACE_SIZE), interpolation = cv2.INTER_AREA)

	return None
