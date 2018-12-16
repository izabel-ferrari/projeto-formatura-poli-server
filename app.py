import os
import shutil
from uuid import uuid4
import cv2
from datetime import datetime
from flask import Flask, request, render_template, send_from_directory
from restoration.restoration import Restoration
from rq import Queue
from worker import conn
from utils import count_words_at_url

# __author__ = 'ibininja' (original template)

app = Flask(__name__, template_folder = './static/html', static_folder='./static')
# app = Flask(__name__, static_url_path="/static", static_folder='/static')

app_root = os.path.dirname(os.path.abspath(__file__))
images_filepath = os.path.join(app_root, 'images/')

@app.route("/", methods=["GET"])
def index():
    q = Queue(connection=conn)
    result = q.enqueue(count_words_at_url, 'http://heroku.com')
    
    # app.logger.debug('GET')
    # # Limpa os arquivos da restauração anterior
    # if os.path.exists(images_filepath):
    #     try:
    #         shutil.rmtree(images_filepath)
    #     except:
    #         print('Dir Images exception')
    #
    # # Cria a pasta para a nova restauração
    # if not os.path.exists(images_filepath):
    #     try:
    #         os.makedirs(images_filepath)
    #     except:
    #         print('Dir Images exception')

    return render_template("upload.html")

@app.route("/", methods=["POST"])
def upload():
    app.logger.debug('POST')
    images_filename = datetime.now().strftime('%Y%m%d-%H%M%S')+'.jpg'

    app.logger.debug("Recebendo o arquivo...")
    upload = request.files.get('image_data')

    while not upload:
        upload = request.files.get('image_data')

    upload.save(os.path.join(images_filepath, images_filename))
    app.logger.debug('OK')

    app.logger.debug("Começando a restauração...")
    img_rest = Restoration().run_restoration(images_filepath, images_filename)
    app.logger.debug("OK")

    app.logger.debug("Salvando as imagens em disco...")
    cv2.imwrite(images_filepath + 'cv2_' + images_filename, cv2.cvtColor(img_rest, cv2.COLOR_BGR2RGB))
    app.logger.debug("OK")

    return images_filename

@app.route("/resultados/<filename>", methods=["GET"])
def resultados(filename):
    return render_template("complete_display_image.html", image_name_orig=filename, image_name_rest='cv2_' + filename)

@app.route('/upload/<filename>')
def send_image(filename):
    app.logger.debug(filename)
    return send_from_directory("images", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
