import os
import shutil
from uuid import uuid4
import cv2
from flask import Flask, request, render_template, send_from_directory
from restoration.restoration import Restoration

# __author__ = 'ibininja' (original template)

app = Flask(__name__)
# app = Flask(__name__, static_url_path="/static", static_folder='/static')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))
IMAGES_FILEPATH = os.path.join(APP_ROOT, 'images/')

@app.route("/")
def index():
    # Limpa os arquivos da restauração anterior
    if os.path.exists(IMAGES_FILEPATH):
        try:
            shutil.rmtree(IMAGES_FILEPATH)
        except:
            print('Dir Images exception')

    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    # Cria a pasta para a nova restauração
    if not os.path.exists(IMAGES_FILEPATH):
        try:
            os.makedirs(IMAGES_FILEPATH)
        except:
            print('Dir Images exception')

    # print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        print ("Recebendo o arquivo...", end = ' ')
        filename = upload.filename
        upload.save(os.path.join(IMAGES_FILEPATH, filename))
        print('OK')

        img_rest = Restoration().run_restoration(IMAGES_FILEPATH, filename)
        cv2.imwrite(IMAGES_FILEPATH + 'cv2_' + filename, cv2.cvtColor(img_rest, cv2.COLOR_BGR2RGB))

    return render_template("complete_display_image.html", image_name_orig=filename, image_name_rest='cv2_'+filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
