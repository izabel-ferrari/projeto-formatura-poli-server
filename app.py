import os
from uuid import uuid4
import cv2
from flask import Flask, request, render_template, send_from_directory
from restoration.restoration import Restoration

# __author__ = 'ibininja' (original template)

app = Flask(__name__)
# app = Flask(__name__, static_folder="images")

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def upload():
    filepath = os.path.join(APP_ROOT, 'images/')
    # print(filepath)
    if not os.path.exists(filepath):
        try:
            os.makedirs(filepath)
            print('Diretório criado')
        except:
            print('Não foi possível criar o diretório')
    # print(request.files.getlist("file"))
    for upload in request.files.getlist("file"):
        # print(upload)
        print("Nome do arquivo: {}".format(upload.filename))

        filename = upload.filename
        print ("Salvando em:", os.path.join(filepath, filename))
        upload.save(os.path.join(filepath, filename))

        img_rest = Restoration().run_restoration(filepath, filename)
        cv2.imwrite(filepath + 'cv2_' + filename, cv2.cvtColor(img_rest, cv2.COLOR_BGR2RGB))

    return render_template("complete_display_image.html", image_name_orig=filename, image_name_rest='cv2_'+filename)

@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
