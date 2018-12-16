import os
import shutil

import time
from datetime import datetime

import boto3
from io import BytesIO
import matplotlib.image as mpimg

import cv2

from flask import Flask, request, render_template, send_from_directory
from rq import Queue, get_failed_queue
from redis import Redis
from rq.job import Job
from worker import conn

from restoration.restoration import run_restoration

app = Flask(__name__, template_folder = './static/html', static_folder='./static')

app_root = os.path.dirname(os.path.abspath(__file__))
images_filepath = os.path.join(app_root, 'images/')

# AWS_ACCESS_KEY = 'AKIAI67AZUVYFH54PNBA'
# AWS_ACCESS_SECRET_KEY = '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'
BUCKET = 'restauracao'

@app.route("/", methods=["GET"])
def index():
    # Limpa os arquivos da restauração anterior
    if os.path.exists(images_filepath):
        shutil.rmtree(images_filepath)

    # Cria a pasta para a nova restauração
    if not os.path.exists(images_filepath):
        os.makedirs(images_filepath)

    return render_template("upload.html")

@app.route("/", methods=["POST"])
def upload():

    images_filename = datetime.now().strftime('%Y%m%d%H%M%S')+'.jpg'

    app.logger.debug("Recebendo o arquivo...")
    upload = request.files.get('image_data')
    upload.save(os.path.join(images_filepath, images_filename))
    app.logger.debug('Imagem salva em ' + os.path.join(images_filepath, images_filename))
    app.logger.debug('OK')

    app.logger.debug('Salvando o arquivo no S3...')
    image = cv2.cvtColor(cv2.imread(os.path.join(images_filepath, images_filename)), cv2.COLOR_BGR2RGB)
    s3 = boto3.client('s3')
    s3.upload_file(os.path.join(images_filepath, images_filename), BUCKET, images_filename)
    app.logger.debug('OK')

    app.logger.debug("Começando a restauração...")
    q = Queue(connection=conn)
    job = q.enqueue(run_restoration, images_filepath, images_filename)
    app.logger.debug("OK")

    return job.id

@app.route("/carregando/<job_id>", methods=["GET"])
def carregando(job_id):
    app.logger.debug('job_id em carregando: ' + job_id)
    return render_template("loading.html", job_id=job_id)

@app.route("/status/<job_id>", methods=["GET"])
def status(job_id):
    # app.logger.debug('job_id recebido em status: ' + job_id)
    time.sleep(5)
    job = Job.fetch(job_id, connection=conn)
    job_status = job.status
    app.logger.debug('job_status obtido em status: ' + job_status)
    return job_status

@app.route("/exc_info/<job_id>", methods=["GET"])
def info(job_id):
    job = Job.fetch(job_id, connection=conn)
    job_info = job.exc_info
    return str(job_info)

@app.route("/resultados/<job_id>", methods=["GET"])
def resultados(job_id):
    job = Job.fetch(job_id, connection=conn)
    filename = job.result
    return render_template("complete_display_image.html", image_name_orig=filename, image_name_rest='cv2_'+filename)

@app.route('/upload/<filename>')
def send_image(filename):
    # app.logger.debug(filename)
    resource = boto3.resource('s3')
    bucket = resource.Bucket(BUCKET)
    image_object = bucket.Object(filename)
    image = mpimg.imread(BytesIO(image_object.get()['Body'].read()), 'jpg')
    cv2.imwrite(images_filepath + filename, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return send_from_directory("images", filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
