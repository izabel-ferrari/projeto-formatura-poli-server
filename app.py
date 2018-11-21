from flask import Flask, request, Response
import findLines
import cv2
import os
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './upload'

@app.route("/")
def hello_world():
    return render_template('account.html')

@app.route("/upload", methods=['POST'])
def upload_file():
	file = request.files['image']
	filename = file.filename
	filepath = app.config['UPLOAD_FOLDER']
	if not os.path.exists(filepath):
		os.makedirs(filepath)
	file.save(os.path.join(filepath, file.filename))
	img = cv2.imread(os.path.join(filepath, file.filename))
	canny = findLines.processImage(img)
	newLines, lines = findLines.getLines(canny)
	data = {}
	data['lines_coord'] = newLines
	json_data = json.dumps(data)

	return Response(json_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
