from FaceDetector import Detector

import cv2 as cv
from flask import Flask, render_template, request, redirect, flash, url_for
import os
from werkzeug.utils import secure_filename

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

app = Flask(__name__)
detector = Detector()

UPLOAD_FOLDER = 'static/detection_result/'
app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# run inference using detectron2


def run_inference(img):
    # run inference using detectron2
    result_img = detector.inference(img)

    return result_img


@app.route("/")
@app.route("/index")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)

        file = request.files['file']
        if file.filename == '':
            flash('No image selected for uploading !')
            return redirect(request.url)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            img = cv.imread(os.path.join(UPLOAD_FOLDER, filename))
            img_pred = run_inference(img)

            img_path = UPLOAD_FOLDER+"/"+filename
            cv.imwrite(str(img_path), img_pred)

            return render_template('index.html', filename=filename)
        else:
            flash('Allowed image types are - png, jpg, jpeg !')
            return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='detection_result/' + filename), code=301)


if __name__ == "__main__":
    app.run(debug=True)
