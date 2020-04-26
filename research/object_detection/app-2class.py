'''
The detection code is partially derived and modified from the object_detection_tutorial.ipynb.
The original author should be honoured:

"Speed/accuracy trade-offs for modern convolutional object detectors."
Huang J, Rathod V, Sun C, Zhu M, Korattikara A, Fathi A, Fischer I, Wojna Z,
Song Y, Guadarrama S, Murphy K, CVPR 2017
'''


from flask import Flask, request, render_template, redirect

import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util

app = Flask(__name__, template_folder='')
from datetime import timedelta
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)  # avoid caching, which prevent showing the detection/splash result

import os
import sys
import random


# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Directory to save logs and trained model
CKPT_DIR = os.path.join(ROOT_DIR, "research/object_detection/data/faster_RCNN_banana_and_pear/frozen_inference_graph.pb")
LABEL_DIR = os.path.join(ROOT_DIR, "research/object_detection/data/faster_RCNN_banana_and_pear/fruit_labelmap.pbtxt")

IMAGE_DIR = os.path.join(ROOT_DIR, "research/object_detection/static/images/")

UPLOAD_FOLDER = os.path.join(ROOT_DIR, "research/object_detection/upload_images")
ALLOWED_EXTENSIONS = set(['jpg'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# avoid caching, which prevent showing the detection/splash result
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = timedelta(seconds=1)
class TOD(object):
    def __init__(self):
        self.PATH_TO_CKPT = CKPT_DIR
        self.PATH_TO_LABELS = LABEL_DIR
        self.NUM_CLASSES = 2
        self.detection_graph = self._load_model()
        self.category_index = self._load_label_map()

    # load the pre-trained model via the frozen inference graph
    def _load_model(self):
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        return detection_graph

    # load the label map so that we know what object has been detected
    def _load_label_map(self):
        label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
        categories = label_map_util.convert_label_map_to_categories(label_map,
                                                                    max_num_classes=self.NUM_CLASSES,
                                                                    use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return category_index

    def detect(self, image):
        with self.detection_graph.as_default():
            with tf.Session(graph=self.detection_graph) as sess:
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image, axis=0)
                image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
                boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
                scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
                classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
                num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
                # Actual detection.
                (boxes, scores, classes, num_detections) = sess.run(
                    [boxes, scores, classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})
                # Visualization of the results of a detection.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    image,
                    np.squeeze(boxes),
                    np.squeeze(classes).astype(np.int32),
                    np.squeeze(scores),
                    self.category_index,
                    use_normalized_coordinates=True,
                    line_thickness=8)

        # cv2.namedWindow("detection", cv2.WINDOW_NORMAL)
        cv2.imwrite(os.path.join(IMAGE_DIR , 'detection_result.jpg'), image)
        cv2.waitKey(0)

################################################################
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS
def run_detection():
    user_file_names = next(os.walk(UPLOAD_FOLDER))[2]
    names_chosen = random.choice(user_file_names)
    image = cv2.imread(os.path.join(UPLOAD_FOLDER, names_chosen))
    print('\n-----------------', len([image]), '---------------\n')

    detecotr = TOD()
    detecotr.detect(image)

@app.route('/')
def home():
    if request.method == 'GET':
        return render_template('index.html')

    return render_template('index.html')

@app.route('/UploadDetect', methods=['GET', 'POST'])
def upload_file_detect():
    if request.method == 'GET':
        return render_template('upload_detect.html')

    if request.method == 'POST':
        f = request.files['file']
        print(request.files)
        if f and allowed_file(f.filename):
            f.save(os.path.join(app.config['UPLOAD_FOLDER'], 'uploaded_image.jpg'))
            return redirect('/detect')
        else:
            print('file type is not correct')
            return render_template('upload_detect.html')

@app.route('/detect')
def detect():
    run_detection()
    return render_template('result_detect.html')

'''
Main function to run Flask server
'''
if __name__ == '__main__':
    app.run()
