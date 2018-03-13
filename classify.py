import numpy as np
import os
import argparse
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import visualization_utils as vis_util
from flask import Flask
from flask import request
from urlparse import urlparse
import boto3
import io
import json

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

NUM_CLASSES = 90


def build_graph(graph_path):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    return detection_graph

# label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
# category_index = label_map_util.create_category_index(categories)
category_index = 0


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3)]

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)


def open_image(image_path):
    parsed = urlparse(image_path)
    if parsed == "":
        return Image.open(image_path)
    else:
        # use AWS_PROFILE & AWS_DEFAULT_REGION environments to change the profile used
        if parsed.scheme != 's3':
            raise ValueError('Only S3 URLs are supported')
        s3 = boto3.resource('s3', region_name='us-east-1')
        obj = s3.Object(parsed.netloc, parsed.path[1:])
        image_data = obj.get()['Body'].read()
        return Image.open(io.BytesIO(image_data))


def detect(detection_graph, image_path):
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            image = open_image(image_path)
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            # Actual detection.
            return sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})


def visualize_results(res, image_path):
    image = Image.open(image_path)
    # the array based representation of the image will be used later in order to prepare the
    # result image with boxes and labels on it.
    image_np = load_image_into_numpy_array(image)
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    # Actual detection.
    (boxes, scores, classes, num) = res
    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        agnostic_mode=True,
        use_normalized_coordinates=True,
        line_thickness=8)
    plt.figure(figsize=IMAGE_SIZE)
    plt.imshow(image_np)


def get_box(res):
    # only one class so hard coding it...
    (boxes, scores, classes, num) = res
    return {'score': scores[0][0].item(0), 'box': boxes[0][0].tolist()}


app = Flask(__name__)


@app.route('/classify', methods=['GET'])
def index():
    link = request.args.get('link')
    graph = app.config['GRAPH']
    res = detect(graph, link)
    box = get_box(res)
    return json.JSONEncoder().encode(box)


parser = argparse.ArgumentParser()
parser.add_argument('--model', dest='model', required=True)
parser.add_argument('--image', dest='image', help='comma delimited list of images (ignored if -s)')
parser.add_argument('-v', dest='visualize', action='store_true', help='visualize result')
parser.add_argument('-s', dest='service', action='store_true', help='as an HTTP service')


def main():
    args = parser.parse_args()
    image_path = args.image
    model_path = args.model
    visualize = args.visualize
    as_a_service = args.service

    graph = build_graph(model_path)
    if as_a_service:
        app.config['GRAPH'] = graph
        app.run(debug=True)
        return

    images = image_path.split(',')
    for image in images:
        res = detect(graph, image)
        box = get_box(res)
        json.dumps(box)

        if visualize:
            visualize_results(res, image)


if __name__ == '__main__':
    main()
    print('done')
