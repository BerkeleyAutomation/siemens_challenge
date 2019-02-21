import numpy as np
import os
import sys
import tensorflow as tf
import IPython
from matplotlib import pyplot as plt
from PIL import Image
import time as timer
import sys
import os
import cv2
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
sys.path.append('/home/autolab/Workspaces/michael_working/models/research')
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)


class Detector():

    def __init__(self, model_name, path_to_labels):
        sys.path.append("..")
        from object_detection.utils import ops as utils_ops

        # if tf.__version__ < '1.4.0':
        #     raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


        # What model to download.
        MODEL_NAME = model_name

        # Path to frozen detection graph. This is the actual model that is used for the object detection.
        PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
        # PATH_TO_CKPT = MODEL_NAME + '/graph.pbtxt'


        # List of the strings that is used to add correct label for each box.
        PATH_TO_LABELS = path_to_labels
        #PATH_TO_LABELS = os.path.join('data', path_to_labels)
        NUM_CLASSES = 600


        print("Reading graph...")
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
        print("Done reading graph.")

        label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
        self.categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
        self.category_index = label_map_util.create_category_index(self.categories)

        PATH_TO_TEST_IMAGES_DIR = 'test_images'
        TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(7, 8) ]

        def run_inference_for_single_image_optimized(image, sess):
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                        tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

            output_dict = sess.run(tensor_dict,
                                   feed_dict={image_tensor: np.expand_dims(image, 0)})

            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict[
                'detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]

            return output_dict

        def run_inference_for_single_image(image, graph):
            with graph.as_default():
                with tf.Session() as sess:
                    # Get handles to input and output tensors
                    ops = tf.get_default_graph().get_operations()
                    all_tensor_names = {output.name for op in ops for output in op.outputs}
                    tensor_dict = {}
                    for key in [
                            'num_detections', 'detection_boxes', 'detection_scores',
                            'detection_classes', 'detection_masks'
                    ]:
                        tensor_name = key + ':0'
                        if tensor_name in all_tensor_names:
                            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                    tensor_name)
                    if 'detection_masks' in tensor_dict:
                        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
                        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                        detection_masks_reframed = tf.cast(
                                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                        tensor_dict['detection_masks'] = tf.expand_dims(
                                detection_masks_reframed, 0)
                    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})

                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                            'detection_classes'][0].astype(np.uint8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]
                    if 'detection_masks' in output_dict:
                        output_dict['detection_masks'] = output_dict['detection_masks'][0]
            return output_dict

        self.run_inference_for_single_image = run_inference_for_single_image
        self.run_inference_for_single_image_optimized = run_inference_for_single_image_optimized

    def predict(self, image_np, thresh=.5, sess=None, file_number=None):
        # image = Image.open(image_path)
        IMAGE_SIZE = (6, 4)

        # image_np = load_image_into_numpy_array(image)

        image_np_expanded = np.expand_dims(image_np, axis=0)
        start_time = timer.time()

        # import ipdb; ipdb.set_trace()

        if sess is None:
            output_dict = self.run_inference_for_single_image(image_np, self.detection_graph)
        else:
            output_dict = self.run_inference_for_single_image_optimized(image_np,sess)
        end_time = timer.time()
        # print("final time: " + str(end_time - start_time))
        # Visualization of the results of a detection.
        thresh = 0.6
        vis_util.visualize_boxes_and_labels_on_image_array(image_np,
                                                           output_dict['detection_boxes'],
                                                           output_dict['detection_classes'],
                                                           output_dict['detection_scores'],
                                                           self.category_index,
                                                           instance_masks=output_dict.get('detection_masks'),
                                                           use_normalized_coordinates=True,
                                                           line_thickness=5,
                                                           min_score_thresh=thresh)

        # vis_util.visualize_boxes_and_labels_on_image_array(
        #         image_np,
        #         output_dict['detection_boxes'],
        #         output_dict['detection_classes'],
        #         output_dict['detection_scores'],
        #         self.category_index,
        #         instance_masks=output_dict.get('detection_masks'),
        #         use_normalized_coordinates=True,
        #         line_thickness=5,
        #         min_score_thresh=thresh)

        self.TARGET_DIR = '/home/benno/experiments/hsr/surface_decluttering/isolated_objects_v2/'
        plt.figure()
        plt.imshow(image_np)
        plt.savefig(self.TARGET_DIR + 'object_classification_' + str(file_number).zfill(3))
        plt.show()

        return output_dict, image_np

if __name__ == '__main__':
    model_path = 'main/output_inference_graph.pb'
    label_map_path = 'main/object-detection.pbtxt'
    det = Detector(model_path, label_map_path)

    path = 'debug_imgs/rgb_raw_0000.jpg'

    output_dict, image_np = det.predict(path)
    img = cv2.imread(path)
    boxes = format_bboxes(output_dict, img.shape)
    for b in boxes:
        print(b)
        img = draw_box(img, b)
    cv2.imwrite("debug_imgs/testing.png", img)
