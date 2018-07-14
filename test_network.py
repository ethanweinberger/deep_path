import argparse
import sys
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import auc

from collections import namedtuple
ROC_Point = namedtuple("ROC_Point", ["true_positive_rate", "false_positive_rate"])

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

def read_tensor_from_image_file(file_name,
                                input_height=224,
                                input_width=224):
    input_name = "file_reader"
    file_reader = tf.read_file(file_name, input_name)
    if not (file_name.endswith(".jpg") or file_name.endswith(".jpeg")):
       raise IOError("Images must be provided in jpeg format!") 

    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")

    image_as_float = tf.image.convert_image_dtype(image_reader, tf.float32)
    image_4d = tf.expand_dims(image_as_float, 0) 
    resize_shape = tf.stack([input_height, input_width])
    resized_shape_as_int = tf.cast(resize_shape, dtype=tf.int32)
    resized_image = tf.image.resize_bilinear(image_4d, resized_shape_as_int)
   
    return resized_image
    
def classify_whole_slides(percent_for_positive_classification):
    """
    Runs all the patches for each test slide as specified in test_slide_list 
    through the trained network.  If a given slide has more than
    `percent_for_positive_classification` slides classified as HER2+, then
    we classify the slide as HER2+

    Args:
        percent_for_positive_classification (float): Float representation
            of the patch percentage we need classed as positive to classify
            an entire slide as HER2+
    Returns:
        roc_point (ROC_Point): Named tuple containing the true and false positive
            rates for our classifier with the specified parameter
    """

    graph = load_graph(args.model_file)
    input_name = "import/" + args.input_layer
    output_name = "import/" + args.output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    df = pd.read_csv(args.label_file)
    
    with tf.Session(graph = graph) as sess:
        with open(args.test_slide_list, "rb") as fp:
            test_slide_list = pickle.load(fp)
        
        slides_classified_correctly = 0
        slides_examined = 0
        num_positives_classified_correctly = 0
        num_false_positives = 0

        num_true_positives = 0
        num_true_negatives = 0
        for test_slide in test_slide_list:
            slide_name = os.path.basename(test_slide)
            slide_label_row = df.loc[df.specnum_formatted == slide_name]
            if slide_label_row.empty:
                continue

            slide_label = slide_label_row.iloc[0].her2_ihc
            if slide_label == 0.0 or slide_label == 1.0:
                slide_category = "neg"
            elif slide_label == 2.0 or slide_label == 3.0:
                slide_category = "pos"
            else:
                continue
 
            num_neg_votes = 0
            num_pos_votes = 0
            for file_name in os.listdir(test_slide):
                img_tensor = read_tensor_from_image_file(os.path.join(test_slide, file_name))
                img = sess.run(img_tensor)
                results = sess.run(output_operation.outputs[0], {
                    input_operation.outputs[0]: img 
                })
                results = np.squeeze(results)
                if results[0] > results[1]:
                    num_neg_votes += 1
                else:
                    num_pos_votes += 1

            num_total_votes = num_neg_votes + num_pos_votes
            pos_vote_percentage = num_pos_votes / num_total_votes


            if slide_category == "pos":
                num_true_positives += 1

            if slide_category == "neg":
                num_true_negatives += 1

            if pos_vote_percentage > percent_for_positive_classification:
                if slide_category == "pos":
                    num_positives_classified_correctly += 1
                elif slide_category == "neg":
                    num_false_positives += 1

        true_positive_rate = num_positives_classified_correctly / num_true_positives
        false_positive_rate = num_false_positives / num_true_negatives

        roc_point = ROC_Point(true_positive_rate, false_positive_rate)
        return roc_point

def draw_roc_curve(roc_points):
    """
    Given a list of ROC points (true_positive_rate, false_positive_rate),
    this function draws an ROC curve and saves the resulting file to disk
    
    Args:
        roc_points (ROC_Point list): List of points to use for the curve
    Returns:
        None (output saved to disk)
    
    """
    plt.figure()
    lw = 2

    x = [point.false_positive_rate for point in roc_points]
    y = [point.true_positive_rate for point in roc_points]
    roc_auc = auc(x, y) 

    plt.plot(x, y, color='darkorange', lw=lw, label="ROC Curve (area = %0.2f)" % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", help="model file to load", default="/tmp/output_graph.pb")
    parser.add_argument("--image_folder", help="folder containing image patches", default="/data/ethan/hne_patches_tumor_only/")
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--label_file", help="label file to load", default="/data/ethan/Breast_Deep_Learning/labels.csv")
    parser.add_argument("--test_slide_list", help="pickle containing list of test slides", default="./testing_slides")
    args = parser.parse_args()
    
    roc_point_list = []
    for i in range(1,10):
        roc_point_list.append(classify_whole_slides(i/10))
    
    with open("roc_points", "wb") as fp:
        pickle.dump(roc_point_list, fp) 
