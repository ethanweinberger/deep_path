import argparse
import sys
import pickle
import os
import numpy as np
import tensorflow as tf
import pandas as pd

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", help="model file to load", default="/tmp/output_graph.pb")
    parser.add_argument("--image_folder", help="folder containing image patches", default="/data/ethan/hne_patches_tumor_only/")
    parser.add_argument("--input_layer", help="name of input layer", default="Placeholder")
    parser.add_argument("--output_layer", help="name of output layer", default="final_result")
    parser.add_argument("--label_file", help="label file to load", default="/data/ethan/Breast_Deep_Learning/labels.csv")
    parser.add_argument("--test_slide_list", help="pickle containing list of test slides", default="./testing_slides")
    args = parser.parse_args()

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

            if (num_neg_votes > num_pos_votes and slide_category == "neg") or (num_pos_votes > num_neg_votes and slide_category == "pos"):
                print(slide_name + " classified correctly")
                slides_classified_correctly += 1
                slides_examined += 1 
            else:
                print(slide_name + " classified incorrectly")
                slides_examined += 1
        print(str(slides_classified_correctly/slides_examined))
