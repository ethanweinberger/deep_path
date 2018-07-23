import argparse
import numpy as np
import re
import os
import sys
import tensorflow as tf
import tensorflow_hub as hub
import collections
import pickle
import shutil

from utils.train_utils import add_jpeg_decoding
from utils.train_utils import add_input_distortions
from utils.train_utils import add_evaluation_step
from utils.train_utils import prepare_file_system
from utils.train_utils import should_distort_images
from utils.train_utils import create_module_graph
from utils.train_utils import add_final_retrain_ops 
from utils.train_utils import cache_bottlenecks
from utils.train_utils import get_random_cached_bottlenecks 
from utils.train_utils import get_random_distorted_bottlenecks
from utils.train_utils import run_final_eval
from utils.train_utils import save_graph_to_file
from utils.file_utils import write_pickle_to_disk
import constants
from sklearn.model_selection import KFold
from datetime import datetime

FLAGS = None
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M

# The location where variable checkpoints will be stored.
CHECKPOINT_NAME = '/tmp/_retrain_checkpoint'

# A module is understood as instrumented for quantization with TF-Lite
# if it contains any of these ops.
FAKE_QUANT_OPS = ('FakeQuantWithMinMaxVars',
                  'FakeQuantWithMinMaxVarsPerChannel')

def create_image_lists_kfold(image_dir, num_folds):
  """
  Builds a list of lists of training images from the file system.  The number
  of lists here will be equal to the num_folds, analogous to k-fold cross
  validation

  Args:
    image_dir: String path to a folder containing subfolders of images.
    num_folds (Int): Number of folds we want to split our data into

  Returns:
    An OrderedDict containing an entry for each label subfolder, with images
    split into training, testing, and validation sets within each label.
    The order of items defines the class indices.
  """
  if not tf.gfile.Exists(image_dir):
    tf.logging.error("Image directory '" + image_dir + "' not found.")
    return None
  result_list = []
  testing_slide_lists = []
  for i in range(num_folds):
    result_list.append(collections.OrderedDict())
    testing_slide_lists.append([])

  class_dirs = []
  slide_dirs = []

  for root,dirs,files in os.walk(image_dir):
    if dirs:
      class_dirs.append(root)
    else:
      slide_dirs.append(root)
  # The root directory comes first, so skip it.
  is_root_dir = True
  for class_dir in class_dirs:
    if is_root_dir:
      is_root_dir = False
      continue
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    slide_list = []
    dir_name = os.path.basename(class_dir)

    if dir_name == image_dir:
      continue

    tf.logging.info("Looking for folders in '" + dir_name + "'")
    file_glob = os.path.join(image_dir, dir_name, '*')
    slide_list.extend(tf.gfile.Glob(file_glob))

    if not slide_list:
      tf.logging.warning('No slide directories found')
      continue

    label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())
    result = collections.OrderedDict()
    
    kf = KFold(n_splits=num_folds, shuffle=True)
    current_fold = 0
    testing_slides_base_names = []
    for train, test in kf.split(slide_list):
      training_slides = np.array(slide_list)[train]
      testing_slides = np.array(slide_list)[test]    
      testing_slide_lists[current_fold].extend(testing_slides)

      training_images = []
      testing_images = []

      for training_slide_name in training_slides:
        for image in os.listdir(training_slide_name):
          training_images.append(image)

      for testing_slide_name in testing_slides:
        for image in os.listdir(testing_slide_name):
          testing_images.append(image)  
      
      result_list[current_fold][label_name] = {
        'dir': dir_name,
        'training': training_images,
        'testing': testing_images,
      }
      current_fold += 1

  if os.path.exists(constants.TEST_SLIDE_FOLDER):
      shutil.rmtree(constants.TEST_SLIDE_FOLDER)

  os.makedirs(constants.TEST_SLIDE_FOLDER)
  for i in range(num_folds):
    write_pickle_to_disk(os.path.join(constants.TEST_SLIDE_FOLDER, "testing_slide_list_" + str(i)),
      testing_slide_lists[i])

  return result_list

 
def main(_):
  # Needed to make sure the logging output is visible.
  # See https://github.com/tensorflow/tensorflow/issues/3047
  tf.logging.set_verbosity(tf.logging.INFO)

  if not FLAGS.image_dir:
    tf.logging.error('Must set flag --image_dir.')
    return -1

  # Prepare necessary directories that can be used during training
  prepare_file_system(FLAGS)

  # Look at the folder structure, and create lists of all the images.
  image_lists_folds = create_image_lists_kfold(FLAGS.image_dir, constants.NUM_FOLDS)
  network_count = 0

  if not os.path.exists(constants.MODEL_FILE_FOLDER):
    os.makedirs(constants.MODEL_FILE_FOLDER)

  for image_lists in image_lists_folds: 

    class_count = len(image_lists.keys())
    if class_count == 0:
      tf.logging.error('No valid folders of images found at ' + FLAGS.image_dir)
      return -1
    if class_count == 1:
      tf.logging.error('Only one valid folder of images found at ' +
                       FLAGS.image_dir +
                       ' - multiple classes are needed for classification.')
      return -1

    # See if the command-line flags mean we're applying any distortions.
    do_distort_images = should_distort_images(
        FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
        FLAGS.random_brightness)

    # Set up the pre-trained graph.
    module_spec = hub.load_module_spec(FLAGS.tfhub_module)
    graph, bottleneck_tensor, resized_image_tensor, wants_quantization = (
        create_module_graph(module_spec))

    # Add the new layer that we'll be training.
    with graph.as_default():
      (train_step, cross_entropy, bottleneck_input,
       ground_truth_input, final_tensor) = add_final_retrain_ops(
           class_count, FLAGS.final_tensor_name, bottleneck_tensor,
           wants_quantization, FLAGS, is_training=True)

    with tf.Session(graph=graph) as sess:
      # Initialize all weights: for the module to their pretrained values,
      # and for the newly added retraining layer to random initial values.
      init = tf.global_variables_initializer()
      sess.run(init)

      # Set up the image decoding sub-graph.
      jpeg_data_tensor, decoded_image_tensor = add_jpeg_decoding(module_spec)

      if do_distort_images:
        # We will be applying distortions, so setup the operations we'll need.
        (distorted_jpeg_data_tensor,
         distorted_image_tensor) = add_input_distortions(
             FLAGS.flip_left_right, FLAGS.random_crop, FLAGS.random_scale,
             FLAGS.random_brightness, module_spec)
      else:
        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        cache_bottlenecks(sess, image_lists, FLAGS.image_dir,
                          FLAGS.bottleneck_dir, jpeg_data_tensor,
                          decoded_image_tensor, resized_image_tensor,
                          bottleneck_tensor, FLAGS.tfhub_module)

      # Create the operations we need to evaluate the accuracy of our new layer.
      evaluation_step, _ = add_evaluation_step(final_tensor, ground_truth_input)

      # Merge all the summaries and write them out to the summaries_dir
      merged = tf.summary.merge_all()
      train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
                                           sess.graph)

      # Create a train saver that is used to restore values into an eval graph
      # when exporting models.
      train_saver = tf.train.Saver()

      # Run the training for as many cycles as requested on the command line.
      for i in range(FLAGS.how_many_training_steps):
        # Get a batch of input bottleneck values, either calculated fresh every
        # time with distortions applied, or from the cache stored on disk.
        if do_distort_images:
          (train_bottlenecks,
           train_ground_truth) = get_random_distorted_bottlenecks(
               sess, image_lists, FLAGS.train_batch_size, 'training',
               FLAGS.image_dir, distorted_jpeg_data_tensor,
               distorted_image_tensor, resized_image_tensor, bottleneck_tensor)
        else:
          (train_bottlenecks,
           train_ground_truth, _) = get_random_cached_bottlenecks(
               sess, image_lists, FLAGS.train_batch_size, 'training',
               FLAGS.bottleneck_dir, FLAGS.image_dir, jpeg_data_tensor,
               decoded_image_tensor, resized_image_tensor, bottleneck_tensor,
               FLAGS.tfhub_module)
        # Feed the bottlenecks and ground truth into the graph, and run a training
        # step. Capture training summaries for TensorBoard with the `merged` op.
        train_summary, _ = sess.run(
            [merged, train_step],
            feed_dict={bottleneck_input: train_bottlenecks,
                       ground_truth_input: train_ground_truth})
        train_writer.add_summary(train_summary, i)

        # Every so often, print out how well the graph is training.
        is_last_step = (i + 1 == FLAGS.how_many_training_steps)
        if (i % FLAGS.eval_step_interval) == 0 or is_last_step:
          train_accuracy, cross_entropy_value = sess.run(
              [evaluation_step, cross_entropy],
              feed_dict={bottleneck_input: train_bottlenecks,
                         ground_truth_input: train_ground_truth})
          tf.logging.info('%s: Step %d: Train accuracy = %.1f%%' %
                          (datetime.now(), i, train_accuracy * 100))
          tf.logging.info('%s: Step %d: Cross entropy = %f' %
                          (datetime.now(), i, cross_entropy_value))

        # Store intermediate results
        intermediate_frequency = FLAGS.intermediate_store_frequency

        if (intermediate_frequency > 0 and (i % intermediate_frequency == 0)
            and i > 0):
          # If we want to do an intermediate save, save a checkpoint of the train
          # graph, to restore into the eval graph.
          train_saver.save(sess, CHECKPOINT_NAME)
          intermediate_file_name = (FLAGS.intermediate_output_graphs_dir +
                                    'intermediate_' + str(i) + '.pb')
          tf.logging.info('Save intermediate result to : ' +
                          intermediate_file_name)
          save_graph_to_file(graph, intermediate_file_name, module_spec,
                             class_count)

      # After training is complete, force one last save of the train checkpoint.
      train_saver.save(sess, CHECKPOINT_NAME)

      # We've completed all our training, so run a final test evaluation on
      # some new images we haven't used before.)
      run_final_eval(sess, module_spec, class_count, image_lists,
                     jpeg_data_tensor, decoded_image_tensor, resized_image_tensor,
                     bottleneck_tensor, FLAGS)

      # Write out the trained graph and labels with the weights stored as
      # constants.
      if wants_quantization:
        tf.logging.info('The model is instrumented for quantization with TF-Lite')

      save_graph_to_file(graph, os.path.join(constants.MODEL_FILE_FOLDER, FLAGS.output_graph + "_" + str(network_count) + ".pb"),
        module_spec, class_count, FLAGS)
      with tf.gfile.FastGFile(FLAGS.output_labels, 'w') as f:
        f.write('\n'.join(image_lists.keys()) + '\n')

      if FLAGS.saved_model_dir:
        export_model(module_spec, class_count, FLAGS.saved_model_dir)
    network_count += 1

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--image_dir',
      type=str,
      default=constants.PATCH_OUTPUT_DIRECTORY,
      help='Path to folders of labeled images.'
  )
  parser.add_argument(
      '--output_graph',
      type=str,
      default='output_graph',
      help='Where to save the trained graph.'
  )
  parser.add_argument(
      '--intermediate_output_graphs_dir',
      type=str,
      default='/tmp/intermediate_graph/',
      help='Where to save the intermediate graphs.'
  )
  parser.add_argument(
      '--intermediate_store_frequency',
      type=int,
      default=0,
      help="""\
         How many steps to store intermediate graph. If "0" then will not
         store.\
      """
  )
  parser.add_argument(
      '--output_labels',
      type=str,
      default='/tmp/output_labels.txt',
      help='Where to save the trained graph\'s labels.'
  )
  parser.add_argument(
      '--summaries_dir',
      type=str,
      default='/tmp/retrain_logs',
      help='Where to save summary logs for TensorBoard.'
  )
  parser.add_argument(
      '--how_many_training_steps',
      type=int,
      default=300,
      help='How many training steps to run before ending.'
  )
  parser.add_argument(
      '--learning_rate',
      type=float,
      default=0.01,
      help='How large a learning rate to use when training.'
  )
  parser.add_argument(
      '--eval_step_interval',
      type=int,
      default=10,
      help='How often to evaluate the training results.'
  )
  parser.add_argument(
      '--train_batch_size',
      type=int,
      default=100,
      help='How many images to train on at a time.'
  )
  parser.add_argument(
      '--test_batch_size',
      type=int,
      default=-1,
      help="""\
      How many images to test on. This test set is only used once, to evaluate
      the final accuracy of the model after training completes.
      A value of -1 causes the entire test set to be used, which leads to more
      stable results across runs.\
      """
  )
  parser.add_argument(
      '--print_misclassified_test_images',
      default=False,
      help="""\
      Whether to print out a list of all misclassified test images.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--bottleneck_dir',
      type=str,
      default='/tmp/bottleneck',
      help='Path to cache bottleneck layer values as files.'
  )
  parser.add_argument(
      '--final_tensor_name',
      type=str,
      default='final_result',
      help="""\
      The name of the output classification layer in the retrained graph.\
      """
  )
  parser.add_argument(
      '--flip_left_right',
      default=True,
      help="""\
      Whether to randomly flip half of the training images horizontally.\
      """,
      action='store_true'
  )
  parser.add_argument(
      '--random_crop',
      type=int,
      default=10,
      help="""\
      A percentage determining how much of a margin to randomly crop off the
      training images.\
      """
  )
  parser.add_argument(
      '--random_scale',
      type=int,
      default=10,
      help="""\
      A percentage determining how much to randomly scale up the size of the
      training images by.\
      """
  )
  parser.add_argument(
      '--random_brightness',
      type=int,
      default=10,
      help="""\
      A percentage determining how much to randomly multiply the training image
      input pixels up or down by.\
      """
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/resnet_v2_152/feature_vector/1'),
      help="""\
      Which TensorFlow Hub module to use.
      See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
      for some publicly available ones.\
      """)
  parser.add_argument(
      '--saved_model_dir',
      type=str,
      default='',
      help='Where to save the exported graph.')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
