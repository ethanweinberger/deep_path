from utils.vis_utils import two_dim_confidence_visualization
import os
import constants
import shutil
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_fold',
      type=int,
      help='Which fold in kfold validation to use as testing data',
      required=True
    )
    args = parser.parse_args()
    if os.path.exists(constants.HEATMAP_FOLDER):
        shutil.rmtree(constants.HEATMAP_FOLDER)

    os.makedirs(constants.HEATMAP_FOLDER)
    two_dim_confidence_visualization(args.which_fold)
