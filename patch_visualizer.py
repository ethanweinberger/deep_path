from utils.vis_utils import visualize_confidence_containers 
from utils.vis_utils import PatchAndConfidence
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_fold',
      type=int,
      help='Which fold in kfold validation to use as testing data',
      required=True
    )
    args = parser.parse_args()
    visualize_confidence_containers(args.which_fold)
