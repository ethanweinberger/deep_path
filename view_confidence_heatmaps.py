from utils.vis_utils import two_dim_confidence_visualization
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--which_fold',
      type=int,
      help='Which fold in kfold validation to use as testing data',
      required=True
    )
    args = parser.parse_args()
    two_dim_confidence_visualization(args.which_fold)
