from utils.vis_utils import draw_confidence_histograms
import constants

if __name__ == "__main__":
    for i in range(constants.NUM_FOLDS):
        draw_confidence_histograms(i, "pos")
        draw_confidence_histograms(i, "neg")
