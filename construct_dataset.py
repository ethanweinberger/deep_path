from utils.slide_utils import construct_training_dataset
import constants

if __name__ == "__main__":
    construct_training_dataset(
        constants.SLIDE_FILE_DIRECTORY,
        constants.SLIDE_FILE_EXTENSION,
        constants.PATCH_OUTPUT_DIRECTORY,
        constants.LABEL_FILE_PATH,
        constants.ANNOTATION_CSV_DIRECTORY
    )
