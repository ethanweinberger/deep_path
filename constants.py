import os

#Training Parameters
NUM_FOLDS = 2

#Data directories
SLIDE_FILE_DIRECTORY     = "/data/ethan/Marrow_Slides"
SLIDE_FILE_EXTENSION     = "svs" 
PATCH_SIZE               = 256
PATCH_OUTPUT_DIRECTORY   = "/data/ethan/marrow_patches_" + str(PATCH_SIZE) + "/"
LARGE_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "large_tumor_cells")
SMALL_CELL_PATCHES       = os.path.join(PATCH_OUTPUT_DIRECTORY, "small_tumor_cells")
#LABEL_FILE_PATH         = "/data/ethan/Breast_Deep_Learning/labels.csv"
LABEL_FILE               = "/data/ethan/lymphoma_case_codes.csv"
ANNOTATION_CSV_DIRECTORY = "/data/ethan/marrow_annotation_csv_files/" 

#Constants for pre-trained models
HOW_MANY_TRAINING_STEPS = 100
BOTTLENECK_DIR          = "/tmp/bottleneck_" + str(PATCH_SIZE)
MODEL_FILE_FOLDER       = "./output_graph_files_" + str(PATCH_SIZE)
INPUT_LAYER             = "Placeholder"
OUTPUT_LAYER            = "final_result"
TEST_SLIDE_FOLDER       = "./testing_slide_lists_" + str(PATCH_SIZE)
TEST_SLIDE_LIST         = "testing_slide_list"

#Visualization output locations
HISTOGRAM_FOLDER = "histograms"
def HISTOGRAM_SUBFOLDER(fold_number):
    return os.path.join(HISTOGRAM_FOLDER, "fold_" + str(fold_number))
HEATMAP_FOLDER = "heatmaps"
def HEATMAP_SUBFOLDER(fold_number):
    return os.path.join(HEATMAP_FOLDER, "fold_" + str(fold_number))

#Visualization helper files
VISUALIZATION_HELPER_FILE_FOLDER = "visualization_helper_files_" + str(PATCH_SIZE)
PATCH_CONFIDENCE_FOLDER          = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_confidences")
PATCH_NAME_TO_COORDS_MAP         = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "patch_name_to_coords_map")
SLIDE_NAME_TO_TILE_DIMS_MAP      = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_tile_dims_map")
SLIDE_NAME_TO_PATCHES_MAP        = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "slide_name_to_patches_map")
FOLD_VOTE_CONTAINER_LISTS_PATH   = os.path.join(VISUALIZATION_HELPER_FILE_FOLDER, "fold_vote_container_lists")
    
def PATCH_CONFIDENCE_FOLD_SUBFOLDER(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number))
def PATCH_NAME_TO_CONFIDENCE_MAP(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "patch_name_to_confidence_map")
def CONFIDENCE_CONTAINER_LIST(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "confidence_containers")
def POS_SLIDE_CONFIDENCE_LISTS(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "pos_slide_confidence_lists")
def NEG_SLIDE_CONFIDENCE_LISTS(fold_number):
    return os.path.join(PATCH_CONFIDENCE_FOLDER, "fold_" + str(fold_number), "neg_slide_confidence_lists")
    



