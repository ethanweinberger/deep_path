import os

LABEL_FILE              = "/data/ethan/Breast_Deep_Learning/labels.csv"

#Data directories
SLIDE_FILE_DIRECTORY     = "/data/ethan/Breast_Deep_Learning/Polaris/263/"
SLIDE_FILE_EXTENSION     = "qptiff" 
PATCH_OUTPUT_DIRECTORY   = "/data/ethan/hne_patches/tumor_stroma_interface/"
LABEL_FILE_PATH          = "/data/ethan/Breast_Deep_Learning/labels.csv"
ANNOTATION_CSV_DIRECTORY = "/data/ethan/Breast_Deep_Learning/annotation_csv_files/" 

#Constants for pre-trained models
MODEL_FILE_FOLDER       = "./output_graph_files"
INPUT_LAYER             = "Placeholder"
OUTPUT_LAYER            = "final_result"
TEST_SLIDE_FOLDER       = "./testing_slide_lists"
TEST_SLIDE_LIST         = "testing_slide_list"

#Visualization output locations
HISTOGRAM_FOLDER        = "./histograms"

#Visualization helper files
VISUALIZATION_HELPER_FILE_FOLDER = "visualization_helper_files"
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
    



