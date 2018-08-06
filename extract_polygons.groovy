def top_level_path        = buildFilePath(PROJECT_BASE_DIR, 'annotation_csv_files')
def stroma_path           = buildFilePath(top_level_path, 'stroma_csv_files')
def large_cell_tumor_path = buildFilePath(top_level_path, 'large_cell_tumor_csv_files')
def small_cell_tumor_path = buildFilePath(top_level_path, 'small_cell_tumor_csv_files')

mkdirs(top_level_path)
mkdirs(stroma_path)
mkdirs(large_cell_tumor_path)
mkdirs(small_cell_tumor_path)

def image_name = getProjectEntry().getImageName()
stroma_image_path = buildFilePath(stroma_path, image_name)
large_cell_tumor_image_path = buildFilePath(large_cell_tumor_path, image_name)
small_cell_tumor_image_path = buildFilePath(small_cell_tumor_path, image_name)

mkdirs(stroma_image_path)
mkdirs(large_cell_tumor_image_path)
mkdirs(small_cell_tumor_image_path)

def annotation_objects = getAnnotationObjects()
def counter = 0

for (annotation_object in annotation_objects) {
      
    //metaClass.methods*.name.sort().unique()
    name = counter.toString() + '.csv'
    if (annotation_object.getPathClass() == getPathClass("Tumor")) {
        annotation_dir = large_cell_tumor_image_path
    }
    else if (annotation_object.getPathClass() == getPathClass("Stroma")) {
        annotation_dir = stroma_image_path
    }
    else {
        annotation_dir = small_cell_tumor_image_path
    }
    
    annotation_file = buildFilePath(annotation_dir, name)
    def file = new File(annotation_file)
    file.text = ''

    // Assuming that our first annotation object is the only one we care about
    def roi = annotation_object.getROI()
    
    for (point in roi.getPolygonPoints()) {
        file << point.getX() << ',' << point.getY() << System.lineSeparator()
    }
    counter++
}
