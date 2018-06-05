def top_level_path = buildFilePath(PROJECT_BASE_DIR, 'annotation_csv_files')
mkdirs(top_level_path)

def image_name = getProjectEntry().getImageName()
image_path = buildFilePath(top_level_path, image_name)
mkdirs(image_path)

def annotation_objects = getAnnotationObjects()
def counter = 0

for (annotation_object in annotation_objects) {
    
    name = counter.toString() + '.csv'
    annotation_path = buildFilePath(image_path, name)
    def file = new File(annotation_path)
    file.text = ''

    // Assuming that our first annotation object is the only one we care about
    def roi = annotation_object.getROI()
    
    for (point in roi.getPolygonPoints()) {
        file << point.getX() << ',' << point.getY() << System.lineSeparator()
    }
    counter++
}
