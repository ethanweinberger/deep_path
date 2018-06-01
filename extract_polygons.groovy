def path = buildFilePath(PROJECT_BASE_DIR, 'annotation_csv_files')
mkdirs(path)

def name = getProjectEntry().getImageName() + '.csv'
path = buildFilePath(path, name)
def file = new File(path)
file.text = ''

def annotation_objects = getAnnotationObjects()
if (annotation_objects.size() > 0) {
    // Assuming that our first annotation object is the only one we care about
    def annotation = getAnnotationObjects()[0]
    def roi = annotation.getROI()
    
    if (annotation != null) {
        for (point in roi.getPolygonPoints()) {
            file << point.getX() << ',' << point.getY() << System.lineSeparator()
        }
    }
}
