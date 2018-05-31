// QuPath script for extracting the vertices of an annotation 

def path = buildFilePath(PROJECT_BASE_DIR, 'polygons.txt')
def file = new File(path)
file.text = ''

// Loop through all objects & write the points to the file
for (pathObject in getAllObjects()) {
    // Check for interrupt (Run -> Kill running script)
    if (Thread.interrupted())
        break
    // Get the ROI
    def roi = pathObject.getROI()
    if (roi == null)
        continue
    // Write the points; but beware areas, and also ellipses!
    file << roi.getPolygonPoints() << System.lineSeparator()
}
