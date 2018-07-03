import json
from pprint import pprint

def windows_path_to_linux_path(windows_path_string):
    """
    Converts a filepath string from Windows format to Linux format.  Used to convert paths in qpproj
    files originating on Windows machines

    Args:
        windows_path_string (String): Windows format filepath string
    Returns:
        linux_path_string (String): Converted filepath string
    """

    linux_path_string = windows_path_string.replace("\\", "/")
    linux_path_string = linux_path_string.replace("F:", "/media/ethan/Breast DL2")
    #linux_path_string = linux_path_string.replace(" ", "\ ")
    return linux_path_string

with open("project.qpproj2") as f:
    data = json.load(f)
    for image in data["images"]:
        image["path"] = windows_path_to_linux_path(image["path"])
    
    outfile = open("output.qpproj", "w")
    json.dump(data, outfile, indent=2)
    pprint(data)


