import pickle
import sys

def load_pickle_from_disk(file_path):
    """
    Helper function that loads a pickle and returns
    the resulting python object. Raises an exception
    if the provided file does not exist.

    Args:
        file_path (String): Path to the pickle we want to load
    Returns:
        pickle_contents (Type depends on what pickle contains)

    """

    try:
        pickle_contents = pickle.load(open(file_path, "rb"))
        return pickle_contents
    except Exception as error:
        print("Couldn't open " + file_path + ". Did you run the helper scripts first?")
        print("Exitting...")
        sys.exit()

def write_pickle_to_disk(file_name, python_object):
    """
    Helper function to write python_object to disk
    as a pickle file with name file_name.  Used to 
    make other code easier to read

    Args:
        file_name (String): File name of new pickle file
        python_object (Any): Python object to be pickled
    Returns:
        None (output saved to disk)

    """
    try:
        file_pointer = open(file_name, "wb")
        pickle.dump(python_object, file_pointer)
    except Exception as error:
        print("Unable to write pickle " + file_name + " to disk.  Do you have sufficient permissions?")
        print("Exitting...")
        sys.exit()
