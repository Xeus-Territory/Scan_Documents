import os


def get_Tree():
    """
        This is a tree class that will be used to create a tree of files and directories
    """
    path = os.path.abspath(os.path.dirname(__file__))
    for dirpath, dirnames, filenames in os.walk(path):
        directory_level = dirpath.replace(path, "")
        directory_level = directory_level.count(os.sep)
        indent = " " * 4
        print("{}{}/".format(indent*directory_level, os.path.basename(dirpath)))

        for f in filenames:
            print("{}{}".format(indent*(directory_level+1), f))
