import os
import shutil

def createDirectories():
    directories = ["plots", "logs"]

    for directory in directories:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory)
        else:
            os.makedirs(directory)

    if(not os.path.exists("models")):
        os.makedirs("models")