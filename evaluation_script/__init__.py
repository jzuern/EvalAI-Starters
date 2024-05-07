"""
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.

import os
import subprocess
import sys
from pathlib import Path

def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def install_local_package(folder_name):
    # Install a local python package

    # Args:
    #     folder_name ([str]): name of the folder placed in evaluation_script/
    
    subprocess.check_output(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        os.path.join(str(Path(__file__).parent.absolute()) + folder_name),
    ]
)

install("shapely==1.7.1")
install("requests==2.25.1")

install_local_package("package_folder_name")

"""

import os
import subprocess
import sys
from pathlib import Path

def install(package):
    # Install a pip python package

    # Args:
    #     package ([str]): Package name with version
    
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    
    
install("shapely==1.7.1")
install("requests==2.25.1")
install("Pillow")
# install("scikit-image")  # this leads to the image not being found
# install("imageio[opencv]")   # this leads to the image not being found
install("scipy")



# pip install evaluation_script/benchmark.

command = "python -m pip install -r {}".format(
   os.path.join(str(Path(__file__).parent.absolute()) + "/requirements.txt"))
subprocess.run(command, shell=True, check=True)


print("INSTALLING ATTEMPT v0.1")

sys.path.append(os.path.join(os.path.dirname(__file__)))




from .main import evaluate