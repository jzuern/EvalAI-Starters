"""
# Q. How to install custom python pip packages?

# A. Uncomment the below code to install the custom python packages.
"""

import os
import subprocess
import sys
from pathlib import Path


command = "python -m pip install -r {}".format(
   os.path.join(str(Path(__file__).parent.parent.absolute()) + "/github/requirements.txt"))
subprocess.run(command, shell=True, check=True)


from .main import evaluate
