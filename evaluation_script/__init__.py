
import os
import subprocess
import sys
from pathlib import Path

command = "{} -m pip install -r {}".format(sys.executable, os.path.join(str(Path(__file__).parent.parent.absolute()) + "/github/requirements.txt"))
subprocess.run(command, shell=True, check=True)


from .main import evaluate
