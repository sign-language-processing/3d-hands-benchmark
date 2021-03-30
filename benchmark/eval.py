from json import dumps
from pathlib import Path
import numpy as np

from benchmark.mace import mace

systems = {}
for path in Path('systems').rglob('*.npy'):
    system_name = str(path)[len('systems/'):-len('.npy')]

    with open(path, 'rb') as f:
        poses = np.load(f)
        systems["system_name"] = {
            'mace': mace(poses)
        }

print(dumps(systems, indent=2))
