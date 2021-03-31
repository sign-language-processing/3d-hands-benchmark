from json import dumps
from pathlib import Path

import numpy as np

from benchmark.cce import cce
from benchmark.mace import mace

systems = {}
for path in Path('systems').rglob('*.npy'):
    system_name = str(path)[len('systems/'):-len('.npy')]

    with open(path, 'rb') as f:
        poses = np.load(f)
        assert poses.ndim == 5

        median_poses = np.expand_dims(np.median(poses, axis=0), 0)

        systems[system_name] = {
            'mace': mace(poses),
            'cce': cce(poses)
        }
        systems[system_name]['mace']['median'] = mace(median_poses)

print(dumps(systems, indent=2))
