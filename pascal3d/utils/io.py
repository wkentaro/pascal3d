import numpy as np


def load_pcd(pcd_file):
    """Load xyz pcd file.

    Parameters
    ----------
    pcd_file: str
        PCD filename.
    """
    points = []
    n_points = None
    with open(pcd_file, 'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue

            meta_fields = [
                'VERSION',
                'FIELDS',
                'SIZE',
                'TYPE',
                'COUNT',
                'WIDTH',
                'HEIGHT',
                'VIEWPOINT',
                'POINTS',
                'DATA',
            ]
            meta = line.strip().split(' ')
            meta_header, meta_contents = meta[0], meta[1:]
            if meta_header == 'FIELDS':
                assert meta_contents == ['x', 'y', 'z']
            elif meta_header == 'POINTS':
                n_points = int(meta_contents[0])
            if meta_header in meta_fields:
                continue

            x, y, z = map(float, line.split(' '))
            points.append((x, y, z))

    points = np.array(points)

    if n_points is not None:
        assert len(points) == n_points
        assert points.shape[1] == 3

    return points
