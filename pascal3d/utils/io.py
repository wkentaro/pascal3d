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


def load_off(off_file):
    verts, faces = None, None
    with open(off_file, 'r') as f:
        verts, faces = None, None
        assert 'OFF' in f.readline()
        n_verts, n_faces, _ = map(int, f.readline().strip().split(' '))
        verts = np.zeros((n_verts, 3), dtype=np.float64)
        faces = np.zeros((n_faces, 3), dtype=np.int64)
        for i in xrange(n_verts):
            verts[i] = map(float, f.readline().strip().split(' '))
        for i in xrange(n_faces):
            faces[i] = map(int, f.readline().strip().split(' ')[1:])
    return verts, faces
