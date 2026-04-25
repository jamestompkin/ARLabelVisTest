"""OFF triangle-mesh file reader.

Used by the RGD all-pairs driver and the ``validation/`` diagnostic scripts
to ingest meshes produced outside the Python pipeline. The writer and
point-cloud PLY helpers that used to live here were unused and have been
removed.
"""
import numpy as np


def read_off(filename):
    """Parse an OFF triangle-mesh file into ``(vertices, faces)`` arrays."""
    with open(filename, 'r') as f:
        line = f.readline().strip()
        if line != 'OFF':
            raise ValueError('Not a valid OFF file')

        line = f.readline().strip()
        while line.startswith('#') or len(line) == 0:
            line = f.readline().strip()

        counts = [int(x) for x in line.split()]
        nv, nf = counts[0], counts[1]

        vertices = []
        for _ in range(nv):
            line = f.readline().strip()
            vertices.append([float(x) for x in line.split()])

        faces = []
        for _ in range(nf):
            line = f.readline().strip()
            parts = [int(x) for x in line.split()]
            if parts[0] != 3:
                raise ValueError('Only triangle meshes supported')
            faces.append(parts[1:4])

    return np.array(vertices), np.array(faces)
