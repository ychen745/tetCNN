import numpy as np
import os

class TetMesh():
    def __init__(self) -> None:
        self.vertices = None
        self.faces = None
        self.voxels = None
        self.num_vertices = 0
        self.num_faces = 0
        self.num_voxels = 0
    
    def load_mesh(self, fnode):
        if not fnode.endswith('.node'):
            print('load_mesh error: please specify correct .node file name.')
            return
        
        try:
            f = open(fnode)
            f.readline()
            f.close()
        except:
            print('Cannot read .node file.')
            return
        
        fvoxel = fnode[:-4] + 'ele'
        fface = fnode[:-4] + 'face'

        self.num_vertices = int(open(fnode).readline().split()[0])
        vertices = np.loadtxt(fnode, skiprows=1, dtype=np.float32)
        self.vertices = vertices[:, 1:]
        if self.vertices.shape[0] != self.num_vertices:
            print('vertex dimension error.')

        self.num_faces = int(open(fface).readline().split()[0])
        faces = np.loadtxt(fface, skiprows=1, dtype=np.int32)
        self.faces = faces[:, 1:-1]
        if self.faces.shape[0] != self.num_faces:
            print('face dimension error.')

        self.num_voxels = int(open(fvoxel).readline().split()[0])
        voxels = np.loadtxt(fvoxel, skiprows=1, dtype=np.int32)
        self.voxels = voxels[:, 1:-1]
        if self.voxels.shape[0] != self.num_voxels:
            print('voxel dimension error.')

def load_mesh(fnode):
    mesh = TetMesh()
    mesh.load_mesh(fnode)
    return mesh

if __name__ == '__main__':
    pass
    # mesh = tet_mesh()
    # mesh.load_mesh('../data/011_S_4827/rh_hippo.1.node')
    # print(mesh.num_vertices)
    # print(mesh.num_faces)
    # print(mesh.num_voxels)
    # print(mesh.vertices)
    # print(mesh.faces)
    # print(mesh.voxels)
    # print(mesh.vertices.dtype)
    # print(mesh.faces.dtype)
    # print(mesh.voxels.dtype)