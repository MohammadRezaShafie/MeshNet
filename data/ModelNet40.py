import numpy as np
import os
import torch
import torch.utils.data as data
import pymeshlab
import open3d as o3d
import numpy as np
from data.preprocess import find_neighbor
from sklearn.model_selection import train_test_split

type_to_index_map = {
    'Infeasible_Designs': 0, 'Feasible_Designs': 1}

class ModelNet40(data.Dataset):

    def __init__(self, cfg, part='train', split_ratio=0.8):
        self.root = cfg['data_root']
        self.max_faces = cfg['max_faces']
        self.part = part
        self.augment_data = cfg['augment_data']
        if self.augment_data:
            self.jitter_sigma = cfg['jitter_sigma']
            self.jitter_clip = cfg['jitter_clip']

        self.data = []
        for type in os.listdir(self.root):
            if type not in type_to_index_map.keys():
                continue
            type_index = type_to_index_map[type]
            type_root = os.path.join(self.root, type)
            for filename in os.listdir(type_root):
                if filename.endswith('.npz') or filename.endswith('.stl'):
                    self.data.append((os.path.join(type_root, filename), type_index))

        # train_data, test_data = train_test_split(self.data, test_size=(1 - split_ratio), stratify=[d[1] for d in self.data],random_state=42)

        # if self.part == 'train':
        #     self.data = train_data
        # else:
        #     self.data = test_data

    def __getitem__(self, i):
        path, type = self.data[i]
        if path.endswith('.npz'):
            data = np.load(path)
            face = data['faces']
            neighbor_index = data['neighbors']
        else:
            face, neighbor_index = process_mesh(path, self.max_faces)
            if face is None:
                return self.__getitem__(0)

        # data augmentation
        if self.augment_data and self.part == 'train':
            # jitter
            jittered_data = np.clip(self.jitter_sigma * np.random.randn(*face[:, :3].shape), -1 * self.jitter_clip, self.jitter_clip)
            face = np.concatenate((face[:, :3] + jittered_data, face[:, 3:]), 1)

        # fill for n < max_faces with randomly picked faces
        num_point = len(face)
        if num_point < self.max_faces:
            fill_face = []
            fill_neighbor_index = []
            for i in range(self.max_faces - num_point):
                index = np.random.randint(0, num_point)
                fill_face.append(face[index])
                fill_neighbor_index.append(neighbor_index[index])
            face = np.concatenate((face, np.array(fill_face)))
            neighbor_index = np.concatenate((neighbor_index, np.array(fill_neighbor_index)))

        # to tensor
        face = torch.from_numpy(face).float()
        neighbor_index = torch.from_numpy(neighbor_index).long()
        target = torch.tensor(type, dtype=torch.float)

        # reorganize
        face = face.permute(1, 0).contiguous()
        centers, corners, normals = face[:3], face[3:12], face[12:]
        corners = corners - torch.cat([centers, centers, centers], 0)

        return centers, corners, normals, neighbor_index, target

    def __len__(self):
        return len(self.data)

def process_mesh(path, max_faces):
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(path)
    
    # Clean up
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_degenerate_triangles()

    
    voxel_size = max(mesh.get_max_bound() - mesh.get_min_bound()) / 10

    mesh = mesh.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)

    # Get elements
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)

    if faces.shape[0] >= max_faces:     # only occur once in train set of Manifold40
        print("Model with more than {} faces ({}): {}".format(max_faces, faces.shape[0], path))
        return None, None

    # Move to center
    center = (np.max(vertices, 0) + np.min(vertices, 0)) / 2
    vertices -= center

    # Normalize
    max_len = np.max(np.sum(vertices**2, axis=1))
    vertices /= np.sqrt(max_len)

    # Get normal vector
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh.compute_triangle_normals()
    face_normals = np.asarray(mesh.triangle_normals)

    # Get neighbors
    faces_contain_this_vertex = [set() for _ in range(len(vertices))]
    centers = []
    corners = []
    for i, face in enumerate(faces):
        v1, v2, v3 = face
        x1, y1, z1 = vertices[v1]
        x2, y2, z2 = vertices[v2]
        x3, y3, z3 = vertices[v3]
        centers.append([(x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3, (z1 + z2 + z3) / 3])
        corners.append([x1, y1, z1, x2, y2, z2, x3, y3, z3])
        faces_contain_this_vertex[v1].add(i)
        faces_contain_this_vertex[v2].add(i)
        faces_contain_this_vertex[v3].add(i)

    neighbors = []
    for i, face in enumerate(faces):
        v1, v2, v3 = face
        n1 = find_neighbor(faces, faces_contain_this_vertex, v1, v2, i)
        n2 = find_neighbor(faces, faces_contain_this_vertex, v2, v3, i)
        n3 = find_neighbor(faces, faces_contain_this_vertex, v3, v1, i)
        neighbors.append([n1, n2, n3])

    centers = np.array(centers)
    corners = np.array(corners)
    faces_combined = np.concatenate([centers, corners, face_normals], axis=1)
    neighbors = np.array(neighbors)

    return faces_combined, neighbors
