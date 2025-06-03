import open3d as o3d
import numpy as np
import os

def visualize (path, Mesh = None):
    if Mesh is None:
        mesh = o3d.io.read_triangle_mesh(path)
    else:
        mesh = Mesh
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])


def simplify_mesh(mesh, target_face_count=500):
    original_face_count = len(mesh.triangles)
    if original_face_count > target_face_count:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_face_count)
    return mesh


def compute_and_save_npz(simplified_mesh, output_path):

    vertices = np.asarray(simplified_mesh.vertices)
    faces = np.asarray(simplified_mesh.triangles)

    # Optional: Normalize mesh to unit cube (like PointNet/MeshNet)
    center = vertices.mean(axis=0)
    scale = np.linalg.norm(vertices - center, axis=1).max()
    vertices = (vertices - center) / scale

    np.savez(output_path, vertices=vertices, faces=faces)

def process_modelnet(modelnet_root, output_root, face_threshold=500):

    for class_name in os.listdir(modelnet_root):
        class_path = os.path.join(modelnet_root, class_name)
        if not os.path.isdir(class_path):
            continue  # skip non-folders
        
        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue
            
            # Output folder (preserve structure)
            out_split_path = os.path.join(output_root, class_name, split)
            os.makedirs(out_split_path, exist_ok=True)

            for fname in os.listdir(split_path):
                if not fname.endswith('.off'):
                    continue
                mesh_path = os.path.join(split_path, fname)
                mesh = o3d.io.read_triangle_mesh(mesh_path)
                mesh.compute_vertex_normals()
                simplified = simplify_mesh(mesh, target_face_count=face_threshold)

                # Save simplified mesh to target location
                out_path = os.path.join(out_split_path, fname)
                compute_and_save_npz(simplified, out_path)
                print(f"Simplified and saved: {out_path}")

