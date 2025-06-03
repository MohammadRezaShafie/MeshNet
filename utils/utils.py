import open3d as o3d
import numpy as np
import os
import tempfile
import shutil   


def visualize (path, Mesh = None):

    if Mesh is None:
        mesh = o3d.io.read_triangle_mesh(path)
    else:
        mesh = Mesh
    mesh.compute_vertex_normals()
    faces = len(mesh.triangles)
    print(f'{faces=}')
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
            continue

        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue

            out_split_path = os.path.join(output_root, class_name, split)
            os.makedirs(out_split_path, exist_ok=True)

            for fname in os.listdir(split_path):
                if not fname.endswith('.off'):
                    continue

                npz_fname = os.path.splitext(fname)[0] + ".npz"
                out_path = os.path.join(out_split_path, npz_fname)

                # ✅ Skip if already processed
                if os.path.exists(out_path):
                    continue

                mesh_path = os.path.join(split_path, fname)

                try:
                    fixed_path = fix_off_header(mesh_path)
                    mesh = o3d.io.read_triangle_mesh(fixed_path)
                    if not mesh.has_triangles():
                        print(f"⚠️ Skipping empty mesh: {mesh_path}")
                        continue
                    mesh.compute_vertex_normals()
                except Exception as e:
                    print(f"❌ Failed to load mesh {mesh_path}: {e}")
                    continue

                simplified = simplify_mesh(mesh, target_face_count=face_threshold)

                try:
                    compute_and_save_npz(simplified, out_path)
                    print(f"Simplified and saved: {out_path}")
                except Exception as e:
                    print(f"❌ Failed to save {out_path}: {e}")


def fix_off_header(path):
    """
    Fixes malformed OFF header (e.g., 'OFF4890 7376 0') by splitting it properly.
    Returns the path to a temporary corrected .off file.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # Check if the header is malformed
    if not lines[0].strip().startswith("OFF"):
        raise ValueError(f"Invalid OFF file: {path}")
    
    if lines[0].strip() != "OFF":
        # Fix the header
        fixed_lines = ["OFF\n", lines[0].strip()[3:] + "\n"] + lines[1:]
    else:
        return path  # No fix needed

    # Write to a temp file
    temp_fd, temp_path = tempfile.mkstemp(suffix=".off")
    with os.fdopen(temp_fd, 'w') as tmp:
        tmp.writelines(fixed_lines)

    return temp_path

def fix_off_header(path):
    """
    Fix OFF file if the first line is malformed (e.g., 'OFF4890 7376 0').
    Returns a path to a corrected temporary file, or the original if no fix is needed.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    if not lines[0].strip().startswith("OFF"):
        raise ValueError(f"Invalid OFF file: {path}")
    
    if lines[0].strip() != "OFF":
        fixed_lines = ["OFF\n", lines[0].strip()[3:] + "\n"] + lines[1:]
        temp_fd, temp_path = tempfile.mkstemp(suffix=".off")
        with os.fdopen(temp_fd, 'w') as tmp:
            tmp.writelines(fixed_lines)
        return temp_path
    else:
        return path


def simplify_mesh(mesh, target_face_count=500):
    """
    Simplify mesh if it has more than target_face_count triangles.
    """
    if len(mesh.triangles) > target_face_count:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_face_count)
    return mesh


def process_modelnet_to_obj(modelnet_root, output_root, face_threshold=500):
    """
    Convert all .off meshes in ModelNet40 to simplified .obj format while preserving directory structure.
    """
    for class_name in os.listdir(modelnet_root):
        class_path = os.path.join(modelnet_root, class_name)
        if not os.path.isdir(class_path):
            continue

        for split in ['train', 'test']:
            split_path = os.path.join(class_path, split)
            if not os.path.exists(split_path):
                continue

            out_split_path = os.path.join(output_root, class_name, split)
            os.makedirs(out_split_path, exist_ok=True)

            for fname in os.listdir(split_path):
                if not fname.endswith('.off'):
                    continue

                obj_fname = os.path.splitext(fname)[0] + ".obj"
                out_path = os.path.join(out_split_path, obj_fname)

                # ✅ Skip if already processed
                if os.path.exists(out_path):
                    continue

                mesh_path = os.path.join(split_path, fname)

                try:
                    fixed_path = fix_off_header(mesh_path)
                    mesh = o3d.io.read_triangle_mesh(fixed_path)
                    if not mesh.has_triangles():
                        print(f"⚠️ Skipping empty mesh: {mesh_path}")
                        continue
                    mesh.compute_vertex_normals()
                except Exception as e:
                    print(f"❌ Failed to load mesh {mesh_path}: {e}")
                    continue

                simplified = simplify_mesh(mesh, target_face_count=face_threshold)

                try:
                    simplified.compute_vertex_normals()
                    o3d.io.write_triangle_mesh(out_path, simplified, write_vertex_normals=True)
                    print(f"Simplified and saved OBJ: {out_path}")
                except Exception as e:
                    print(f"❌ Failed to save OBJ {out_path}: {e}")
