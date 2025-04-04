import numpy as np
import pandas as pd
import os

def apply_transformation(points, transformation_matrix):
    """
    Apply a 4x4 transformation matrix to a set of points using homogeneous coordinates.
    :param points: Nx3 numpy array of points.
    :param transformation_matrix: 4x4 transformation matrix.
    :return: Transformed Nx3 matrix of points.
    """
    if transformation_matrix.shape != (4, 4):
        raise ValueError("Transformation matrix must be 4x4.")
    
    # Convert to homogeneous coordinates
    ones = np.ones((points.shape[0], 1))
    points_h = np.hstack((points, ones))  # Nx4 matrix
    
    # Apply transformation
    transformed_points_h = points_h @ transformation_matrix.T  # Nx4 matrix
    return transformed_points_h[:, :3]  # Convert back to Nx3

def load_xyz_to_numpy(filename, delimiter=' '):
    # genfromtxt read csv file
    data = np.genfromtxt(filename, delimiter=delimiter, skip_header=0, usecols=(0, 1, 2))
    return data

def load_transformation_matrix(filename):
    """
    Load a transformation matrix from a text file.
    
    :param filename: The file path where the matrix is saved.
    :return: The transformation matrix as a numpy array.
    """
    return np.loadtxt(filename)

def save_to_xyz(points, filename):
    if isinstance(points, np.ndarray):
        points = pd.DataFrame(points)
    with open(filename, 'w') as f:
        for _,row in points.iterrows():
            f.write(f"{row[0]} {row[1]} {row[2]}\n")

def process_xyz_files(folder_path, ScanningRobot_CMM_transformation_matrix, CMM_WeldingRobot_transformation_matrix):
    """
    Process all .xyz files in a given folder, apply the transformation, and save new files.
    """
    xyz_files = [f for f in os.listdir(folder_path) if f.endswith('.xyz')]
    
    for xyz_file in xyz_files:
        file_path = os.path.join(folder_path, xyz_file)
        points = load_xyz_to_numpy(file_path)
        transformed_points = apply_transformation(points, ScanningRobot_CMM_transformation_matrix)
        final_transformed_points = apply_transformation(transformed_points, CMM_WeldingRobot_transformation_matrix)
        new_filename = xyz_file.replace('_no_transformation.xyz', '_calibrated.xyz')
        save_to_xyz(final_transformed_points, os.path.join(folder_path, new_filename))
        print(f"Processed {xyz_file} -> {new_filename}")
    return new_filename
        
def main():
    CMM_WeldingRobot_transformation_matrix = load_transformation_matrix('Welding_robot_transformation_matrix.txt')
    ScanningRobot_CMM_transformation_matrix = load_transformation_matrix('Scanning_robot_transformation_matrix.txt')
    folder_path = os.path.dirname(os.path.abspath(__file__))
    new_filename = process_xyz_files(folder_path, ScanningRobot_CMM_transformation_matrix, CMM_WeldingRobot_transformation_matrix)
    return new_filename