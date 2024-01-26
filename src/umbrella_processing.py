import numpy as np
from os import path
import os
from dataclasses import dataclass
from numpy import ndarray
from io import TextIOWrapper
from numpy import zeros
import subprocess

dir_path = os.path.dirname(os.path.realpath(__file__))
#get the path /resources 
resources_path = path.join(dir_path, "../resources/pre_umbrella")

@dataclass
class XYZFile:
    comment: str
    number_of_atoms: int
    atoms: list[str]
    coordinates: ndarray
    number_of_frames: int

class XYZReader:
    @staticmethod
    def read(filename: str) -> XYZFile:
        with open(filename, "r") as xyz_file:
            number_of_atoms = int(xyz_file.readline().strip())
            number_of_lines = sum(1 for _ in xyz_file) + 1
            number_of_frames = number_of_lines // (number_of_atoms + 2)
        atoms = [None] * number_of_atoms
        coordinates = zeros((number_of_frames, number_of_atoms, 3))
        comment = ""
        with open(filename, "r") as xyz_file:
            for frame_index in range(number_of_frames):
                _ = xyz_file.readline()
                comment = xyz_file.readline().strip()
                for atom_index in range(number_of_atoms):
                    xyz_line = xyz_file.readline()
                    xyz_line_splitted = xyz_line.split()
                    atoms[atom_index] = xyz_line_splitted[0]
                    coordinates[frame_index, atom_index, 0] = float(xyz_line_splitted[1])
                    coordinates[frame_index, atom_index, 1] = float(xyz_line_splitted[2])
                    coordinates[frame_index, atom_index, 2] = float(xyz_line_splitted[3])
        return XYZFile(comment, number_of_atoms, atoms, coordinates, number_of_frames)

class XYZWriter:
    @staticmethod
    def write_xyz(xyz: XYZFile, xyz_file: TextIOWrapper) -> None:
        for frame_index in range(xyz.number_of_frames):
            print(xyz.number_of_atoms, file=xyz_file)
            print(xyz.comment, file=xyz_file)
            for atom_index, atoms in enumerate(xyz.atoms):
                print(f"{xyz.atoms[atom_index]} {xyz.coordinates[frame_index][atom_index][0]} {xyz.coordinates[frame_index][atom_index][1]} {xyz.coordinates[frame_index][atom_index][2]}", file=xyz_file)

def get_all_folder_names(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.endswith("A")]
    folders.sort()
    return folders

def get_umbrella_centers(path):
    folders = get_all_folder_names(path)
    return [float(folder.replace("A","")) * 1.88973 for folder in folders] #to bohr

def get_reaction_coordinate(xyz, collective_variable_indices,frame):
    coordinate_1 = xyz.coordinates[frame][collective_variable_indices[0]]
    coordinate_2 = xyz.coordinates[frame][collective_variable_indices[1]]
    #calculate distance between two atoms
    return np.linalg.norm(coordinate_1 - coordinate_2) * 1.88973 #to bohr

def get_reaction_coordinate_values_for_window(path_to_window,collective_variable_indices):
    xyz = XYZReader.read(f"{path_to_window}/output/run.xyz")
    return [get_reaction_coordinate(xyz, collective_variable_indices,frame) for frame in range(xyz.number_of_frames)]

def get_reaction_coordinate_values_for_all_windows(path,collective_variable_indices):
    folders = get_all_folder_names(path)
    reaction_coordinate_values_all_windows = []
    for folder in folders:
        path_to_window = os.path.join(path, folder)
        reaction_coordinate_values_for_window = get_reaction_coordinate_values_for_window(path_to_window,collective_variable_indices)
        reaction_coordinate_values_all_windows.append(reaction_coordinate_values_for_window)
    return reaction_coordinate_values_all_windows

def create_input_files(path, reaction_coordinate_values_all_windows, force_constant, umbrella_centers):
    #create data folder if it does not exist
    subprocess.run(["mkdir", "-p", f"{path}/data"])
        
    for index,window in enumerate(umbrella_centers):
        #create file
        with open(f"{path}/data/{window:.2f}.dat", "w") as file:
            #write reaction coordinate values
            for value in reaction_coordinate_values_all_windows[index]:
                file.write(f"{value} {force_constant} {float(window)}\n")

force_constant = 0.1
umbrella_centers = get_umbrella_centers(resources_path)
reaction_coordinate_values_all_windows = get_reaction_coordinate_values_for_all_windows(resources_path,[145,277])
print(min(reaction_coordinate_values_all_windows[0]))
print(max(reaction_coordinate_values_all_windows[-1]))

create_input_files(dir_path, reaction_coordinate_values_all_windows, force_constant, umbrella_centers)




    



