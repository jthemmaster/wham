import numpy as np
from os import path
import os
from dataclasses import dataclass
from numpy import ndarray
from io import TextIOWrapper
from numpy import zeros

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


class WHAM:
    def __init__(self, histograms, kBT, tolerance=1e-6, max_iterations=1000):
        """
        histograms: List of numpy arrays, each array is a histogram from a different umbrella window
        kBT: Thermal energy, k_B * T, where k_B is Boltzmann constant and T is temperature
        tolerance: Convergence tolerance
        max_iterations: Maximum number of iterations for the algorithm
        """
        self.histograms = histograms
        self.kBT = kBT
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.free_energies = np.zeros(len(histograms))  # Initial guess

    def calculate_free_energies(self):
        """
        Main method to calculate the free energies
        """
        for _ in range(self.max_iterations):
            free_energies_old = self.free_energies.copy()
            self.update_free_energies()
            if np.max(np.abs(self.free_energies - free_energies_old)) < self.tolerance:
                break
        return self.free_energies

    def update_free_energies(self):
        """
        Update the free energy estimates in each iteration
        """
        # Compute weighted averages
        weights = np.exp(-self.free_energies / self.kBT)
        denominator = np.sum([h * w for h, w in zip(self.histograms, weights)], axis=0)
        
        for i, histogram in enumerate(self.histograms):
            numerator = np.sum(histogram / denominator)
            self.free_energies[i] = -self.kBT * np.log(numerator)
    
    def plot_free_energy_landscape(self):
        """
        Optional: Implement a method to plot the free energy landscape
        """
        pass  # You can use matplotlib or another plotting library here

def get_all_folder_names(path):
    folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f)) and f.endswith("A")]
    folders.sort()
    return folders

def get_umbrella_centers(path):
    folders = get_all_folder_names(path)
    return [float(folder.replace("A","")) for folder in folders]

def get_bin_width(umbrella_centers):
    return umbrella_centers[1] - umbrella_centers[0]

def get_reaction_coordinate(xyz, collective_variable_indices,frame):
    coordinate_1 = xyz.coordinates[frame][collective_variable_indices[0]]
    coordinate_2 = xyz.coordinates[frame][collective_variable_indices[1]]
    #calculate distance between two atoms
    return np.linalg.norm(coordinate_1 - coordinate_2)

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

def create_histogram_for_window(reaction_coordinate_values_for_window, num_bins, umbrella_centers, bin_width):
    histogram = np.zeros(num_bins)
    for reaction_coordinate_value in reaction_coordinate_values_for_window:
        #check if reaction coordinate value is within each umbrella window
        for i, umbrella_center in enumerate(umbrella_centers):
            if reaction_coordinate_value >= umbrella_center - bin_width / 2 and reaction_coordinate_value < umbrella_center + bin_width / 2:
                histogram[i] += 1
                break
    # Normalize the histogram
    histogram = histogram / np.sum(histogram) * 100
    return histogram

def get_histograms(reaction_coordinate_values_all_windows, num_bins, umbrella_centers, bin_width):
    histograms = []
    for i, reaction_coordinate_values_for_window in enumerate(reaction_coordinate_values_all_windows):
        histogram = create_histogram_for_window(reaction_coordinate_values_for_window, num_bins, umbrella_centers, bin_width)
        histograms.append(histogram)
    return histograms

umbrella_centers = get_umbrella_centers(resources_path)
num_bins = len(umbrella_centers)
bin_width = get_bin_width(umbrella_centers)
collective_variable_indices = [145,277]
reaction_coordinate_values_all_windows = get_reaction_coordinate_values_for_all_windows(resources_path, collective_variable_indices)

histograms = get_histograms(reaction_coordinate_values_all_windows, num_bins, umbrella_centers, bin_width)

print(f"first histogram: {histograms[0]}\n")
print(f"second histogram: {histograms[1]}\n")
print(f"third histogram: {histograms[2]}\n")
print(f"fourth histogram: {histograms[3]}\n")

second_histogram = histograms[30]

import matplotlib.pyplot as plt
#make history plot for first window
print(umbrella_centers)
plt.bar(umbrella_centers, second_histogram, width=bin_width)
#change histogram tick labels
plt.xticks(umbrella_centers,rotation=90)
plt.xlabel("bins")
plt.ylabel("Probability [%]")
plt.title("Histogram of distance for first window")
plt.show()

#now do wham
kBT = 0.0083144621 * 300
wham = WHAM(histograms, kBT)
free_energies = wham.calculate_free_energies()
print(free_energies)
