import pandas as pd
import numpy as np
from os import path 
import os
dir_path = os.path.dirname(os.path.realpath(__file__))

# Read the file into a DataFrame, skipping rows that start with '#'
# Assuming your file has two columns without a header
df_global_historgam = pd.read_csv(f"{dir_path}/global_histogram.xy", sep="\s+", header=None, comment='#', names=["reaction_coordinate", "probability"],skiprows=2)
df_umbrella_integration = pd.read_csv(f"{dir_path}/fe_ui.xy", sep="\s+", header=None, comment='#', names=["reaction_coordinate", "energy"],skiprows=2)
df_wham = pd.read_csv(f"{dir_path}/fe_wham.xy", sep="\s+", header=None, comment='#', names=["reaction_coordinate", "energy"],skiprows=2)

#hartree to kcal/mol

df_umbrella_integration["energy"] = df_umbrella_integration["energy"] * 627.509
df_wham["energy"] = df_wham["energy"] * 627.509
#make matplotlib histogram
import matplotlib.pyplot as plt
#plot all three histograms
#plt.plot(df_global_historgam["reaction_coordinate"], df_global_historgam["probability"], label="global histogram")
plt.plot(df_umbrella_integration["reaction_coordinate"], df_umbrella_integration["energy"], label="umbrella integration")
plt.plot(df_wham["reaction_coordinate"], df_wham["energy"], label="wham")
plt.legend()
plt.xlabel("reaction coordinate [Bohr]")
plt.ylabel("energy [kcal/mol]")
plt.show()
        