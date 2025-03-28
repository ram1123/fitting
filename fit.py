import yaml
from modules.DistributionCompare import DistributionCompare

with open("config/plot_config.yaml") as f:
    config = yaml.safe_load(f)

year = config["year"]
control_region = config["control_region"]
directoryTag = config["directoryTag"]
input_paths_labels = config["input_paths_labels"]
fields_to_load = ["dimuon_mass", "wgt_nominal", "mu1_eta", "mu2_eta"]


comparer = DistributionCompare(year, input_paths_labels, fields_to_load, control_region, directoryTag)
comparer.load_data()

if config["plot_types"]["fit_z_peak"] and control_region in ["z-peak", "z_peak"]:
    if config["fit_categories"]["z-peak"]["binned"]:
        comparer.fit_dimuonInvariantMass_DCBXBW(suffix="Inclusive")
    if config["fit_categories"]["z-peak"]["unbinned"]:
        comparer.fit_dimuonInvariantMass_DCBXBW_Unbinned(suffix="Inclusive")
