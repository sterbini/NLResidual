# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import standard library modules
import os

# Import third-party modules
# Import user-defined modules
from study_da import create, submit
from study_da.utils import load_template_configuration_as_dic, write_dic_to_path

# ==================================================================================================
# --- Script to generate a study
# ==================================================================================================

# Tree maker config
config_tree  = "01_config_tree.yaml"


# # Load some configuration to modify
# config_gen_1    = None
# config_gen_2    = None


# # config_gen_1  = "configs/config_1_base.yaml"
# #--------------------------------------------------
# if config_gen_1 is not None:
#     config, ryaml = load_template_configuration_as_dic(config_gen_1)

#     # Do changes if needed
#     # config["config_simulation"]["n_turns"] = 1000000

#     # Drop the configuration locally
#     write_dic_to_path(config, config_gen_1, ryaml)
# #--------------------------------------------------

 

# Now generate the study in the local directory
path_tree, name_main_config = create(path_config_scan=config_tree, force_overwrite=True,add_prefix_to_folder_names=True)
print(path_tree, name_main_config)
# path_tree, name_main_config = create(path_config_scan=config_tree, force_overwrite=True)

