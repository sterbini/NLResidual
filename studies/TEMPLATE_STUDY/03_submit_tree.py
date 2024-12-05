# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import standard library modules
import os
import argparse

# Import third-party modules
# Import user-defined modules
from study_da import submit



# ==================================================================================================
# --- Script to submit the study
# ==================================================================================================
def main(path_tree,name_main_config = 'configs/config_1_base.yaml'):
    # In case gen_1 is submitted locally
    dic_additional_commands_per_gen = {
        # To copy back the particles folder from the first generation if submitted to HTC
        1 : "cp -r particles $path_job/particles \n",
    }

    # Dependencies for the executable of each generation. Only needed if one uses HTC or Slurm.
    dic_dependencies_per_gen = {
        1: ["acc-models-lhc"],
        2: ["path_collider_file_for_configuration_as_input", "path_distribution_folder_input"],
    }
    

    # Submit the study
    submit(
        path_tree=path_tree,
        path_python_environment_container="/usr/local/DA_study/miniforge_docker",
        path_container_image="/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cdroin/da-study-docker:757f55da",
        dic_dependencies_per_gen=dic_dependencies_per_gen,
        name_config=name_main_config,
        dic_additional_commands_per_gen=dic_additional_commands_per_gen,
        one_generation_at_a_time=True,
    )


# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":

    # Parse potential arguments, e.g. to save output collider
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "-t", "--target", help="Path of tree to submit", default = 'tree.yaml'
    )
    args = aparser.parse_args()
    main(path_tree=args.target)
