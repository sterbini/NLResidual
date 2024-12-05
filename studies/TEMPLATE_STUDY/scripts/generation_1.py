"""This is a template script for generation 1 of simulation study, in which ones generates a
particle distribution and a collider from a MAD-X model."""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import standard library modules
import logging
import os
import sys
from pathlib import Path

# Import third-party modules
# Import user-defined modules
from study_da.generate import MadCollider, ParticlesDistribution, BBCWInstaller
from study_da.utils import (
    load_dic_from_path,
    set_item_in_dic,
    write_dic_to_path,
)

# Set up the logger here if needed


# ==================================================================================================
# --- Script functions
# ==================================================================================================
def build_distribution(config_particles):
    # Build object for generating particle distribution
    distr = ParticlesDistribution(config_particles)

    # Build particle distribution
    particle_list = distr.return_distribution_as_list()

    # Write particle distribution to file
    distr.write_particle_distribution_to_disk(particle_list)


def build_collider(config_mad):
    # Build object for generating collider from mad
    mc = MadCollider(config_mad)

    # Build mad model
    mad_b1b2, mad_b4 = mc.prepare_mad_collider()

    # Build collider from mad model
    collider = mc.build_collider(mad_b1b2, mad_b4)

    # Installing bbcw
    #------------------
    bbcwtools = BBCWInstaller(mc.ver_hllhc_optics, mc.ver_lhc_run)
    # Install and knobify bbcw
    bbcwtools.install_bbcw(collider)
    bbcwtools.knobify_kq_bbcw(collider,config_mad['bbcw_installation'])
    #------------------

    # Twiss to ensure everything is ok
    mc.activate_RF_and_twiss(collider)

    # Clean temporary files
    mc.clean_temporary_files()

    # Save collider to json
    mc.write_collider_to_disk(collider)


# ==================================================================================================
# --- Parameters placeholders definition
# ==================================================================================================
dict_mutated_parameters = {}  ###---parameters---###
path_configuration = "{}  ###---main_configuration---###"
path_root_study = "{}  ###---path_root_study---###"

# In case the placeholders have not been replaced, use default path
if path_configuration.startswith("{}"):
    path_configuration = "config.yaml"

if path_root_study.startswith("{}"):
    path_root_study = "."

sys.path.append(path_root_study)
# Import modules placed at the root of the study here

# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    logging.info("Starting script to build particle distribution and collider")

    # Load full configuration
    full_configuration, ryaml = load_dic_from_path(path_configuration)

    # Mutate parameters in configuration
    for key, value in dict_mutated_parameters.items():
        set_item_in_dic(full_configuration, key, value)

    # Dump configuration
    name_configuration = os.path.basename(path_configuration)
    write_dic_to_path(full_configuration, name_configuration, ryaml)

    # Build and save particle distribution
    build_distribution(full_configuration["config_particles"])

    # Build and save collider
    build_collider(full_configuration["config_mad"])

    logging.info("Script finished")


    if Path(full_configuration["config_mad"]['path_collider_file_for_configuration_as_output']).exists() or Path(full_configuration["config_mad"]['path_collider_file_for_configuration_as_output'] + '.zip').exists():
        Path('.finished').touch()
    