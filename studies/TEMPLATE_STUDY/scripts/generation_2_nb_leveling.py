"""This is a template script for generation 2 of simulation study, in which ones configures a
a Xsuite collider (including luminosity levelling and beam-beam) and tracks a given particle
distribution."""

# ==================================================================================================
# --- Imports
# ==================================================================================================

# Import standard library modules
import argparse
import contextlib
import logging
import os
import sys
import time
from pathlib import Path

# Import third-party modules
import numpy as np
import pandas as pd

# Import user-defined modules
from study_da.generate import XsuiteCollider, XsuiteTracking, BBCWInstaller
from study_da.utils import (
    load_dic_from_path,
    set_item_in_dic,
    write_dic_to_path,
)

# Set up the logger here if needed


# ==================================================================================================
# --- Script functions
# ==================================================================================================
def configure_collider(full_configuration):
    # Get configuration
    config_collider = full_configuration["config_collider"]
    ver_hllhc_optics = full_configuration["config_mad"]["ver_hllhc_optics"]
    ver_lhc_run = full_configuration["config_mad"]["ver_lhc_run"]
    ions = full_configuration["config_mad"]["ions"]
    collider_filepath = full_configuration["config_collider"][
        "path_collider_file_for_configuration_as_input"
    ]

    # Build object for configuring collider
    xc = XsuiteCollider(config_collider, collider_filepath, ver_hllhc_optics, ver_lhc_run, ions)

    # Load collider
    collider = xc.load_collider()

    # Install beam-beam
    xc.install_beam_beam_wrapper(collider)

    # Build trackers
    # For now, start with CPU tracker due to a bug with Xsuite
    # Refer to issue https://github.com/xsuite/xsuite/issues/450
    collider.build_trackers()  # (_context=context)

    # Set knobs
    xc.set_knobs(collider)

    # Match tune and chromaticity
    xc.match_tune_and_chroma(collider, match_linear_coupling_to_zero=True)

    # Set filling scheme
    xc.set_filling_and_bunch_tracked(ask_worst_bunch=False)

    # Compute the number of collisions in the different IPs
    n_collisions_ip1_and_5, n_collisions_ip2, n_collisions_ip8 = xc.compute_collision_from_scheme()

    # Do the leveling if requested
    if "config_lumi_leveling" in config_collider and not config_collider["skip_leveling"]:
        xc.level_ip1_5_by_bunch_intensity(collider, n_collisions_ip1_and_5)
        xc.level_ip2_8_by_separation(n_collisions_ip2, n_collisions_ip8, collider)
    else:
        logging.warning(
            "No leveling is done as no configuration has been provided, or skip_leveling"
            " is set to True."
        )

    # Add linear coupling
    xc.add_linear_coupling(collider)

    # Rematch tune and chromaticity
    xc.match_tune_and_chroma(collider, match_linear_coupling_to_zero=False)

    # Assert that tune, chromaticity and linear coupling are correct one last time
    xc.assert_tune_chroma_coupling(collider)

    # Update BBCW optics (alignment and sigmas)
    bbcwtools = BBCWInstaller(xc.ver_hllhc_optics, xc.ver_lhc_run)
    bbcwtools.update_optics_bbcw(collider,_with_checks = True)

    # Record beta functions in the configuration
    xc.record_beta_functions(collider)

    # Configure beam-beam if needed
    if not xc.config_beambeam["skip_beambeam"]:
        xc.configure_beam_beam(collider)

    # Power the bbcw
    bbcwtools.power_bbcw(collider, config_collider)

    # Update configuration with luminosity now that bb is known
    l_n_collisions = [
        n_collisions_ip1_and_5,
        n_collisions_ip2,
        n_collisions_ip1_and_5,
        n_collisions_ip8,
    ]
    xc.record_final_luminosity(collider, l_n_collisions)

    # Save collider to json (flag to save or not is inside function)
    xc.write_collider_to_disk(collider, full_configuration)

    # Get fingerprint
    fingerprint = xc.return_fingerprint(collider)

    return collider, fingerprint


def track_particles(full_configuration, collider, fingerprint):
    # Get emittances
    n_emitt_x = full_configuration["config_collider"]["config_beambeam"]["nemitt_x"]
    n_emitt_y = full_configuration["config_collider"]["config_beambeam"]["nemitt_y"]
    xst = XsuiteTracking(full_configuration["config_simulation"], n_emitt_x, n_emitt_y)

    # Prepare particle distribution
    particles, particle_id, l_amplitude, l_angle = xst.prepare_particle_distribution_for_tracking(
        collider
    )

    # Track
    particles_dict = xst.track(collider, particles)

    # Convert particles to dataframe
    particles_df = pd.DataFrame(particles_dict)

    # ! Very important, otherwise the particles will be mixed in each subset
    # Sort by parent_particle_id
    particles_df = particles_df.sort_values("parent_particle_id")

    # Assign the old id to the sorted dataframe
    particles_df["particle_id"] = particle_id

    # Register the amplitude and angle in the dataframe
    particles_df["normalized amplitude in xy-plane"] = l_amplitude
    particles_df["angle in xy-plane [deg]"] = l_angle * 180 / np.pi

    # Add some metadata to the output for better interpretability
    particles_df.attrs["hash"] = hash(fingerprint)
    particles_df.attrs["fingerprint"] = fingerprint
    particles_df.attrs["configuration"] = full_configuration
    particles_df.attrs["date"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # Save output
    particles_df.to_parquet(
        full_configuration["config_simulation"]["path_distribution_file_output"]
    )


def clean():
    # Remote the correction folder, and potential C files remaining
    with contextlib.suppress(Exception):
        os.system("rm -rf correction")
        os.system("rm -f *.cc")


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

from compute_residual import compute_residual
# ==================================================================================================
# --- Script for execution
# ==================================================================================================

if __name__ == "__main__":
    logging.info("Starting script to configure collider and track")

    # Parse potential arguments, e.g. to save output collider
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "-s", "--save", help="Save output collider for inspection", action="store_true"
    )
    args = aparser.parse_args()
    if args.save:
        dict_mutated_parameters["save_output_collider"] = True

    # Load full configuration
    full_configuration, ryaml = load_dic_from_path(path_configuration)

    # Mutate parameters in configuration
    for key, value in dict_mutated_parameters.items():
        set_item_in_dic(full_configuration, key, value)

    # Configure collider
    collider, fingerprint = configure_collider(full_configuration)

    # Drop updated configuration
    name_configuration = os.path.basename(path_configuration)
    write_dic_to_path(full_configuration, name_configuration, ryaml)

    # Duplicate of collider before tracking
    collider_original_1 = collider.copy()
    collider_original_2 = collider.copy()

    # # Track particles and save to disk
    # track_particles(full_configuration, collider, fingerprint)

    # Compute the residuals and save to disk
    compute_residual(config_file        = path_root_study + '/config_2_nlr.yaml', 
                     collider           = collider_original_1,
                     full_configuration = full_configuration)

    # Compute the residuals and save to disk
    compute_residual(config_file        = path_root_study + '/config_3_nlr.yaml', 
                     collider           = collider_original_2,
                     full_configuration = full_configuration)

    # Clean temporary files
    clean()

    logging.info("Script finished")