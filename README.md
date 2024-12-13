# NLResidual

## Minimal installation
The `installme.sh` script can be used to install additionnal packages on top of a basic distribution (the file should be inspected and modified accordingly).
The minimal installation simply requires the study-DA package on top of xsuite, so you can do the following:
```bash
git clone --recurse-submodules https://github.com/pbelange/study-DA.git ./study-DA
pip install -e ./study-DA

# Change to the BBCW branch
git checkout feature/wires
```


## Important files:

The important files to potentially modify for the wire are:
```bash
# All bbcw functionnalities
./study-DA/study_da/generate/master_classes/xsuite_bbcw.py

# version-specific installation parameters
# HL
./study-DA/study_da/generate/version_specific_files/hllhc16/bbcw_installation.py
# LHC
./study-DA/study_da/generate/version_specific_files/runIII/bbcw_installation.py
```

## Quick explanations:
The workflow is as following:
1. Prepare the study/scripts. `./studies/TEMPLATE_STUDY/configs` contains the base config of the mask + the config for the NLR
2. Prepare the generation scripts. `studies/TEMPLATE_STUDY/scripts/generation_1.py` and `generation_2.py` are the scripts which will be copied in the tree (the template)

You're ready to create the tree
1. Modify `./studies/TEMPLATE_STUDY/01_config_tree.yaml` to scan the relevant parameters
2. Run `./studies/TEMPLATE_STUDY/02_build_tree.py` to create the tree
3. Go run `./studies/TEMPLATE_STUDY/trees/BBCW_SCAN_2025/ID_0_generation_1/generation_1.py` to have a collider with wires installed
