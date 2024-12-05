# Create your own virtual environment in a new folder
source /home/phbelang/base/miniforge/bin/activate
ENV_NAME="py-NLR"
mkdir ./Executables
conda update -n base -c conda-forge conda
conda create -n $ENV_NAME python=3.12.6
conda activate $ENV_NAME


# Install generic python packages
#========================================
pip install jupyterlab
pip install ipywidgets
pip install PyYAML
pip install pyarrow
pip install pandas
pip install dask
pip install dask[dataframe]
pip install matplotlib
pip install scipy
pip install ipympl
pip install ruamel.yaml
pip install rich

# Adding the jupyter kernel to the list of kernels
python -m ipykernel install --user --name $ENV_NAME --display-name "$ENV_NAME"


# Installing fortran and other compilers
conda install compilers cmake
if [ "$(uname)" == "Linux" ]; then
    # Do something under Linux platform
    conda install gcc_linux-64 gxx_linux-64
fi
# If needed, could add the same line for Mac eventually: conda install clang_osx-arm64 clangxx_osx-arm64


# Install CERN packages
#=========================================
git clone --recurse-submodules https://github.com/pbelange/study-DA.git ./study-DA
pip install -e ./study-DA


git clone https://github.com/pbelange/nafflib.git ./Executables/$ENV_NAME/nafflib
pip install -e ./Executables/$ENV_NAME/nafflib


git clone https://github.com/xsuite/xobjects ./Executables/$ENV_NAME/xobjects
pip install -e ./Executables/$ENV_NAME/xobjects

git clone https://github.com/xsuite/xdeps ./Executables/$ENV_NAME/xdeps
pip install -e ./Executables/$ENV_NAME/xdeps

git clone https://github.com/xsuite/xpart ./Executables/$ENV_NAME/xpart
pip install -e ./Executables/$ENV_NAME/xpart

git clone https://github.com/xsuite/xtrack ./Executables/$ENV_NAME/xtrack
pip install -e ./Executables/$ENV_NAME/xtrack

git clone https://github.com/xsuite/xmask ./Executables/$ENV_NAME/xmask
pip install -e ./Executables/$ENV_NAME/xmask

git clone https://github.com/xsuite/xfields ./Executables/$ENV_NAME/xfields
pip install -e ./Executables/$ENV_NAME/xfields


# Download outsourced files
#=========================================
cd ./Executables/$ENV_NAME/xmask
git config --global --add safe.directory ./
git submodule init
git submodule update

git clone https://github.com/xsuite/xsuite ./Executables/$ENV_NAME/xsuite
pip install -e ./Executables/$ENV_NAME/xsuite
xsuite-prebuild regenerate

