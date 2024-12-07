#!/bin/bash

# Path to your installme.sh file
INSTALLME_PATH="./installme.sh"

# Extract ENV_NAME value from installme.sh
ENV_NAME=$(grep '^ENV_NAME=' $INSTALLME_PATH | cut -d '"' -f 2)

source ~/base/miniforge/bin/activate $ENV_NAME