#!/bin/bash

# This script loads the necessary modules or dependencies for project prj_ICNet_pytorch
#
# Make it executable:
# $ chmod +x load_modules.sh
#
# Usage:
#    $ . load_modules.sh
# OR $ source load_modules.sh  (This ensures that environment changes persist in your terminal session)
# 
# Run it under the home directory of repo `prj_ICNet_pytorch`
#

# Load a Python module
echo "Loading Python module..."
module load python/3.12.4 || { echo "Failed to load Python module"; exit 1; }

# Load opencv-python module
echo "Loading OpenCV module..."
module load opencv/4.10.0 || { echo "Failed to load OpenCV module"; exit 1; }

# Activate the virtual environment
echo "Activating virtual environment..."
source env/bin/activate || { echo "Failed to activate virtual environment"; exit 1; }