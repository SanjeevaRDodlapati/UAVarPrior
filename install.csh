#!/bin/tcsh

# Installation script for UAVarPrior

echo "Installing UAVarPrior..."

# Check if we're in the right directory
if (! -e setup.py) then
    echo "Error: This script must be run from the UAVarPrior root directory"
    exit 1
endif

# Check if Python is available
python --version >& /dev/null
if ($status != 0) then
    echo "Error: Python is not available"
    exit 1
endif

# Clean up old installation files
echo "Cleaning up old installation files..."
python scripts/clean_installation.py

# Install the package
echo "Installing package..."
pip install -e .

# Verify installation
echo "Verifying installation..."
python -c "import uavarprior; print(f'UAVarPrior version: {uavarprior.__version__}')"

if ($status == 0) then
    echo "Installation successful!"
    echo "To get started, try: uavarprior --help"
else
    echo "Installation verification failed!"
endif
