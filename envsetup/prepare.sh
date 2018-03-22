source ./install.sh

echo 'Preparing environment. Make sure you run with source command'

echo 'Adding /anaconda/lib to use zlib 1.2.11'
[[ ":$LD_LIBRARY_PATH:" != *":/anaconda/lib:"* ]] && echo 'export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/anaconda/lib' >> ~/.bash_profile

echo 'Setting up Python virtual environment in ~/tensorflow-py3ve'
#
# VirtualEnv Setup
#
source ~/tensorflow-py3ve/bin/activate

