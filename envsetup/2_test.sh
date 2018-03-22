echo 'Running NVidia CUDA 9.0 and CuDNN 7.x install'

source ./0_vars.sh

[[ ":$PATH:" != *":/usr/local/cuda/bin:"* ]] && export PATH=/usr/local/cuda/bin:${PATH}

# assume nouveau is blacklisted already

echo 'Installing CUDA command line tools'

# install command line tools including libcupti.dev
sudo apt-get install cuda-command-line-tools-${CUDA_PKG_VERSION}

echo 'Installing CUDA samples'
# install samples
sudo apt-get install -y cuda-samples-${CUDA_PKG_VERSION}
echo 'Copying CUDA samples to home directory'
# copy samples to homedir
cuda-install-samples-${CUDA_VERSION}.sh ~
# run samples
echo 'Compiling and running deviceQuery sample'
cd ~/NVIDIA_CUDA-${CUDA_VERSION}_Samples/1_Utilities/deviceQuery
make
cd ~/NVIDIA_CUDA-${CUDA_VERSION}_Samples/bin/x86_64/linux/release/
./deviceQuery

echo 'Getting NVIDIA driver info'
# Print out driver version
cat /proc/driver/nvidia/version

