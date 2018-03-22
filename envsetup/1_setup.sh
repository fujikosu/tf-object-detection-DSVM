echo 'Running NVidia CUDA 9.0 and CuDNN 7.x install'

source ./0_vars.sh

apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
apt-key adv --export --no-emit-version -a ${NVIDIA_GPGKEY_FPR} | tail -n +5 > cudasign.pub
echo "${NVIDIA_GPGKEY_SUM} cudasign.pub" | sha256sum -c --strict -
rm cudasign.pub

if [ ! -f /etc/apt/sources.list.d/cuda.list ]; then
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" | sudo tee --append /etc/apt/sources.list.d/cuda.list
else
if ! grep -Fxq "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" /etc/apt/sources.list.d/cuda.list; then
echo "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64 /" | sudo tee --append /etc/apt/sources.list.d/cuda.list
fi
fi

if [ ! -f /etc/apt/sources.list.d/nvidia-ml.list ]; then
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" | sudo tee --append /etc/apt/sources.list.d/nvidia-ml.list
else
if ! grep -Fxq "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" /etc/apt/sources.list.d/nvidia-ml.list; then
echo "deb http://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1604/x86_64 /" | sudo tee --append /etc/apt/sources.list.d/nvidia-ml.list
fi
fi

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    cuda-cudart-${CUDA_PKG_VERSION} \
    cuda-libraries-${CUDA_PKG_VERSION} \
    libnccl2=${NCCL_VERSION}-1+cuda${CUDA_VERSION} \
    libcudnn7=${CUDNN_VERSION}-1+cuda${CUDA_VERSION}

sudo rm /usr/local/cuda
sudo ln -s cuda-${CUDA_VERSION} /usr/local/cuda