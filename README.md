# selfgated_noncartesian_reconstruction
Self-gated 3D Stack-of-Spirals Ultra-Short Echo-Time Pulmonary imaging at 0.55T


# Installation
This toolbox was designed to be used with GPUs and cannot be installed without cuda support. This release was tested with CUDA 11.0 and above but may work with older releases of cuda. 

## Install Pre-requisites:
apt install --no-install-recommends --no-install-suggests --yes \
  apt-utils build-essential cmake cpio cython3 gcc-multilib git-core h5utils hdf5-tools jove jq libace-dev libarmadillo-dev libatlas-base-dev libcrypto++-dev           libfftw3-dev libfreetype6-dev libgtest-dev libhdf5-serial-dev liblapack-dev liblapacke-dev libopenblas-base 
  libopenblas-dev \
  libplplot-dev \
  libpugixml-dev \
  libxml2-dev \
  libxslt-dev \
  librocksdb-dev \
  net-tools \
  ninja-build \
  pkg-config \
  python3-dev \
  python3-pip \
  software-properties-common \
  supervisor \
  wget \
  googletest \
  googletest-tools \
  librange-v3-dev \
  nlohmann-json3-dev \
  libboost-all-dev
  

`mkdir mrprogs && cd mrprogs`

Install Gadgetron: 
`git clone https://github.com/gadgetron/gadgetron.git`
`cd gadgetron`
`mkdir build`
`mkdir /home/$USER/local`
`cmake ../ -DCMAKE_INSTALL_PREFIX=/home/$USER/local -G Ninja`
`ninja install`




