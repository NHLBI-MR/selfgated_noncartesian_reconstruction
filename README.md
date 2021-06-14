# selfgated_noncartesian_reconstruction
Self-gated 3D Stack-of-Spirals Ultra-Short Echo-Time Pulmonary imaging at 0.55T

This reconstruction code will take free-breathing stack-of-spirals data and apply the following processing:
- self-navigation:
  -  Angular filter
  -  Bandpass filter
  -  PCA
  -  Coil-clustering
- gradient transfer function correction
- Concomitant field correction

Example data can be found here:

Installation instructions below.

# Pre-requisites
Gadgetron: https://github.com/gadgetron/gadgetron.git @ HASH
ISMRMRD: https://github.com/ismrmrd/ismrmrd @HASH

Gadgetron and MRD installation instructions can be found [here](https://github.com/gadgetron/gadgetron/wiki/Linux-Installation-%28Gadgetron-4%29)

Alternatively, a pre-installed Docker image can be found here: 

## Hardware
NVIDIA GPUs, with memory of at least XXGB. 

# Installation
```
git clone XXX
etc etc
```

# Inline Implementation
The data and gradient waveforms were streamed from the MRI scanner to the reconstructor using an interface with Siemens' reconstuctor called IceGadgetron. If you are a Siemens customer with a research agreement, you can request access to this repo.   


