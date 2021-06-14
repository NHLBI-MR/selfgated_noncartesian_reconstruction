#pragma once
#include <gadgetron/hoNDArray.h>
#include <ismrmrd/ismrmrd.h>

namespace Gadgetron {
template <template<class> class ARRAY,typename T = float_complext,unsigned int D = 2>
    struct SpiralBuffer { 
        ARRAY<T> data;
        ARRAY<vector_td<float,D>> trajectory;
        ARRAY<float> dcw;
        std::vector<ISMRMRD::AcquisitionHeader> headers;
    };
}