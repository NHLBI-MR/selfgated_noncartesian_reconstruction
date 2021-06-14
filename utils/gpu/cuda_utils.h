#pragma once

#include <gadgetron/cuNDArray.h>

namespace lit_sgncr_toolbox
{
    namespace cuda_utils
    {
        template <typename T>
        Gadgetron::cuNDArray<T> cuexp(Gadgetron::cuNDArray<T> &x);

        template <typename T>
        Gadgetron::cuNDArray<T> cucos(Gadgetron::cuNDArray<T> &x);

        template <typename T>
        Gadgetron::cuNDArray<T> cusin(Gadgetron::cuNDArray<T> &x);

    } // namespace cuda_util
} // namespace lit_sgncr_toolbox