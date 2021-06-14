#pragma once

#include <gadgetron/mri_core_grappa.h>
#include <gadgetron/vector_td_utilities.h>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <gadgetron/cgSolver.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNFFT.h>
#include <gadgetron/hoNDFFT.h>
#include <numeric>
#include <random>
#include <gadgetron/NonCartesianTools.h>
#include <gadgetron/NFFTOperator.h>
#include <sstream>
#include <fstream>
#include <iostream>
#include <boost/range/combine.hpp>
#include <boost/range/algorithm.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/mri_core_coil_map_estimation.h>
#include <gadgetron/GenericReconBase.h>
#include <boost/hana/functional/iterate.hpp>
#include <gadgetron/cuNDArray_converter.h>
#include <gadgetron/ImageArraySendMixin.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include <gadgetron/b1_map.h>
#include <gadgetron/cudaDeviceManager.h>
#include <iterator>
#include "../spiral/SpiralBuffer.h"
#include <omp.h>
#include <gadgetron/mri_core_kspace_filter.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/cuSDC.h>
#include "../utils/gpu/cuda_utils.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;

namespace lit_sgncr_toolbox
{
    namespace utils
    {
        cuNDArray<float_complext> estimateCoilmaps_slice(cuNDArray<float_complext> &data);

        void attachHeadertoImageArray(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, const ISMRMRD::IsmrmrdHeader &h);

        void filterImagealongSlice(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma);
        void filterImage(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma);
        int selectCudaDevice();
                std::vector<int> FindCudaDevices(unsigned long);
        void setNumthreadsonGPU(int Number);
        template <typename T>
        void write_gpu_nd_array(cuNDArray<T> &data, std::string filename);
        template <typename T>
        void write_cpu_nd_array(hoNDArray<T> &data, std::string filename);
        template <typename T>
        cuNDArray<T> concat(std::vector<cuNDArray<T>> &arrays);
        template <typename T>
        hoNDArray<T> concat(std::vector<hoNDArray<T>> &arrays);
                float correlation(hoNDArray<float>a,hoNDArray<float>b);

        template <typename T>
        std::vector<T> sliceVec(std::vector<T> &v, int start, int end, int stride);
        std::vector<hoNDArray<float>> estimateDCF_slice(std::vector<std::vector<hoNDArray<floatd3>>> trajectories, std::vector<std::vector<hoNDArray<float>>> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                                        std::vector<size_t> image_dims_, bool fullySampled);

        template <template <class> class ARRAY> void set_data(ARRAY<float_complext> &sp_data, ARRAY<floatd3> &sp_traj, ARRAY<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                 boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, int currDev, bool useIterativeDCW, bool useIterativeDCWEstimated);

        arma::fmat33 lookup_PCS_DCS(std::string PCS_description);

        hoNDArray<float> estimateDCF(hoNDArray<floatd3> trajectories, hoNDArray<float> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                                  std::vector<size_t> image_dims_, bool fullySampled);
                 
        }
    } // namespace utils