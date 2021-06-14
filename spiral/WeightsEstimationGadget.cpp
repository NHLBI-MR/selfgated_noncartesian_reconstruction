#include <gadgetron/Node.h>
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
// #include <boost/range/combine.hpp>
// #include <boost/range/algorithm.hpp>
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
#include <iterator>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoCgSolver.h>
#include <gadgetron/hoNDImage_util.h>
#include <gadgetron/cuNonCartesianSenseOperator.h>
#include <omp.h>
#include <gadgetron/cuCgSolver.h>
#include <gadgetron/cuNlcgSolver.h>
#include <gadgetron/cuCgPreconditioner.h>
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuImageOperator.h>
#include <gadgetron/cuTvOperator.h>
#include <gadgetron/cuTvPicsOperator.h>
#include "../spiral/SpiralBuffer.h"
#include <gadgetron/mri_core_kspace_filter.h>
#include <gadgetron/ImageIOBase.h>
#include <gadgetron/hoNDArray_fileio.h>
#include "../utils/util_functions.h"
#include <gadgetron/cuSDC.h>
#include "../utils/concomitantFieldCorrection/mri_concomitant_field_correction.h"
#include "../utils/gpu/cuda_utils.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

class WeightsEstimationGadget : public ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                                                   Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;
    bool csm_calculated_ = false;
    cuCgSolver<float_complext> cg_;

    boost::shared_ptr<cuNFFT_plan<float, 3>> nfft_plan_;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;
    boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>> E_;
    boost::shared_ptr<cuCgPreconditioner<float_complext>> D_;
    boost::shared_ptr<cuImageOperator<float_complext>> R_;
    Gadgetron::ImageIOAnalyze gt_exporter_;
    boost::shared_ptr<cuNDArray<float>> _precon_weights;
    boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

    WeightsEstimationGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                                                                                                             Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>(context, props)
    {
        kernel_width_ = 5.5;
        verbose = false;
    }
    std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj)
    {
        auto sl_traj = arma::vec(sp_traj.get_size(1));
        for (auto ii = 0; ii < sp_traj.get_size(1); ii++)
            sl_traj[ii] = sp_traj(1, ii)[2];

        auto z_encodes = arma::vec(arma::unique(sl_traj));
        //z_encodes.print();

        std::vector<arma::uvec> slice_indexes;
        for (auto ii = 0; ii < z_encodes.n_elem; ii++)
        {
            arma::uvec temp = (find(sl_traj == z_encodes[ii]));
            slice_indexes.push_back(temp);
            //  slice_indexes[ii].print();
        }

        return slice_indexes;
    }

    template <unsigned int D>
    cuNDArray<float> estimate_DCF(cuNDArray<vector_td<float, D>> &traj, cuNDArray<float> &dcw, std::vector<size_t> image_dims)
    {
        // GadgetronTimer timer("estimate_DCF");

        std::vector<size_t> flat_dims = {traj.get_number_of_elements()};
        cuNDArray<vector_td<float, D>> flat_traj(flat_dims, traj.get_data_ptr());
        traj = cuNDArray<vector_td<float, D>>(flat_traj);
        cuNDArray<float> flat_dcw(flat_dims, dcw.get_data_ptr());
        dcw = cuNDArray<float>(flat_dcw);
        if (D == 2)
            image_dims.pop_back();

        if (useIterativeDCWEstimated)
        {
            auto temp = *(Gadgetron::estimate_dcw<float, D>(&traj,
                                                            &dcw,
                                                            from_std_vector<size_t, D>(image_dims),
                                                            oversampling_factor_,
                                                            size_t(iterations), kernel_width_, ConvolutionType::ATOMIC));
            dcw = cuNDArray<float>(temp);
        }
        else
        {
            auto temp = *(Gadgetron::estimate_dcw<float, D>(&traj,
                                                            from_std_vector<size_t, D>(image_dims),
                                                            oversampling_factor_,
                                                            size_t(iterations), kernel_width_, ConvolutionType::ATOMIC));
            dcw = cuNDArray<float>(temp);
        }

        return dcw;
    }

    hoNDArray<floatd2> traj3Dto2D(hoNDArray<floatd3> &sp_traj)
    {
        auto dims = sp_traj.get_dimensions();
        auto traj = hoNDArray<floatd2>(dims);

        auto traj_ptr = traj.get_data_ptr();
        auto ptr = sp_traj.get_data_ptr();

        for (size_t i = 0; i < sp_traj.get_number_of_elements(); i++)
        {

            traj_ptr[i][0] = ptr[i][0];
            traj_ptr[i][1] = ptr[i][1];
        }

        return traj;
    }
    hoNDArray<float> estimate_DCF_slice(hoNDArray<floatd3> &sp_traj, hoNDArray<float> &sp_dcw, std::vector<size_t> image_dims)
    {
        hoNDArray<float> hodcw(sp_dcw.get_size(0), sp_dcw.get_size(1));
        auto slice_index = extractSlices(sp_traj);
        auto traj_2D = traj3Dto2D(sp_traj);
        bool estimate_oneslice = true;

        auto eligibleGPUs = lit_sgncr_toolbox::utils::FindCudaDevices(0);
        while (eligibleGPUs.size() > maxDevices)
            eligibleGPUs.pop_back();

        for (auto ii = 1; ii < slice_index.size(); ii++)
        {
            if (slice_index[ii - 1].n_elem != slice_index[ii].n_elem)
                estimate_oneslice = false;
        }
        std::string pout = estimate_oneslice ? "Fully Sampled Estimating only one slice" : "Estimating DCF for all slices";
        GDEBUG_STREAM(pout);

#pragma omp parallel for num_threads(eligibleGPUs.size())
        for (auto ii = 0; ii < (estimate_oneslice ? 1 : slice_index.size()); ii++)
        {
            auto slvec = slice_index[ii];
            GDEBUG_STREAM("Selected Device: " << ii % (eligibleGPUs.size()));
            cudaSetDevice(ii % (eligibleGPUs.size()));
            hoNDArray<floatd2> temp_traj({sp_traj.get_size(0), slvec.n_elem});
            hoNDArray<float> temp_dcw({sp_traj.get_size(0), slvec.n_elem});
            for (auto jj = 0; jj < slvec.n_elem; jj++)
            {
                temp_traj(slice, jj) = traj_2D(slice, slvec[jj]);
                temp_dcw(slice, jj) = sp_dcw(slice, slvec[jj]);
            }
            auto cutraj = cuNDArray<floatd2>(temp_traj);
            auto cudcw = cuNDArray<float>(temp_dcw);
            cudcw = estimate_DCF<2>(cutraj, cudcw, image_dims_);
            temp_dcw = *cudcw.to_host();
            temp_dcw.reshape(sp_dcw.get_size(0), -1);

            for (auto kk = ii; kk < (estimate_oneslice ? slice_index.size() : ii + 1); kk++)
            {
                auto slvec = slice_index[kk];
                for (auto jj = 0; jj < slvec.n_elem; jj++)
                {
                    hodcw(slice, slvec[jj]) = temp_dcw(slice, jj);
                }
            }
        }
        return hodcw;
    }

    void process(InputChannel<Core::variant<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>,
                                            Gadgetron::SpiralBuffer<hoNDArray, float_complext, 2>, Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>> &in,
                 OutputChannel &out) override
    {

        auto matrixSize = this->header.encoding.front().reconSpace.matrixSize;
        auto fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();

        oversampling_factor_ = overSampling;
        kernel_width_ = kernelWidth;
        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;
        constexpr double GAMMA = 4258.0; /* Hz/G */

        image_dims_.push_back(matrixSize.x);
        image_dims_.push_back(matrixSize.y);
        if (this->header.encoding.front().encodedSpace.matrixSize.z % warp_size != 0)
            image_dims_.push_back(warp_size * (this->header.encoding.front().encodedSpace.matrixSize.z / warp_size + 1));
        else
            image_dims_.push_back(this->header.encoding.front().encodedSpace.matrixSize.z);

        using namespace Gadgetron::Indexing;
        for (auto message : in)
        {
            int selectedDevice = lit_sgncr_toolbox::utils::selectCudaDevice();
            cudaSetDevice(selectedDevice);
            std::shared_ptr<hoNDArray<float>> dcw;
            std::shared_ptr<cuNDArray<floatd3>> traj;

            GadgetronTimer timer("Iterative Estimation of Weights:");

            if (holds_alternative<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>>(message))
            {
                auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>>(message);

                auto hodcw = estimate_DCF_slice(*sp_traj.to_host(), *sp_dcw.to_host(), image_dims_);
                dcw = std::make_shared<hoNDArray<float>>((hodcw));
            }
            else if (holds_alternative<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message))
            {
                auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message);

                if (slicewiseDCF)
                {
                    auto hodcw = estimate_DCF_slice(sp_traj, sp_dcw, image_dims_);
                    dcw = std::make_shared<hoNDArray<float>>((hodcw));
                }
                else
                {
                    auto slice_index = extractSlices(sp_traj);
                    auto traj_2D = traj3Dto2D(sp_traj);
                    bool estimate_oneslice = true;

                    for (auto ii = 1; ii < slice_index.size(); ii++)
                    {
                        if (slice_index[ii - 1].n_elem != slice_index[ii].n_elem)
                            estimate_oneslice = false;
                    }
                    if (estimate_oneslice && slicewiseDCF_FS)
                    {
                        auto hodcw = estimate_DCF_slice(sp_traj, sp_dcw, image_dims_);
                        dcw = std::make_shared<hoNDArray<float>>(hodcw);
                    }
                    else
                    {
                        auto cutraj = cuNDArray<floatd3>(sp_traj);
                        auto cudcw = cuNDArray<float>(sp_dcw);
                        cudcw = estimate_DCF(cutraj, cudcw, image_dims_);
                        auto hodcw = *cudcw.to_host();
                        dcw = std::make_shared<hoNDArray<float>>(hodcw);
                    }
                }

                //dcw = estimate_DCF(traj, dcw);
            }
            if (holds_alternative<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>>(message))
            {
                auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<cuNDArray, float_complext, 3>>(message);
                out.push(SpiralBuffer<cuNDArray, float_complext, 3>{std::move(sp_data),
                                                                    std::move(sp_traj),
                                                                    std::move(cuNDArray<float>(*dcw)),
                                                                    std::move(sp_headers)});
            }
            else if (holds_alternative<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message))
            {
                auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message);
                out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(sp_data),
                                                                    std::move(sp_traj),
                                                                    std::move(*dcw),
                                                                    std::move(sp_headers)});
            }
        }
    }

protected:
    NODE_PROPERTY(useIterativeDCWEstimated, bool, "Iterative DCW with Estimates", true);
    NODE_PROPERTY(iterations, size_t, "Number of Iterations", 15);
    NODE_PROPERTY(overSampling, float, "oversampling factor for DCF", 1.5);
    NODE_PROPERTY(kernelWidth, float, "oversampling factor for DCF", 5.5);
    NODE_PROPERTY(Debug, double, "Debug", 0);
    NODE_PROPERTY(slicewiseDCF, bool, "slicewiseDCF", true);
    NODE_PROPERTY(slicewiseDCF_FS, bool, "slicewiseDCFforFullysampled", true);
    NODE_PROPERTY(maxDevices, size_t, "Max Number of GPUs", 4);

    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(WeightsEstimationGadget)