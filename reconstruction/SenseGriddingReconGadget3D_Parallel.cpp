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
#include <gadgetron/cuSDC.h>
#include "../utils/concomitantFieldCorrection/mri_concomitant_field_correction.h"
#include "../utils/gpu/cuda_utils.h"
#include "../utils/util_functions.h"

using namespace Gadgetron;
using namespace Gadgetron::Core;
using namespace Gadgetron::Indexing;

class SenseGriddingReconGadget3D_Parallel : public ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>,
                                            public ImageArraySendMixin<SenseGriddingReconGadget3D_Parallel>
{
public:
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
    float oversampling_factor_;
    float kernel_width_;
    bool verbose;
    bool csm_calculated_ = false;
    cuCgSolver<float_complext> cg_;
    size_t numElements;
    float dcf_scale;
    boost::shared_ptr<cuNDArray<float_complext>> csm_;
    boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>> E_;
    boost::shared_ptr<cuCgPreconditioner<float_complext>> D_;
    boost::shared_ptr<cuImageOperator<float_complext>> R_;
    Gadgetron::ImageIOAnalyze gt_exporter_;
    boost::shared_ptr<cuNDArray<float>> _precon_weights;
    boost::shared_ptr<cuNDArray<float_complext>> precon_weights;

    SenseGriddingReconGadget3D_Parallel(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Core::variant<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>>(context, props)
    {
        kernel_width_ = 3;
        verbose = false;
    }

    hoNDArray<floatd3> traj2grad(const hoNDArray<floatd3> &trajectory, float kspace_scaling, float gamma)
    {
        auto gradients = trajectory;

        for (auto jj = 0; jj < trajectory.get_size(1); jj++)
        {
            auto traj_ptr = trajectory.get_data_ptr() + jj * trajectory.get_size(0);
            auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
            for (auto ii = 0; ii < trajectory.get_size(0); ii++)
            {
                if (ii > 0)
                {
                    grad_ptr[ii][0] = (traj_ptr[ii][0] - traj_ptr[ii - 1][0]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                    grad_ptr[ii][1] = (traj_ptr[ii][1] - traj_ptr[ii - 1][1]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                    grad_ptr[ii][2] = (traj_ptr[ii][2] - traj_ptr[ii - 1][2]) * 1000000 / (gamma * 10 * 2 * kspace_scaling);
                }
            }
        }
        return gradients;
    }
    std::tuple<std::vector<arma::vec>, std::vector<hoNDArray<float_complext>>> divide_bins(arma::vec in, hoNDArray<float_complext> &weights, int numGPUs)
    {
        std::vector<arma::vec> out;
        std::vector<hoNDArray<float_complext>> out_weights;
        auto sindex = 0;
        auto stride = floor(in.n_elem / numGPUs) - 1;
        for (auto ii = 0; ii < numGPUs; ii++)
        {
            if (ii == (numGPUs - 1))
            {
                out.push_back(arma::vec(in(arma::span(sindex, in.n_elem - 1))));
                std::vector<size_t> rdims = {weights.get_size(0), weights.get_size(1), weights.get_size(2), out[ii].n_elem};
                hoNDArray<float_complext> warray(rdims);
                for (auto jj = 0; jj < out[ii].n_elem; jj++)
                {
                    warray(slice, slice, slice, jj) = weights(slice, slice, slice, sindex + jj);
                }
                out_weights.push_back(warray);
            }
            else
            {
                out.push_back(arma::vec(in(arma::span(sindex, sindex + stride))));
                std::vector<size_t> rdims = {weights.get_size(0), weights.get_size(1), weights.get_size(2), out[ii].n_elem};
                hoNDArray<float_complext> warray(rdims);
                for (auto jj = 0; jj < out[ii].n_elem; jj++)
                {
                    warray(slice, slice, slice, jj) = weights(slice, slice, slice, sindex + jj);
                }
                out_weights.push_back(warray);

                sindex += stride + 1;
            }

            //auto nn = arma::vec(n);
        }
        return std::make_tuple(out, out_weights);
    }
    void demodulate_kspace(
        cuNDArray<float_complext> &demodulated_data,
        const cuNDArray<float> &scaled_time,
        float demodulation_freq)
    {
        //GadgetronTimer timer("Demodulation");
        constexpr float PI = boost::math::constants::pi<float>();

        auto recon_dim = demodulated_data.get_dimensions();
        recon_dim->pop_back();
        //hoNDArray<std::complex<float>> phase_term(recon_dim);

        auto val = float(-2.0 * PI * demodulation_freq);
        //  scaled_time *= val;

        //scaled_time *= val;

        auto arg_exp = std::move(*imag_to_complex<float_complext>(&scaled_time));
        arg_exp *= val;
        auto phase_term = lit_sgncr_toolbox::cuda_utils::cuexp<float_complext>(arg_exp);
        phase_term.squeeze();
        arg_exp.clear();

        std::vector<size_t> recon_dims({demodulated_data.get_size(0), demodulated_data.get_size(1)});
        cuNDArray<float_complext> temp_data(recon_dims);

        for (auto ich = 0; ich < demodulated_data.get_size(2); ich++)
        {
            cudaMemcpy(temp_data.get_data_ptr(),
                       demodulated_data.get_data_ptr() + demodulated_data.get_size(0) * demodulated_data.get_size(1) * ich,
                       demodulated_data.get_size(0) * demodulated_data.get_size(1) * sizeof(float_complext), cudaMemcpyDefault);

            temp_data *= phase_term;

            cudaMemcpy(demodulated_data.get_data_ptr() + demodulated_data.get_size(0) * demodulated_data.get_size(1) * ich,
                       temp_data.get_data_ptr(),
                       demodulated_data.get_size(0) * demodulated_data.get_size(1) * sizeof(float_complext), cudaMemcpyDefault);
        }

    }
    std::vector<arma::uvec> extractSlices(hoNDArray<floatd3> sp_traj)
    {
        auto sl_traj = arma::vec(sp_traj.get_size(1));
        for (auto ii = 0; ii < sp_traj.get_size(1); ii++)
            sl_traj[ii] = sp_traj(1, ii)[2];

        auto z_encodes = arma::vec(arma::unique(sl_traj));

        std::vector<arma::uvec> slice_indexes;
        for (auto ii = 0; ii < z_encodes.n_elem; ii++)
        {
            arma::uvec temp = (find(sl_traj == z_encodes[ii]));
            slice_indexes.push_back(temp);
        }

        return slice_indexes;
    }
    boost::shared_ptr<cuNDArray<float_complext>> reconstruct(
        cuNDArray<float_complext> *data,
        cuNDArray<floatd3> *traj,
        cuNDArray<float> *dcw,
        std::vector<size_t> recon_dims)
    {
        std::string printStatement;

        GadgetronTimer timer("Reconstruct");
        auto RO = data->get_size(0);
        auto E1E2 = data->get_size(1);
        auto CHA = data->get_size(2);

        auto nfft_plan_ = NFFT<cuNDArray, float, 3>::make_plan(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, kernel_width_, ConvolutionType::ATOMIC);
        nfft_plan_->preprocess(traj, NFFT_prep_mode::NC2C);

        auto result = boost::make_shared<cuNDArray<float_complext>>(recon_dims);
        recon_dims.pop_back();

        auto temp = boost::make_shared<cuNDArray<float_complext>>(recon_dims);

        cuNDArray<float_complext> tempdata(std::vector<size_t>{RO, E1E2});

        for (int iCHA = 0; iCHA < CHA; iCHA++)
        {
            cudaMemcpy(tempdata.get_data_ptr(),
                       data->get_data_ptr() + RO * E1E2 * iCHA,
                       RO * E1E2 * sizeof(float_complext), cudaMemcpyDefault);
            nfft_plan_->compute(tempdata, *temp.get(), dcw, NFFT_comp_mode::BACKWARDS_NC2C);

            cudaMemcpy(result->get_data_ptr() + recon_dims[0] * recon_dims[1] * recon_dims[2] * iCHA,
                       temp->get_data_ptr(),
                       recon_dims[0] * recon_dims[1] * recon_dims[2] * sizeof(float_complext), cudaMemcpyDefault);
        }

        return result;
    }

    cuNDArray<float_complext> concomitant_reconstruction(hoNDArray<float_complext> &data, hoNDArray<floatd3> &trajectory_in, hoNDArray<float> &dcf_in, hoNDArray<float_complext> &cweights_ho, hoNDArray<float> &scaled_time_in, arma::vec fbins)
    {

        boost::shared_ptr<hoNDArray<float_complext>> zpadded_cw;
        boost::shared_ptr<cuNDArray<float>> scaled_time;
        data.squeeze();
        auto CHA = data.get_size(size_t(data.get_number_of_dimensions() - 1));

        auto traj = boost::make_shared<cuNDArray<floatd3>>(trajectory_in);
        std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
        cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
        traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

        auto cuData = boost::make_shared<cuNDArray<float_complext>>(hoNDArray<float_complext>(data));
        auto dcw = boost::make_shared<cuNDArray<float>>(dcf_in);
        dcf_scale = asum((dcw.get()));
        float scale_factor = float(prod(image_dims_os_)) / asum((dcw.get()));
        *dcw *= scale_factor;

        std::vector<size_t> recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA}; // Cropped to size of Recon Matrix
        cuNDArray<float_complext> channel_images(recon_dims);
        fill(&channel_images, float_complext(0, 0));
        recon_dims.pop_back();

        if (fbins.n_elem > 1)
        {
            if (Debug)
                GDEBUG_STREAM("Loading Scaled Time");

            scaled_time = boost::make_shared<cuNDArray<float>>(scaled_time_in);

            if (Debug)
                GDEBUG_STREAM("Padding Weights");

            auto padded_cw = pad<float_complext, 4>(uint64d4(image_dims_[0], image_dims_[1], image_dims_[2], fbins.n_elem), cweights_ho, float_complext(0));

            zpadded_cw = boost::make_shared<hoNDArray<float_complext>>(padded_cw);
        }
        for (int ii = 0; ii < fbins.n_elem; ii++)
        {

            //GadgetronTimer timer("MFI");

            if (Debug)
            {
                GDEBUG_STREAM("Iteration# " << ii);
                GDEBUG_STREAM("Frequency: " << fbins[ii]);
            }
            if (fbins.n_elem > 1)
            {
                if (ii == 0)
                    demodulate_kspace(*cuData.get(), *scaled_time, fbins[ii]);
                else
                    demodulate_kspace(*cuData.get(), *scaled_time, fbins[ii] - fbins[ii - 1]);
            }
            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};

            auto channel_images_temp = *reconstruct(&(*cuData.get()), &(*traj.get()), &(*dcw.get()), recon_dims);

            if (Debug)
                GDEBUG_STREAM("Reconstructed");
            if (fbins.n_elem > 1)
            {
                if (Debug)
                    GDEBUG_STREAM("Reconstructed");
                auto hotempCW = (*zpadded_cw)(slice, slice, slice, ii);
              
                auto temp_cw = cuNDArray<float_complext>(hotempCW);
                channel_images_temp *= *conj(&temp_cw);
            }
            channel_images += channel_images_temp;
        }
        return channel_images;
    }

    void process(InputChannel<Core::variant<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>> &in,
                 OutputChannel &out) override
    {
        int selectedDevice = lit_sgncr_toolbox::utils::selectCudaDevice();
        cudaSetDevice(selectedDevice);

        auto matrixSize = this->header.encoding.front().reconSpace.matrixSize;
        auto fov = this->header.encoding.front().encodedSpace.fieldOfView_mm;
        unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
        size_t RO, E1, E2, CHA, N, S, SLC;

        boost::shared_ptr<cuNDArray<float_complext>> cuData;
        boost::shared_ptr<hoNDArray<float_complext>> hoData;
        boost::shared_ptr<cuNDArray<float>> dcw;
        boost::shared_ptr<cuNDArray<floatd3>> traj;
        boost::shared_ptr<cuNDArray<float_complext>> csm;

        IsmrmrdImageArray imarray;
        std::vector<size_t> recon_dims;

        hoNDArray<floatd3> gradients;

        auto kspace_scaling = 1e-3 * fov.x / matrixSize.x;
        constexpr double GAMMA = 4258.0; /* Hz/G */
        oversampling_factor_ = 1.5;

        image_dims_.push_back(matrixSize.x);
        image_dims_.push_back(matrixSize.y);
        if (this->header.encoding.front().encodedSpace.matrixSize.z % warp_size != 0)
            image_dims_.push_back(warp_size * (this->header.encoding.front().encodedSpace.matrixSize.z / warp_size + 1));
        else
            image_dims_.push_back(this->header.encoding.front().encodedSpace.matrixSize.z);

        lit_sgncr_toolbox::corrections::mri_concomitant_field_correction field_correction(this->header);

        // make the z dimension be multiple of 32 not sure of this discuss with David it was causing issues with cuda
        //Figure out what the oversampled matrix size should be taking the warp size into consideration.

        image_dims_os_ = uint64d3(((static_cast<size_t>(std::ceil(image_dims_[0] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                                  ((static_cast<size_t>(std::ceil(image_dims_[1] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size,
                                  ((static_cast<size_t>(std::ceil(image_dims_[2] * oversampling_factor_)) + warp_size - 1) / warp_size) * warp_size); // No oversampling is needed in the z-direction for SOS

        auto maxNumElements = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                               (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) * (header.encoding[0].encodingLimits.average.get().maximum + 1) * (header.encoding[0].encodingLimits.repetition.get().maximum + 1)); // use -1 for data acquired b/w 12/23 - 01/21
        // this->cn.plan();
        for (auto message : in)
        {
            GadgetronTimer timer("Sense Gridding Recon");

            auto &[sp_data, sp_traj, sp_dcw, sp_headers] = Core::get<Gadgetron::SpiralBuffer<hoNDArray, float_complext, 3>>(message);

            auto slice_index = extractSlices(sp_traj);
            bool estimate_oneslice = true;

            for (auto ii = 1; ii < slice_index.size(); ii++)
            {
                if (slice_index[ii - 1].n_elem != slice_index[ii].n_elem)
                    estimate_oneslice = false;
            }

            gradients = traj2grad(sp_traj, kspace_scaling, GAMMA);

            RO = (sp_data).get_size(0);
            E1 = (sp_data).get_size(1);
            E2 = this->header.encoding.front().encodedSpace.matrixSize.z;
            CHA = (sp_data).get_size(2);
            N = (sp_data).get_size(3);
            S = 1;
            SLC = 1;
            numElements = sp_data.get_number_of_elements();
            numElements /= CHA;
            if (Debug)
            {
                GDEBUG_STREAM("RO:" << RO);
                GDEBUG_STREAM("E1:" << E1);
                GDEBUG_STREAM("E2:" << E2);
                GDEBUG_STREAM("CHA:" << CHA);
                GDEBUG_STREAM("N:" << N);
            }
            float scaleMultiplier = float(sp_headers[0].user_float[7]);
            if (scaleMultiplier < 0.001)
                scaleMultiplier = 1.0;

            //auto scale_term = prod(image_dims_os_) / oversampling_factor_ * sqrt(float(maxNumElements * RO)) * 1e-6;
            //auto scale_term =  std::sqrt(float(maxNumElements * RO)) / prod(image_dims_os_) ;
            this->initialize_encoding_space_limits(this->header);
            //float scale_term = (float)sp_headers[0].user_float[7];
            //scale_term = 1.0;
            using namespace Gadgetron::Indexing;

            recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};

            // Seting up solvers
            E_ = boost::shared_ptr<cuNonCartesianSenseOperator<float, 3>>(new cuNonCartesianSenseOperator<float, 3>(ConvolutionType::ATOMIC));
            D_ = boost::shared_ptr<cuCgPreconditioner<float_complext>>(new cuCgPreconditioner<float_complext>());
            R_ = boost::shared_ptr<cuImageOperator<float_complext>>(new cuImageOperator<float_complext>());
            boost::shared_ptr<cuTvOperator<float_complext, 3>> TV(new cuTvOperator<float_complext, 3>);

            cuNDArray<float_complext> channel_images;
            if (doConcomitantFieldCorrection && (csm_calculated_)) // Does not do concomitant Field correction of CSM
            {
                field_correction.setup(gradients, sp_headers[0]);

                auto cwall = hoNDArray<float_complext>(field_correction.combinationWeights);

                auto size_data = sp_data.get_number_of_elements() * 2 * 4 + sp_traj.get_number_of_elements() * 3 * 4 +
                                 sp_dcw.get_number_of_elements() * 4 + field_correction.scaled_time.get_number_of_elements() * 4;

                auto eligibleGPUs = lit_sgncr_toolbox::utils::FindCudaDevices(size_data);
                if (Debug)
                    GDEBUG_STREAM("eligibleGPUs# " << eligibleGPUs.size());

                while (eligibleGPUs.size() > maxDevices || eligibleGPUs.size() > field_correction.demodulation_freqs.n_elem)
                    eligibleGPUs.pop_back();
                if (eligibleGPUs.size() < 1)
                    throw std::runtime_error("Need more space on the GPUs");
                auto [freq_bins, cweights] = divide_bins(field_correction.demodulation_freqs, cwall, eligibleGPUs.size());

                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA, cweights.size()};
                hoNDArray<float_complext> hochannel_images(recon_dims);
                if (Debug)
                    GDEBUG_STREAM("Going into the parallel Loop");

#pragma omp parallel for num_threads(eligibleGPUs.size())
                for (auto iter = 0; iter < eligibleGPUs.size(); iter++)
                {
                    cudaSetDevice(eligibleGPUs[iter]);
                    auto ci = concomitant_reconstruction(sp_data, sp_traj, sp_dcw, cweights[iter], field_correction.scaled_time, freq_bins[iter]);
                    hochannel_images(slice, slice, slice, slice, iter) = *ci.to_host();
                }
                cudaSetDevice(selectedDevice);

                channel_images = cuNDArray<float_complext>(*sum(&hochannel_images, hochannel_images.get_number_of_dimensions() - 1));
            }
            else
            {
                arma::vec freq_bins(1, arma::fill::zeros);
                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], 1};

                hoNDArray<float_complext> cweights(recon_dims);
                channel_images = concomitant_reconstruction(sp_data, sp_traj, sp_dcw, cweights, field_correction.scaled_time, freq_bins);
            }

            recon_dims = {image_dims_[0], image_dims_[1], header.encoding.front().encodedSpace.matrixSize.z, CHA}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> channel_images_cropped(recon_dims);
            crop<float_complext, 4>(uint64d4(0, 0, (image_dims_[2] - header.encoding.front().encodedSpace.matrixSize.z) / 2, 0),
                                    uint64d4(image_dims_[0], image_dims_[1], header.encoding.front().encodedSpace.matrixSize.z, CHA),
                                    channel_images,
                                    channel_images_cropped);

            //  recon_dims = {image_dims_[0], image_dims_[1], CHA, E2}; // Cropped to size of Recon Matrix
            channel_images = pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], image_dims_[2], CHA),
                                                    channel_images_cropped, float_complext(0));
            if (!csm_calculated_)
            {
                series_counter = 0;
                //                auto temp = boost::make_shared<cuNDArray<float_complext>>(estimate_b1_map<float, 3>(channel_images_cropped));
                auto temp = boost::make_shared<cuNDArray<float_complext>>(lit_sgncr_toolbox::utils::estimateCoilmaps_slice(channel_images_cropped));

                csm = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 4>(uint64d4(recon_dims[0], recon_dims[1], image_dims_[2], CHA),
                                                                                           *temp, float_complext(0)));
                csm_calculated_ = true;
                if (Debug)
                {
                    lit_sgncr_toolbox::utils::write_gpu_nd_array<float_complext>(channel_images, "/opt/data/gt_data/channel_images_full.complex");
                    lit_sgncr_toolbox::utils::write_gpu_nd_array<float_complext>(channel_images_cropped, "/opt/data/gt_data/channel_images_cropped.complex");
                }
                if (!SOS)
                {
                    if (Debug)
                        GDEBUG_STREAM("Not doing SOS");
                    channel_images *= *conj(csm.get());
                }
                else
                {
                    if (Debug)
                        GDEBUG_STREAM("SOS");
                    auto temp = channel_images;
                    channel_images *= *conj(&temp);
                }
                auto combined = sum(&channel_images, channel_images.get_number_of_dimensions() - 1);

                recon_dims = {image_dims_[0], image_dims_[1], CHA, header.encoding.front().reconSpace.matrixSize.z}; // Cropped to size of Recon Matrix
                cuNDArray<float_complext> images_cropped(recon_dims);
                crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - header.encoding.front().reconSpace.matrixSize.z) / 2),
                                        uint64d3(image_dims_[0], image_dims_[1], header.encoding.front().reconSpace.matrixSize.z),
                                        combined.get(),
                                        images_cropped);

                auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images_cropped.to_host())));

                lit_sgncr_toolbox::utils::filterImagealongSlice(images_all, get_kspace_filter_type(ftype), fwidth, fsigma);
                lit_sgncr_toolbox::utils::filterImage(images_all, get_kspace_filter_type(inftype), infwidth, infsigma);

                auto scale_term = scalewithinterleaves ? std::sqrt(float(maxNumElements * RO) / float(numElements)) * float(scaleMultiplier) : float(scaleMultiplier);

                if (Debug)
                {
                    GDEBUG_STREAM("maxNumElements:" << maxNumElements * RO);
                    GDEBUG_STREAM("numElements:" << numElements);
                    GDEBUG_STREAM("dcf_scale:" << dcf_scale);
                    GDEBUG_STREAM("image_dims_os_:" << prod(image_dims_os_));
                    GDEBUG_STREAM("scale_term:" << scale_term);
                }

                auto findMax = 0.0f;
                for (unsigned long int i = 0; i < images_all.get_number_of_elements(); i++)
                {
                    if (std::abs(images_all[i]) > findMax)
                        findMax = std::abs(images_all[i]);
                }
                if (Debug)
                {
                    GDEBUG("Max before Scaling: %f\n", findMax);
                }
                for (unsigned long int i = 0; i < images_all.get_number_of_elements(); i++)
                {
                    images_all[i] *= std::complex<float>(scale_term, 0.0);
                }
                imarray.data_ = images_all;
                if (Debug)
                    lit_sgncr_toolbox::utils::write_gpu_nd_array<float_complext>(*csm, "/opt/data/gt_data/csm_full.complex");

                lit_sgncr_toolbox::utils::attachHeadertoImageArray(imarray, sp_headers[0], this->header);

                prepare_image_array(imarray, (size_t)0, ((int)series_counter + 1), GADGETRON_IMAGE_REGULAR);

                out.push(imarray);
                imarray.meta_.clear();
                imarray.headers_.clear();
            }
            else
            {
                series_counter = 1;

                if (Debug)
                {
                    lit_sgncr_toolbox::utils::write_gpu_nd_array<float_complext>(*csm, "/opt/data/gt_data/csm.complex");
                    lit_sgncr_toolbox::utils::write_gpu_nd_array<float_complext>(channel_images, "/opt/data/gt_data/channel_images.complex");
                }

                cudaSetDevice(selectedDevice);
                auto traj = boost::make_shared<cuNDArray<floatd3>>(sp_traj);
                std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
                cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
                traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

                auto dcw = boost::make_shared<cuNDArray<float>>(sp_dcw);

                float scale_factor = float(prod(image_dims_os_)) / asum((dcw.get()));
                *dcw *= scale_factor;

                sqrt_inplace(dcw.get());

                R_->set_weight(kappa);

                E_->setup(from_std_vector<size_t, 3>(image_dims_), image_dims_os_, kernel_width_);
                E_->set_dcw(dcw);
                E_->set_csm(csm);

                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], field_correction.numfreqbins};
                cuNDArray<float_complext> image_array(recon_dims);

                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2]};
                cuNDArray<float_complext> reg_image(recon_dims);

                E_->mult_csm_conj_sum(&channel_images, &reg_image);
                channel_images.clear();

                auto codomain_dims = *sp_data.get_dimensions();

                recon_dims = {image_dims_[0], image_dims_[1], image_dims_[2], CHA};
                recon_dims.pop_back();

                E_->set_domain_dimensions(&recon_dims);
                E_->set_codomain_dimensions(&codomain_dims);
                E_->preprocess(traj.get());

                cg_.set_encoding_operator(E_);
                cg_.set_preconditioner(D_);
                cg_.add_regularization_operator(R_); // regularization matrix

                cg_.set_max_iterations(iterationsSense);
                cg_.set_tc_tolerance(tolSense);
                cg_.set_output_mode(decltype(cg_)::OUTPUT_VERBOSE);

                auto combined = boost::make_shared<Gadgetron::cuNDArray<Gadgetron::float_complext>>(reg_image);
                cg_.set_x0(combined);
                R_->compute(combined.get());
                boost::shared_ptr<cuNDArray<float>> R_diag = R_->get();
                *R_diag *= float(kappa);
                _precon_weights = sum(abs_square(csm.get()).get(), 3);
                *_precon_weights += *R_diag;
                R_diag.reset();

                recon_dims = {image_dims_[0], image_dims_[1], header.encoding.front().encodedSpace.matrixSize.z}; // Cropped to size of Recon Matrix
                cuNDArray<float> _precon_weights_cropped(recon_dims);

                crop<float, 3>(uint64d3(0, 0, (image_dims_[2] - header.encoding.front().encodedSpace.matrixSize.z) / 2),
                               uint64d3(image_dims_[0], image_dims_[1], header.encoding.front().encodedSpace.matrixSize.z),
                               *_precon_weights,
                               _precon_weights_cropped);

                reciprocal_sqrt_inplace(&_precon_weights_cropped);

                precon_weights = boost::make_shared<cuNDArray<float_complext>>(pad<float_complext, 3>(uint64d3(recon_dims[0], recon_dims[1], image_dims_[2]),
                                                                                                      *real_to_complex<float_complext>(&_precon_weights_cropped), float_complext(0)));

                D_->set_weights(precon_weights);

                try
                {
                    if (NoSense)
                        reg_image = *combined;

                    else
                        reg_image = *cg_.solve_from_rhs(&reg_image);

                }
                catch (const std::exception &e)
                {
                    reg_image = *combined;
                }
                recon_dims = {image_dims_[0], image_dims_[1], CHA, header.encoding.front().reconSpace.matrixSize.z}; // Cropped to size of Recon Matrix
                cuNDArray<float_complext> images_cropped(recon_dims);

                crop<float_complext, 3>(uint64d3(0, 0, (image_dims_[2] - header.encoding.front().reconSpace.matrixSize.z) / 2),
                                        uint64d3(image_dims_[0], image_dims_[1], header.encoding.front().reconSpace.matrixSize.z),
                                        reg_image,
                                        images_cropped);

                auto images_all = hoNDArray<std::complex<float>>(std::move(*boost::reinterpret_pointer_cast<hoNDArray<std::complex<float>>>(images_cropped.to_host())));

                lit_sgncr_toolbox::utils::filterImagealongSlice(images_all, get_kspace_filter_type(ftype), fwidth, fsigma);
                lit_sgncr_toolbox::utils::filterImage(images_all, get_kspace_filter_type(inftype), infwidth, infsigma);

                auto scale_term = scalewithinterleaves ? std::sqrt(float(maxNumElements * RO) / float(numElements)) * float(scaleMultiplier) : float(scaleMultiplier);

                if (Debug)
                {
                    GDEBUG_STREAM("maxNumElements:" << maxNumElements * RO);
                    GDEBUG_STREAM("numElements:" << numElements);
                    GDEBUG_STREAM("dcf_scale:" << dcf_scale);
                    GDEBUG_STREAM("image_dims_os_:" << prod(image_dims_os_));
                    GDEBUG_STREAM("scale_term:" << scale_term);
                }

                auto findMax = 0.0f;
                for (unsigned long int i = 0; i < images_all.get_number_of_elements(); i++)
                {
                    if (std::abs(images_all[i]) > findMax)
                        findMax = std::abs(images_all[i]);
                }
                if (Debug)
                    GDEBUG("Max Before Scaling: %f\n", findMax);

                for (unsigned long int i = 0; i < images_all.get_number_of_elements(); i++)
                {
                    images_all[i] *= std::complex<float>(1 / scale_term, 0.0);
                }
                imarray.data_ = images_all;

                lit_sgncr_toolbox::utils::attachHeadertoImageArray(imarray, sp_headers[0], this->header);

                prepare_image_array(imarray, (size_t)0, ((int)series_counter + 1), GADGETRON_IMAGE_BINNED);

                out.push(imarray);
                imarray.meta_.clear();
                imarray.headers_.clear();
            }
        }
    }

protected:
    NODE_PROPERTY(iterationsSense, size_t, "Number of Iterations Sense", 5);
    NODE_PROPERTY(tolSense, float, "Number of Iterations Sense", 1e-6);
    NODE_PROPERTY(kappa, double, "Kappa", 0.0);


    NODE_PROPERTY(fwidth, double, "filterWidth", 0.15);       // Filter width through slice
    NODE_PROPERTY(fsigma, double, "filterSigma", 1.0);        // Filter sigma through slice
    NODE_PROPERTY(ftype, std::string, "FilterType", "none");  // Filter type through slice
    NODE_PROPERTY(inftype, std::string, "inftype", "none");   // Filter type inplane
    NODE_PROPERTY(infwidth, double, "infwidth", 0.15);        // Filter width inplane
    NODE_PROPERTY(infsigma, double, "infsigma", 1.0);         // Filter sigma inplane

    NODE_PROPERTY(Debug, double, "Debug", 0);
    NODE_PROPERTY(NoSense, double, "NoSense", 1);
    NODE_PROPERTY(SOS, bool, "SOS", false);

    NODE_PROPERTY(doConcomitantFieldCorrection, bool, "doConcomitantFieldCorrection", true);
    NODE_PROPERTY(doConcomitantFieldCorrectionCSM, bool, "doConcomitantFieldCorrectionCSM", true);
   
    NODE_PROPERTY(maxDevices, size_t, "Max Number of GPUs", 4);
    
    NODE_PROPERTY(scalewithinterleaves, bool, "scalewinterleaves", true);

    int series_counter = 0;
};

GADGETRON_GADGET_EXPORT(SenseGriddingReconGadget3D_Parallel)
