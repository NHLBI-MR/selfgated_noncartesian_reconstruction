#include <gadgetron/Node.h>
#include "TrajectoryParameters.h"
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/vector_td_utilities.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <map>
#include <gadgetron/mri_core_data.h>
#include <boost/algorithm/string.hpp>
#include <gadgetron/hoNDArray_elemwise.h>
#include <gadgetron/hoNDArray_reductions.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <ismrmrd/xml.h>
#include <gadgetron/cuNFFT.h>
#include <gadgetron/cuNDFFT.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/cuNDArray.h>
#include "../utils/util_functions.h"
#include <gadgetron/GadgetronTimer.h>

#include "SpiralBuffer.h"
using namespace Gadgetron;
using namespace Gadgetron::Core;

class SpiralAccumulateFast : public ChannelGadget<Core::Acquisition>
{

public:
    SpiralAccumulateFast(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props), header{context.header}
    {
    }
    // Convert to Templates to deal with view arrays etc.
    std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D(
        hoNDArray<float> *traj_dcw, int iSL)
    {
        std::vector<size_t> dims = *traj_dcw->get_dimensions();
        std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); //Copy vector, but leave out first dim
        auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

        auto traj = boost::make_shared<hoNDArray<floatd3>>(reduced_dims);

        auto dcw_ptr = dcw->get_data_ptr();
        auto traj_ptr = traj->get_data_ptr();
        auto ptr = traj_dcw->get_data_ptr();

        //std::ofstream ofs("/tmp/traj_grad_flat.log");
        for (size_t i = 0; i < traj_dcw->get_number_of_elements() / 3; i++)
        {
            auto zencoding = float(-0.5 + iSL * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z));

            traj_ptr[i][0] = ptr[i * 3];
            traj_ptr[i][1] = ptr[i * 3 + 1];
            traj_ptr[i][2] = zencoding;
            dcw_ptr[i] = ptr[i * 3 + 2];
        }

        return std::make_tuple(traj, dcw);
    }
    // Convert to Templates to deal with view arrays etc.
    std::tuple<boost::shared_ptr<hoNDArray<floatd2>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_2(
        hoNDArray<float> *traj_dcw)
    {

        std::vector<size_t> dims = *traj_dcw->get_dimensions();
        std::vector<size_t> reduced_dims(dims.begin() + 1, dims.end()); //Copy vector, but leave out first dim
        auto dcw = boost::make_shared<hoNDArray<float>>(reduced_dims);

        auto traj = boost::make_shared<hoNDArray<floatd2>>(reduced_dims);

        auto dcw_ptr = dcw->get_data_ptr();
        auto traj_ptr = traj->get_data_ptr();
        auto ptr = traj_dcw->get_data_ptr();
        //std::ofstream ofs("/tmp/traj_grad_flat.log");
        for (size_t i = 0; i < traj_dcw->get_number_of_elements() / 3; i++)
        {
            traj_ptr[i][0] = ptr[i * 3];
            traj_ptr[i][1] = ptr[i * 3 + 1];
            dcw_ptr[i] = ptr[i * 3 + 2];
        }

        return std::make_tuple(traj, dcw);
    }
    // Convert to Templates to deal with view arrays etc.
    std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> separate_traj_and_dcw_3D_gen(
        hoNDArray<floatd2> *traj_input, hoNDArray<float> *dcw_input, int iSL)
    {

        std::vector<size_t> dims = *traj_input->get_dimensions();
        auto dcw = boost::make_shared<hoNDArray<float>>(dims);

        auto traj = boost::make_shared<hoNDArray<floatd3>>(dims);

        auto dcw_ptr = dcw->get_data_ptr();
        auto traj_ptr = traj->get_data_ptr();
        auto ptr = traj_input->get_data_ptr();
        auto dcwptr = dcw_input->get_data_ptr();
        //std::ofstream ofs("/tmp/traj_grad_flat.log");
        for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
        {
            auto zencoding = float(-0.5 + iSL * 1 / ((float)this->header.encoding.front().encodedSpace.matrixSize.z));

            traj_ptr[i][0] = ptr[i][0];
            traj_ptr[i][1] = ptr[i][1];
            traj_ptr[i][2] = zencoding;
            dcw_ptr[i] = dcwptr[i];
        }

        return std::make_tuple(traj, dcw);
    }
    void normalize_trajectory(
        hoNDArray<floatd2> *traj_input)
    {

        std::vector<size_t> dims = *traj_input->get_dimensions();
        auto ptr = traj_input->get_data_ptr();

        float max_x = 0;
        float min_x = 0;
        float max_y = 0;
        float min_y = 0;

        for (size_t i = 0; i < traj_input->get_number_of_elements(); i++)
        {
            if (max_x < ptr[i][0])
                max_x = ptr[i][0];

            if (min_x > ptr[i][0])
                min_x = ptr[i][0];

            if (max_y < ptr[i][1])
                max_y = ptr[i][1];

            if (min_y > ptr[i][1])
                min_y = ptr[i][1];
        }

        // auto normx = 2*std::max(std::abs(max_x),std::abs(min_x));
        // auto normy = 2*std::max(std::abs(max_y),std::abs(min_y));
        // for (size_t i = 0; i < traj_input->get_number_of_elements() / 3; i++)
        // {
        //     ptr[i][0]/=normx;
        //     ptr[i][1]/=normy;
        // }
    }

    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        size_t acq_count = 0;
        bool csmSent = false;

        using namespace Gadgetron::Indexing;
        auto [acq_header, data, traj] = in.pop();

        /* Initializations */
        //header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1}
        std::map<size_t, std::vector<hoNDArray<float_complext>>> acquisitions;
        std::map<size_t, std::vector<ISMRMRD::AcquisitionHeader>> headers;
        std::map<size_t, std::vector<hoNDArray<floatd3>>> trajectories;
        std::map<size_t, std::vector<hoNDArray<float>>> density_weights;

        //std::vector<std::vector<std::vector<hoNDArray<float_complext>>>> acquisitions(10);
        //std::vector<std::vector<std::vector<ISMRMRD::AcquisitionHeader>>> headers(10);
        //std::vector<std::vector<std::vector<hoNDArray<floatd3>>>> trajectories(10);
        //std::vector<std::vector<std::vector<hoNDArray<float>>>> density_weights(10);

        std::tuple<boost::shared_ptr<hoNDArray<floatd3>>, boost::shared_ptr<hoNDArray<float>>> traj_dcw;
        std::pair<hoNDArray<floatd2>, hoNDArray<float>> tw_gen;
        hoNDArray<floatd2> trajgen;
        hoNDArray<float> dcwgen;

        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        auto maxAcq = ((header.encoding.at(0).encodingLimits.kspace_encoding_step_1.get().maximum + 1) *
                       (header.encoding[0].encodingLimits.kspace_encoding_step_2.get().maximum + 1) * (header.encoding[0].encodingLimits.average.get().maximum + 1) * (header.encoding[0].encodingLimits.repetition.get().maximum + 1)); // use -1 for data acquired b/w 12/23 - 01/21

        auto RO = data.get_size(0);
        auto CHA = acq_header.active_channels;

        /* Take trajectories from waveform to trajectory */

        traj_dcw = separate_traj_and_dcw_3D(&(*traj), acq_header.idx.kspace_encode_step_2);

        acquisitions.insert(std::pair<size_t, std::vector<hoNDArray<float_complext>>>(acq_header.idx.phase,
                                                                                      std::vector<hoNDArray<float_complext>>(0)));

        trajectories.insert(std::pair<size_t, std::vector<hoNDArray<floatd3>>>(acq_header.idx.phase,
                                                                               std::vector<hoNDArray<floatd3>>(0)));

        density_weights.insert(std::pair<size_t, std::vector<hoNDArray<float>>>(acq_header.idx.phase,
                                                                                std::vector<hoNDArray<float>>(0)));

        headers.insert(std::pair<size_t, std::vector<ISMRMRD::AcquisitionHeader>>(acq_header.idx.phase,
                                                                                  std::vector<ISMRMRD::AcquisitionHeader>(0)));

        acquisitions.find(acq_header.idx.phase)->second.push_back(hoNDArray<float_complext>(data));
        trajectories.find(acq_header.idx.phase)->second.push_back(hoNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
        density_weights.find(acq_header.idx.phase)->second.push_back(hoNDArray<float>(*std::get<1>(traj_dcw).get()));
        headers.find(acq_header.idx.phase)->second.push_back(acq_header);
        acq_count++;
        /* Get this index to use for Reordering data before pushing it along */
        auto E2 = this->header.encoding.front().encodedSpace.matrixSize.z;

        /* Getting data, trajectories and headers and sorting them and pushing them along */
        for (auto [acq_header, data, traj] : in)
        {
            //acquisitions.emplace_back(cuNDArray<float_complext>(hoNDArray<float_complext>(data)));
            // acquisitions[acq_header.idx.kspace_encode_step_2].emplace_back((hoNDArray<float_complext>(data)));

            //headers[acq_header.idx.kspace_encode_step_2].emplace_back(acq_header);

            if (acq_header.idx.average + 1 > (avg))
                avg = acq_header.idx.average + 1;

            /* Generate VDS trajectories and make them 3D*/
            if (generateTraj)
            {
                auto traj = hoNDArray<floatd2>(trajgen(slice, acq_header.idx.kspace_encode_step_1));
                auto densitycw = hoNDArray<float>(dcwgen(slice, acq_header.idx.kspace_encode_step_1));
                traj_dcw = separate_traj_and_dcw_3D_gen(&traj, &densitycw,
                                                        acq_header.idx.kspace_encode_step_2);
            }
            /* Take trajectories from waveform to trajectory */
            else
                traj_dcw = separate_traj_and_dcw_3D(&(*traj), acq_header.idx.kspace_encode_step_2);

            // trajectories.push_back(cuNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
            // density_weights.push_back(cuNDArray<float>(*std::get<1>(traj_dcw).get()));
            if (acquisitions.find(acq_header.idx.phase) == acquisitions.end())
            {
                acquisitions.insert(std::pair<size_t, std::vector<hoNDArray<float_complext>>>(acq_header.idx.phase,
                                                                                              std::vector<hoNDArray<float_complext>>(0)));

                trajectories.insert(std::pair<size_t, std::vector<hoNDArray<floatd3>>>(acq_header.idx.phase,
                                                                                       std::vector<hoNDArray<floatd3>>(0)));

                density_weights.insert(std::pair<size_t, std::vector<hoNDArray<float>>>(acq_header.idx.phase,
                                                                                        std::vector<hoNDArray<float>>(0)));

                headers.insert(std::pair<size_t, std::vector<ISMRMRD::AcquisitionHeader>>(acq_header.idx.phase,
                                                                                          std::vector<ISMRMRD::AcquisitionHeader>(0)));

                acquisitions.find(acq_header.idx.phase)->second.push_back(hoNDArray<float_complext>(data));
                trajectories.find(acq_header.idx.phase)->second.push_back(hoNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
                density_weights.find(acq_header.idx.phase)->second.push_back(hoNDArray<float>(*std::get<1>(traj_dcw).get()));
                headers.find(acq_header.idx.phase)->second.push_back(acq_header);
            }
            else
            {
                acquisitions.find(acq_header.idx.phase)->second.push_back(hoNDArray<float_complext>(data));
                trajectories.find(acq_header.idx.phase)->second.push_back(hoNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
                density_weights.find(acq_header.idx.phase)->second.push_back(hoNDArray<float>(*std::get<1>(traj_dcw).get()));
                headers.find(acq_header.idx.phase)->second.push_back(acq_header);
            }
            acq_count++;
            //acquisitions[acq_header.idx.phase].at(acq_header.idx.kspace_encode_step_2).push_back((hoNDArray<float_complext>(data)));
            //trajectories[acq_header.idx.phase].at(acq_header.idx.kspace_encode_step_2).push_back(hoNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
            //density_weights[acq_header.idx.phase].at(acq_header.idx.kspace_encode_step_2).push_back(hoNDArray<float>(*std::get<1>(traj_dcw).get()));
            //headers[acq_header.idx.phase].at(acq_header.idx.kspace_encode_step_2).push_back(acq_header);
            //trajectories[acq_header.idx.kspace_encode_step_2].push_back(hoNDArray<floatd3>(*std::get<0>(traj_dcw).get()));
            //density_weights[acq_header.idx.kspace_encode_step_2].push_back(hoNDArray<float>(*std::get<1>(traj_dcw).get()));

            if (acq_count == (fixedINT * maxZencode))
            {
                std::vector<hoNDArray<float_complext>> acquisitions_concat;
                std::vector<hoNDArray<floatd3>> trajectories_concat_all;
                std::vector<hoNDArray<float>> dcf_concat_all;
                hoNDArray<float> combined_density;

                if ((acq_count == (fixedINT * maxZencode) || maxAcq < (fixedINT * maxZencode)) && !csmSent)
                {
                    GadgetronTimer timer("Spiral Acuumulate 3D Trigger:");

                    GDEBUG_STREAM("Acquisitions " << acq_count);

                    for (auto ii = 0; ii < acquisitions.size(); ii++)
                    {
                        if (!acquisitions[ii].empty())
                        {
                            acquisitions_concat.push_back(concat(acquisitions[ii]));
                            trajectories_concat_all.push_back(concat(trajectories[ii]));
                            dcf_concat_all.push_back(concat(density_weights[ii]));
                        }
                    }
                    auto combined_acquisitions = lit_sgncr_toolbox::utils::concat<float_complext>(acquisitions_concat);
                    combined_acquisitions.reshape(combined_acquisitions.get_size(0),
                                                  combined_acquisitions.get_size(1),
                                                  -1,
                                                  1);                                     //AVG =1
                    combined_acquisitions = permute(combined_acquisitions, {0, 2, 1, 3}); // RO INT CHA SLICE
                    cudaSetDevice(0);
                    // auto combined_density = lit_sgncr_toolbox::utils::concat<float>(density_weights);
                    // std::vector<std::vector<hoNDArray<floatd3>>> trajectories_concat(trajectories[0].size());
                    // std::vector<std::vector<hoNDArray<float>>> dcf_concat(density_weights[0].size());

                    auto combined_traj = lit_sgncr_toolbox::utils::concat<floatd3>(trajectories_concat_all);
                    combined_traj.reshape(combined_traj.get_size(0),
                                          -1,
                                          1);

                    //  combined_density = lit_sgncr_toolbox::utils::estimateDCF(combined_traj, lit_sgncr_toolbox::utils::concat<float>(dcf_concat_all), true, oversampling_factor_, iterations_dcf, image_dims_, false);
                    combined_density = lit_sgncr_toolbox::utils::concat<float>(dcf_concat_all);
                    //auto combined_density = lit_sgncr_toolbox::utils::concat<float>(dcw);
                    combined_density.reshape(combined_acquisitions.get_size(0),
                                             -1,
                                             1);
                    //   combined_traj = permute(combined_traj, {0, 1, 2}); // RO INT CHA SLICE
                    std::vector<ISMRMRD::AcquisitionHeader> headers_copy;

                    if (headers[0].empty())
                        headers_copy = headers[1];
                    else
                        headers_copy = headers[0];

                    out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(combined_acquisitions),
                                                                        std::move(combined_traj),
                                                                        std::move(combined_density),
                                                                        std::move(headers_copy)});

                    csmSent = true;
                }
            }
            // else
            // {
            //     GDEBUG_STREAM("Acquisitions " << acq_count);
            // }
        }

        if (!csmSent)
        {
            std::vector<hoNDArray<float_complext>> acquisitions_concat;
            std::vector<hoNDArray<floatd3>> trajectories_concat_all;
            std::vector<hoNDArray<float>> dcf_concat_all;
            hoNDArray<float> combined_density;

            if ((acq_header.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT) || acq_count == (fixedINT * maxZencode) || maxAcq < (fixedINT * maxZencode)) && !csmSent)
            {
                GadgetronTimer timer("Spiral Acuumulate 3D Trigger:");

                GDEBUG_STREAM("Acquisitions " << acq_count);

                for (auto ii = 0; ii < acquisitions.size(); ii++)
                {
                    if (!acquisitions[ii].empty())
                    {
                        acquisitions_concat.push_back(concat(acquisitions[ii]));
                        trajectories_concat_all.push_back(concat(trajectories[ii]));
                        dcf_concat_all.push_back(concat(density_weights[ii]));
                    }
                }
                auto combined_acquisitions = lit_sgncr_toolbox::utils::concat<float_complext>(acquisitions_concat);
                combined_acquisitions.reshape(combined_acquisitions.get_size(0),
                                              combined_acquisitions.get_size(1),
                                              -1,
                                              1);                                     //AVG =1
                combined_acquisitions = permute(combined_acquisitions, {0, 2, 1, 3}); // RO INT CHA SLICE
                cudaSetDevice(0);

                auto combined_traj = lit_sgncr_toolbox::utils::concat<floatd3>(trajectories_concat_all);
                combined_traj.reshape(combined_traj.get_size(0),
                                      -1,
                                      1);

                combined_density = lit_sgncr_toolbox::utils::concat<float>(dcf_concat_all);
                combined_density.reshape(combined_acquisitions.get_size(0),
                                         -1,
                                         1);

                std::vector<ISMRMRD::AcquisitionHeader> headers_copy;

                if (headers[0].empty())
                    headers_copy = headers[1];
                else
                    headers_copy = headers[0];

                out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(combined_acquisitions),
                                                                    std::move(combined_traj),
                                                                    std::move(combined_density),
                                                                    std::move(headers_copy)});

                csmSent = true;
            }
        }

        if (csmSent)
        {
            std::vector<hoNDArray<float_complext>> acquisitions_concat;
            std::vector<hoNDArray<floatd3>> trajectories_concat_all;
            std::vector<hoNDArray<float>> dcf_concat_all;
            hoNDArray<float> combined_density;

            for (auto ii = ((acquisitions.size() > 2) ? reconstruct_phase_start : 1); ii <= ((acquisitions.size() < reconstruct_phase_end) ? reconstruct_phase_end : (acquisitions.size() - 1)); ii++)
            {
                acquisitions_concat.clear();
                trajectories_concat_all.clear();
                dcf_concat_all.clear();
                combined_density.clear();

                GDEBUG_STREAM("Acquisitions Binned:" << acquisitions[ii].size());

                auto combined_acquisitions = concat(acquisitions[ii]);
                combined_acquisitions.reshape(combined_acquisitions.get_size(0),
                                              combined_acquisitions.get_size(1),
                                              -1,
                                              1);                                     //AVG =1
                combined_acquisitions = permute(combined_acquisitions, {0, 2, 1, 3}); // RO INT CHA SLICE

                auto combined_traj = concat(trajectories[ii]);
                combined_traj.reshape(combined_traj.get_size(0),
                                      -1,
                                      1);

                combined_density = concat(density_weights[ii]);
                combined_density.reshape(combined_acquisitions.get_size(0),
                                         -1,
                                         1);

                auto headers_copy = headers[ii];
                out.push(SpiralBuffer<hoNDArray, float_complext, 3>{std::move(combined_acquisitions),
                                                                    std::move(combined_traj),
                                                                    std::move(combined_density),
                                                                    std::move(headers_copy)});
            }

            headers.clear();
            acquisitions.clear();
            trajectories.clear();
            density_weights.clear();
            if (CSMReset)
                csmSent = false;
        }
    }

protected:
    ISMRMRD::IsmrmrdHeader header;

    NODE_PROPERTY(generateTraj, bool, "generate trajectories", false);
    NODE_PROPERTY(perform_GIRF, bool, " Perform GIRF", true);
    NODE_PROPERTY(GIRF_folder, std::string, "Path where GIRF Data is stored", "/opt/data/GIRF/");
    NODE_PROPERTY(GIRF_samplingtime, float, "girf sampling time", 10e-6);
    NODE_PROPERTY(do_slwindow, bool, "Do Sliding window", false);
    NODE_PROPERTY(slwindow_len, size_t, "Length of the SLwindow", 23);
    NODE_PROPERTY(slwindow_str, size_t, "Length of the SLwindow", 23);
    NODE_PROPERTY(reconstruct_phase_start, size_t, "Phase to reconstruction", 0); // zero reconstructs all
    NODE_PROPERTY(reconstruct_phase_end, size_t, "Phase to reconstruction", 1);   // zero reconstructs all
    NODE_PROPERTY(iterations_dcf, size_t, "iterations for DCF estimation", 20);   // zero reconstructs all
    NODE_PROPERTY(fixedINT, size_t, "fixedINT", 600);
    NODE_PROPERTY(CSMReset, bool, "CSMReset", false); //
private:
    size_t avg = 0;
    size_t phase = 0;
    std::vector<size_t> image_dims_;
    uint64d3 image_dims_os_;
};

GADGETRON_GADGET_EXPORT(SpiralAccumulateFast)