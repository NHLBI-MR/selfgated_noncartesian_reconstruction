//
// Created by ajaved on 08/17/2020
//

#include <gadgetron/Node.h>
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
#include <gadgetron/hoNDArray_fileio.h>
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/GadgetronTimer.h>
#include <ismrmrd/xml.h>
#include "../utils/util_functions.h"
#include "../utils/window_filter.h"
#include "hoArmadillo.h"
#include <gadgetron/GadgetronTimer.h>
#include <gadgetron/mri_core_def.h>
#include <gadgetron/hoNDImage_util.h>
#include <numeric>   // std::iota
#include <algorithm> // std::sort, std::stable_sort

#define PI 3.14159265358                       //Go as long as you want, the longer the more accurate
#define radianstodegrees(R) ((180.0 * R) / PI) //Converts Radians to Degrees
#define degreestoradians(D) ((PI * D) / 180.0) //Converts Radians to Degrees

using namespace Gadgetron;
using namespace Gadgetron::Core;

class GatingandBinningGadget : public ChannelGadget<Core::Acquisition>
{

public:
    GatingandBinningGadget(const Core::Context &context, const Core::GadgetProperties &props) : ChannelGadget<Acquisition>(context, props), header{context.header}
    {
    }

    std::vector<std::tuple<float, float>> stableBinning(hoNDArray<float> data, float acceptancePercent)
    {
        auto inputData = data;

        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(inputData, "/opt/data/gt_data/stableBin_data.real");

        std::sort(inputData.begin(), inputData.end());

        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(inputData, "/opt/data/gt_data/stableBin_data_sorted.real");
        //mmstruct m;
        int slopeLen = acceptancePercent / 100 * inputData.get_size(0);

        hoNDArray<float> slope(inputData.get_size(0) - slopeLen);
        for (auto i = 0; i < inputData.get_size(0) - slopeLen; i++)
        {
            slope[i] = (inputData[i + slopeLen] - inputData[i]) / slopeLen;
        }

        auto SIndex = std::distance(slope.begin(), std::min_element(slope.begin(), slope.end()));

        // m.min = inputData[SIndex];
        // m.max = inputData[SIndex+slopeLen];
        std::vector<std::tuple<float, float>> output;
        output.push_back(std::make_tuple(inputData[SIndex], inputData[SIndex + slopeLen + 1]));
        return output; // Min,Max
    }

    std::vector<std::tuple<float, float>> respiratoryPhases(hoNDArray<float> data)
    {
        auto inputData = data;
        std::vector<std::tuple<float, float>> output;
        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(inputData, "/opt/data/gt_data/stableBin_data.real");

        std::sort(inputData.begin(), inputData.end());

        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(inputData, "/opt/data/gt_data/stableBin_data_sorted.real");

        auto ulimit = Gadgetron::percentile(data, float(up_perc));
        auto llimit = Gadgetron::percentile(data, float(low_perc));

        auto delta = (ulimit - llimit) / (numberOfBins - 1);

        for (auto ii = 0; ii < numberOfBins; ii++)
            output.push_back(std::make_tuple(llimit + ii * delta, llimit + (ii + 1) * delta));

        return output;
    }

    std::vector<float> findNavAngle(std::vector<uint32_t> navTimestamps, std::vector<uint32_t> acqTimestamps, std::vector<float> traj_angles)
    {
        GadgetronTimer timer("Find Nav angles:");

        std::vector<float> navAngles;
        for (auto jj = 0; jj < navTimestamps.size(); jj++)
        {
            auto tstamp = navTimestamps[jj];
            for (auto ii = 0; ii < acqTimestamps.size(); ii++)
            {
                if (ii > 0 && (int(tstamp) - int(acqTimestamps[ii])) < 0 && (int(tstamp) - int(acqTimestamps[ii - 1])) > 0)
                {
                    navAngles.push_back(traj_angles[ii - 1]);
                    continue;
                }
                else if ((int(tstamp) - int(acqTimestamps[ii])) < 0 && ii == 0 && jj == 0)
                {
                    navAngles.push_back(traj_angles[ii]);
                    continue;
                }
            }
        }
        return navAngles;
    }

    std::vector<size_t> sort_indexes(std::vector<float> &v)
    {

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::stable_sort(idx.begin(), idx.end(),
                         [&v](size_t i1, size_t i2)
                         { return v[i1] < v[i2]; });

        return idx;
    }
    // Angluar Filteration
    hoNDArray<std::complex<float>> correct_trajectory_fluctuations(hoNDArray<std::complex<float>> signal, std::vector<float> navAngles)
    {
        using namespace Gadgetron::Indexing;

        auto findUnNA = navAngles;
        sort(findUnNA.begin(), findUnNA.end());
        findUnNA.erase(unique(findUnNA.begin(), findUnNA.end()), findUnNA.end());

        auto temp = navAngles;
        auto idx = sort_indexes(temp);

        float interleaves = findUnNA.size();
        int factor = (int(std::ceil(signal.get_size(0) / interleaves)) % int(interleaves)) - std::round(signal.get_size(0) / interleaves);

        float nav_samplingTime = 0;

        for (int ii = 3; ii < navAngles.size(); ii = ii + 3)
            nav_samplingTime += navAngles[idx[ii]] - navAngles[idx[ii - 3]];

        nav_samplingTime /= interleaves;

        // Generate the Highpass filter
        arma::vec filterBandsHP(1);
        filterBandsHP[0] = 0.04;

        arma::vec errorsHP(2);
        errorsHP[0] = 0.01;
        errorsHP[1] = 0.01;

        lit_sgncr_toolbox::kaiserFilter kfilterHP(filterBandsHP, errorsHP, 1 / nav_samplingTime, lit_sgncr_toolbox::kaiserFilter::filterType::highPass);
        kfilterHP.generateFilter();

        auto sort_signal = signal;

        //Sort the signal wrt to angles
        for (auto ii = 0; ii < signal.get_size(0); ii++)
        {
            sort_signal(ii, slice) = signal(idx[ii], slice);
        }

        // organize data for HP filteration
        sort_signal.reshape(size_t(std::round(signal.get_size(0) / interleaves)), size_t(interleaves + factor), signal.get_size(1));
        sort_signal = permute(sort_signal, {1, 0, 2});
        sort_signal.reshape(-1, size_t(std::round(signal.get_size(0) / interleaves)) * signal.get_size(1));
        {
            GadgetronTimer timer("HP filter:");

            // HP filter wrt angular rotations
            kfilterHP.filterData(sort_signal);
        }
        sort_signal.reshape(-1, size_t(std::round(signal.get_size(0) / interleaves)), signal.get_size(1));
        sort_signal = permute(sort_signal, {1, 0, 2});
        sort_signal.reshape(size_t(std::round(signal.get_size(0) / interleaves)) * size_t(interleaves + factor), signal.get_size(1));

        //Reverse the sort wrt to angles
        for (auto ii = 0; ii < signal.get_size(0); ii++)
        {
            signal(idx[ii], slice) = sort_signal(ii, slice);
        }

        lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(signal, "/opt/data/gt_data/signalHP.complex");

        // End Angluar Filteration

        return signal;
    }
    std::tuple<float, std::vector<std::vector<float>>> estimateGatingSignal(std::vector<Core::Acquisition> navacquisitions, float acceptancePercent, std::vector<float> navAngles)
    {
        GadgetronTimer timer("estimateGatingSignal:");

        std::vector<hoNDArray<std::complex<float>>> navdata;
        std::vector<ISMRMRD::AcquisitionHeader> headers;
        float samplingTime = 0;

        //temp.erase(temp.begin());

        // organize the data
        for (auto &[acq_header, data, traj] : navacquisitions)
        {
            navdata.emplace_back(std::move(data));
            headers.emplace_back(acq_header);
        }

        auto concatNavdata = concat(navdata);
        concatNavdata.reshape(concatNavdata.get_size(0), concatNavdata.get_size(1), -1);
        auto numCH = concatNavdata.get_size(1);
        if (!useDC)
            Gadgetron::hoNDFFT<float>::instance()->fft(&concatNavdata, 0);

        auto centerNavData = hoNDArray<std::complex<float>>(concatNavdata);
        auto temp = *Gadgetron::abs(&centerNavData);
        centerNavData = hoNDArray<std::complex<float>>(temp);

        centerNavData.reshape(concatNavdata.get_size(0) * concatNavdata.get_size(1), -1); // Np*Nc x Nt
        centerNavData = permute(centerNavData, {1, 0});                                   // Nt x Np*Nc Filteration happens along the first dimension

        auto filename = "/opt/data/gt_data/centerNavData_bfsvd.complex";
        lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(centerNavData, filename);

        if (doAngularFilteration)
        {
            centerNavData = correct_trajectory_fluctuations(centerNavData, navAngles);
        }
        // Generate the Bandpass filter
        for (int ii = 1; ii < headers.size() - 1; ii++)
            samplingTime += float(headers[ii].physiology_time_stamp[0] - headers[ii - 1].physiology_time_stamp[0]) * 2.5;

        samplingTime /= headers.size();

        arma::vec filterBands(4);
        filterBands[0] = 0.08;
        filterBands[1] = 0.10;
        filterBands[2] = 0.70;
        filterBands[3] = 0.90;

        arma::vec errors(3);
        errors[0] = 0.01;
        errors[1] = 0.01;
        errors[2] = 0.01;

        lit_sgncr_toolbox::kaiserFilter kfilter(filterBands, errors, (1 / (samplingTime * 1e-3)), lit_sgncr_toolbox::kaiserFilter::filterType::bandPass);
        kfilter.generateFilter();
        {
            GadgetronTimer timer("BP filter:");

            kfilter.filterData(centerNavData);
        }
        centerNavData.reshape(-1, concatNavdata.get_size(0), concatNavdata.get_size(1));

        hoNDArray<complex_float_t> temp1(centerNavData.get_size(0), centerNavData.get_size(2));

        //filename = "/opt/data/gt_data/centerNavData_bfsvd.complex";
        lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(centerNavData, filename);

        using namespace Gadgetron::Indexing;
        for (auto ii = 0; ii < centerNavData.get_size(2); ii++)
        {

            hoNDArray<complex_float_t> temp2 = centerNavData(slice, slice, ii);
            //        GDEBUG_STREAM("cnd_temp(0):" << cnd_temp.get_size(0));
            //         GDEBUG_STREAM("cnd_temp(1):" << cnd_temp.get_size(1));
            hoNDArray<float> V_({temp2.get_size(1), temp2.get_size(1)});
            hoNDArray<float> U_({temp2.get_size(0), temp2.get_size(0)});
            arma::fmat Vm = as_arma_matrix(V_);
            arma::fmat Um = as_arma_matrix(U_);
            arma::fvec Sv;
            auto A_ = real(temp2);
            auto A = as_arma_matrix(A_);

            arma::svd(Um, Sv, Vm, A);

            //hoNDArray<std::complex<float>> signal({Um.n_rows, 1});

            for (auto jj = 0; jj < Um.n_rows; jj++)
            {
                temp1(jj, ii) = Um(jj);
            }
        }

        centerNavData = temp1;
        //filename = "/opt/data/gt_data/centerNavData_afsvd.complex";
        lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(centerNavData, filename);

        // Coil clustering
        auto threshold = 0.98;
        hoNDArray<float> C(centerNavData.get_size(1), centerNavData.get_size(1));
        hoNDArray<float> G(centerNavData.get_size(1), centerNavData.get_size(1));

        for (auto ii = 0; ii < centerNavData.get_size(1); ii++)
        {
            for (auto jj = 0; jj < centerNavData.get_size(1); jj++)
            {

                auto r = lit_sgncr_toolbox::utils::correlation(Gadgetron::real(hoNDArray<complex_float_t>(centerNavData(slice, ii))), Gadgetron::real(hoNDArray<complex_float_t>(centerNavData(slice, jj))));
                C(ii, jj) = r;
                G(ii, jj) = (r > threshold) ? 1 : 0;
            }
        }

        hoNDArray<float> V_({G.get_size(1), G.get_size(1)});
        hoNDArray<float> U_({G.get_size(0), G.get_size(0)});
        arma::fmat Vm = as_arma_matrix(V_);
        arma::fmat Um = as_arma_matrix(U_);
        arma::fvec Sv;
        auto A_ = real(G);
        auto A = as_arma_matrix(A_);

        arma::svd(Um, Sv, Vm, A);

        hoNDArray<std::complex<float>> signal({Um.n_rows, 2});

        auto threshold_eigenvalue = 0.1;
        std::vector<float> idx_dom_motion;
        for (auto jj = 0; jj < Sv.n_rows; jj++)
        {
            if (abs(Um.col(0)[jj]) > 0.1)
                idx_dom_motion.push_back(jj);
        }

        std::vector<float> idx_neg_corr;
        hoNDArray<float> C_dom_motion(idx_dom_motion.size(), idx_dom_motion.size());

        for (auto jj = 0; jj < idx_dom_motion.size(); jj++)
        {
            for (auto ii = 0; ii < idx_dom_motion.size(); ii++)
            {
                C_dom_motion(ii, jj) = C(idx_dom_motion[jj], idx_dom_motion[ii]);
            }
        }

        for (auto jj = 0; jj < idx_dom_motion.size(); jj++)
        {
            if (C_dom_motion(jj, 1) < 0)
                idx_neg_corr.push_back(jj);
        }

        hoNDArray<float> clusterNav(centerNavData.get_size(0), idx_dom_motion.size());

        for (auto jj = 0; jj < idx_dom_motion.size(); jj++)
        {
            auto it = find(idx_neg_corr.begin(), idx_neg_corr.end(), jj);
            auto temp1 = hoNDArray<std::complex<float>>(centerNavData(slice, idx_dom_motion[jj]));
            temp1 *= ((it != idx_neg_corr.end()) ? -1.0 : 1.0);
            //  auto filename = "/opt/data/gt_data/cluster_nav" + std::string("_") + std::to_string(jj) + ".complex";
            //  lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(temp1, filename);

            clusterNav(slice, jj) = Gadgetron::real(temp1);
        }
        clusterNav = sum(clusterNav, 1);
        clusterNav /= idx_dom_motion.size();

        auto sig = clusterNav;
        sig.squeeze();

        auto filenametowrite = "/opt/data/gt_data/" + std::string(this->header.studyInformation.get().studyDate->c_str()) + "_" + std::string(this->header.studyInformation.get().studyTime->c_str()) + "_NumberofINT" + std::to_string(sig.get_size(0)) + "_NumberofCH" + std::to_string(numCH) + "_respiratorymotionsignal.real";
        lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(sig, filenametowrite);

        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(sig, "/opt/data/gt_data/sig.real");

        float minsig;
        minValue(sig, minsig);
        for (auto ii = 0; ii < sig.get_size(0); ii++)
            sig(ii) = sig(ii) - minsig;

        float maxsig;
        maxValue(sig, maxsig);

        sig /= maxsig;
        //lit_sgncr_toolbox::utils::write_cpu_nd_array<float>(sig, "/opt/data/gt_data/sig_norm.real");
        std::vector<std::tuple<float, float>> binLimits;
        if (useStableBinning)
            binLimits = stableBinning(abs(sig), acceptancePercent);
        else
            binLimits = respiratoryPhases(abs(sig));
        // [ mins, maxs ]
        // if(doStableBinning)
        //     auto [mins, maxs] = stableBinning(sig, acceptancePercent);
        // else
        //     auto rep_phase_lims = respiratoryPhases(sig);

        std::vector<std::vector<float>> acceptedTimes;
        for (auto [mins, maxs] : binLimits)
        {
            std::vector<float> tempAT;
            for (auto i = 0; i < sig.get_size(0); i++)
            {
                if (sig[i] < maxs && sig[i] >= mins)
                    tempAT.push_back(float(headers[i].acquisition_time_stamp));
            }

            if (tempAT.empty())
            {
                GDEBUG_STREAM("Binning failed reconstruction all data: If this is not a phantom experiment something failed");
                for (auto i = 0; i < sig.get_size(0); i++)
                {
                    tempAT.push_back(float(headers[i].acquisition_time_stamp));
                }
            }
            acceptedTimes.push_back(tempAT);
        }
        return std::make_tuple(samplingTime / 2.5, acceptedTimes);
    }
    void process(InputChannel<Acquisition> &in, OutputChannel &out) override
    {
        GadgetronTimer timer("Gating and Binning Total:");

        std::vector<Core::Acquisition> acquisitionsvec;
        std::vector<Core::Acquisition> navigatoracqvec;
        std::vector<ISMRMRD::AcquisitionHeader> headers;
        std::vector<float> traj_angles;
        std::vector<uint32_t> acq_tstamp;
        std::vector<uint32_t> nav_tstamp;
        auto data_sent = false;
        float val = 0;
        uint64_t lastFlags;
        auto maxZencode = header.encoding.front().encodingLimits.kspace_encoding_step_2.get().maximum + 1;
        std::vector<float> navAngles;
        // Collect all the data for calculating the gating signal and for binning -
        // costly but necessary there may be other smarter ways of doing this optimize later.
        for (auto message : in)
        {

            auto &[head, data, traj] = (message);
            if (head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT))
                lastFlags = head.flags;

            if (useDC && (useDCall ? true : (head.idx.kspace_encode_step_2 == maxZencode / 2)) && !((head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA))))
            {
                using namespace Gadgetron::Indexing;
                auto &[head, data, traj] = message;
                auto extraSamples = data.get_size(0) - traj->get_size(1);

                auto data_cropped = crop<std::complex<float>, 2>(vector_td<size_t, 2>(0, 0),
                                                                 vector_td<size_t, 2>(extraSamples, data.get_size(1)),
                                                                 data);
                auto trajectory = traj.value();
                //allangles=rad2deg(angle(traj.uncorrected(25,:,1)+1j*traj.uncorrected(25,:,2)));
                if (val != std::arg(std::complex<float>(trajectory(0, 25), trajectory(1, 25))))
                {
                    val = std::arg(std::complex<float>(trajectory(0, 25), trajectory(1, 25)));
                    //  GDEBUG_STREAM("Angles " << radianstodegrees(val));
                }
                nav_tstamp.push_back(head.acquisition_time_stamp);
                navAngles.push_back(val);
                navigatoracqvec.push_back(Core::Acquisition(head, data_cropped, traj));
            }
            if (!useDC && (head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {

                navigatoracqvec.push_back(message);
                nav_tstamp.push_back(head.acquisition_time_stamp);

                continue;
            }
            if (!(head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_NAVIGATION_DATA) || head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_IS_RTFEEDBACK_DATA)))
            {
                using namespace Gadgetron::Indexing;
                auto &[head, data, traj] = message;
                auto extraSamples = data.get_size(0) - traj->get_size(1);
                head.number_of_samples = data.get_size(0) - extraSamples;

                data = crop<std::complex<float>, 2>(vector_td<size_t, 2>(10, 0),
                                                    vector_td<size_t, 2>(data.get_size(0) - extraSamples, data.get_size(1)),
                                                    data);
                acquisitionsvec.push_back(Core::Acquisition(head, data, traj));
                headers.push_back(head);
                auto trajectory = traj.value();
                //allangles=rad2deg(angle(traj.uncorrected(25,:,1)+1j*traj.uncorrected(25,:,2)));
                if (val != std::arg(std::complex<float>(trajectory(0, 25), trajectory(1, 25))))
                {
                    val = std::arg(std::complex<float>(trajectory(0, 25), trajectory(1, 25)));
                    //  GDEBUG_STREAM("Angles " << radianstodegrees(val));
                }

                traj_angles.push_back(radianstodegrees(val));
                acq_tstamp.push_back(head.acquisition_time_stamp);
                // out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj))); // Pushes the acquisitions as they come in so we have the CSM ready
                //continue;
            }
            if (head.isFlagSet(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT))
            {
                if (!testAcceptance)
                {
                    float samplingTime;
                    std::vector<std::vector<float>> acceptedTimesAll;
                    GadgetronTimer timer("Binning:");
                    if (useDC)
                    {
                        auto out = estimateGatingSignal(navigatoracqvec, acceptancePercentage, navAngles);
                        samplingTime = std::get<0>(out);
                        acceptedTimesAll = std::get<1>(out);
                    }
                    else
                    {
                        navAngles = findNavAngle(nav_tstamp, acq_tstamp, traj_angles);
                        auto out = estimateGatingSignal(navigatoracqvec, acceptancePercentage, navAngles);
                        samplingTime = std::get<0>(out);
                        acceptedTimesAll = std::get<1>(out);
                    }
                    std::vector<size_t> idx_to_send;

                    // Super inefficient
                    //#pragma omp parallel private() shared
                    auto phase = useStableBinning ? 1 : 0;
                    for (auto acceptedTimes : acceptedTimesAll)
                    {
                        auto counter = 0;
                        {
                            GadgetronTimer timer("Binning idx_to_send :");
                            //#pragma omp parallel for
                            for (auto i = 0; i < headers.size(); i++)
                            {
                                // #pragma omp parallel
                                // #pragma omp for
                                for (int j = counter; j < acceptedTimes.size(); j++)
                                {
                                    auto t = acceptedTimes[j];
                                    if (abs(t - float(acq_tstamp[i])) < samplingTime)
                                    {
                                        if (!std::count(idx_to_send.begin(), idx_to_send.end(), i))
                                        {
                                            idx_to_send.push_back(i);
                                            counter = j;
                                        }
                                    }
                                    if (float(acq_tstamp[i]) - t < 0 && abs(t - float(acq_tstamp[i])) > 2 * samplingTime)
                                        continue;
                                }
                            }
                        }
                        {
                            GadgetronTimer timer("Binning send :");
                            int sent_counter = 0;
                            //#pragma omp parallel for shared(acquisitionsvec,idx_to_send,sent_counter)
                            // #pragma omp parallel
                            // #pragma omp for
                            for (auto i = 0; i < acquisitionsvec.size(); i++)
                            {

                                if (!idx_to_send.empty() && i == idx_to_send[sent_counter])
                                {
                                    //out.push(std::move(acquisitionsvec[idx_to_send[i]]));
                                    auto [head, data, traj] = (acquisitionsvec[idx_to_send[sent_counter]]);
                                    //head.flags = lastFlags;
                                    //head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
                                    head.idx.phase = phase;
                                    out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                                    sent_counter++;
                                }
                                else
                                {
                                    if (useStableBinning)
                                    {
                                        auto [head, data, traj] = (acquisitionsvec[i]);
                                        head.idx.phase = 0;
                                        out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                                        //head.flags = lastFlags;
                                        //head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
                                    }
                                }
                            }
                            idx_to_send.clear();
                        }
                        phase++;
                    }
                    acq_tstamp.clear();
                    nav_tstamp.clear();
                    acquisitionsvec.clear();
                    navigatoracqvec.clear();
                    traj_angles.clear();
                    navAngles.clear();
                    data_sent = true;
                }
                //     else
                //     {
                //         for (float ap = 40; ap >= 10; ap = ap - 10)
                //         {
                //             auto tempacq = navigatoracqvec;

                //             auto navAngles = findNavAngle(nav_tstamp, acq_tstamp, traj_angles);
                //             auto [samplingTime, acceptedTimes] = estimateGatingSignal(tempacq, ap, navAngles);

                //             std::vector<size_t> idx_to_send;
                //             auto counter = 0;
                //             // Super inefficient
                //             //#pragma omp parallel for
                //             for (auto i = 0; i < headers.size(); i++)
                //             {
                //                 for (int j = counter; j < acceptedTimes.size(); j++)
                //                 {
                //                     auto t = acceptedTimes[j];
                //                     if (abs(t - float(headers[i].acquisition_time_stamp)) < samplingTime)
                //                     {
                //                         if (!std::count(idx_to_send.begin(), idx_to_send.end(), i))
                //                         {
                //                             idx_to_send.push_back(i);
                //                             counter = j;
                //                         }
                //                     }
                //                     if (float(headers[i].acquisition_time_stamp) - t < 0 && abs(t - float(headers[i].acquisition_time_stamp)) > 2 * samplingTime)
                //                         continue;
                //                 }
                //             }
                //             //#pragma omp parallel for
                //             for (auto i = 0; i < idx_to_send.size(); i++)
                //             {
                //                 if (i < idx_to_send.size() - 1)
                //                 {
                //                     out.push(std::move(acquisitionsvec[idx_to_send[i]]));
                //                 }
                //                 else
                //                 {
                //                     auto &[head, data, traj] = (acquisitionsvec[idx_to_send[i]]);
                //                     head.flags = lastFlags;
                //                     head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);

                //                     out.push(std::move(Core::Acquisition((head), (data), (traj))));
                //                 }
                //             }
                //             idx_to_send.clear();
                //         }
                //     }
            }
        }
        if (!data_sent)
        {
            if (!testAcceptance)
            {
                float samplingTime;
                std::vector<std::vector<float>> acceptedTimesAll;
                GadgetronTimer timer("Binning:");
                if (useDC)
                {
                    auto out = estimateGatingSignal(navigatoracqvec, acceptancePercentage, navAngles);
                    samplingTime = std::get<0>(out);
                    acceptedTimesAll = std::get<1>(out);
                }
                else
                {
                    navAngles = findNavAngle(nav_tstamp, acq_tstamp, traj_angles);
                    auto out = estimateGatingSignal(navigatoracqvec, acceptancePercentage, navAngles);
                    samplingTime = std::get<0>(out);
                    acceptedTimesAll = std::get<1>(out);
                }
                std::vector<size_t> idx_to_send;

                // Super inefficient
                //#pragma omp parallel private() shared
                auto phase = useStableBinning ? 1 : 0;
                for (auto acceptedTimes : acceptedTimesAll)
                {
                    auto counter = 0;
                    {
                        GadgetronTimer timer("Binning idx_to_send :");
                        //#pragma omp parallel for
                        for (auto i = 0; i < headers.size(); i++)
                        {
                            // #pragma omp parallel
                            // #pragma omp for
                            for (int j = counter; j < acceptedTimes.size(); j++)
                            {
                                auto t = acceptedTimes[j];
                                if (abs(t - float(acq_tstamp[i])) < samplingTime)
                                {
                                    if (!std::count(idx_to_send.begin(), idx_to_send.end(), i))
                                    {
                                        idx_to_send.push_back(i);
                                        counter = j;
                                    }
                                }
                                if (float(acq_tstamp[i]) - t < 0 && abs(t - float(acq_tstamp[i])) > 2 * samplingTime)
                                    continue;
                            }
                        }
                    }
                    {
                        GadgetronTimer timer("Binning send :");
                        int sent_counter = 0;
                        //#pragma omp parallel for shared(acquisitionsvec,idx_to_send,sent_counter)
                        // #pragma omp parallel
                        // #pragma omp for
                        for (auto i = 0; i < acquisitionsvec.size(); i++)
                        {

                            if (!idx_to_send.empty() && i == idx_to_send[sent_counter])
                            {
                                //out.push(std::move(acquisitionsvec[idx_to_send[i]]));
                                auto [head, data, traj] = (acquisitionsvec[idx_to_send[sent_counter]]);
                                //head.flags = lastFlags;
                                //head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
                                head.idx.phase = phase;
                                out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                                sent_counter++;
                            }
                            else
                            {
                                if (useStableBinning)
                                {
                                    auto [head, data, traj] = (acquisitionsvec[i]);
                                    head.idx.phase = 0;
                                    out.push(Core::Acquisition(std::move(head), std::move(data), std::move(traj)));
                                    //head.flags = lastFlags;
                                    //head.setFlag(ISMRMRD::ISMRMRD_ACQ_LAST_IN_MEASUREMENT);
                                }
                            }
                        }
                        idx_to_send.clear();
                    }
                    phase++;
                }
                acq_tstamp.clear();
                nav_tstamp.clear();
                acquisitionsvec.clear();
                navigatoracqvec.clear();
                traj_angles.clear();
                navAngles.clear();
            }
        }
    }

protected:
    ISMRMRD::IsmrmrdHeader header;
    NODE_PROPERTY(acceptancePercentage, float, "BinningAcceptance", 40);
    NODE_PROPERTY(doStableBinning, bool, "StableBinning", true);
    NODE_PROPERTY(numberOfBins, size_t, "numberOfBins", 4);
    NODE_PROPERTY(testAcceptance, bool, "TestAcceptance", false);
    NODE_PROPERTY(useDC, bool, "useDC", false);
    NODE_PROPERTY(useDCall, bool, "useDCalldata", false);
    NODE_PROPERTY(doAngularFilteration, bool, "perform AngularFilteration", true);
    NODE_PROPERTY(dotwostepSVD, bool, "dotwoStepsvd", false);
    NODE_PROPERTY(cs_freq_filter, float, "channel selection frequency filter sigma", 0.05); // cs = channel selection frequency filter
    NODE_PROPERTY(useStableBinning, bool, "Use stable binning", true);                      // cs = channel selection frequency filter
    NODE_PROPERTY(up_perc, float, "up_perc for binning", 0.95);
    NODE_PROPERTY(low_perc, float, "low_perc for binning", 0.05);

private:
};

GADGETRON_GADGET_EXPORT(GatingandBinningGadget)