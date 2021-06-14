
#include <gadgetron/Node.h>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/mri_core_acquisition_bucket.h>
#include <complex>
#include <ismrmrd/ismrmrd.h>
#include <ismrmrd/xml.h>
#include <gadgetron/gadgetron_mricore_export.h>
#include <gadgetron/mri_core_data.h>
#include <gadgetron/hoNDArray_math.h>
#include <gadgetron/hoNDArray_utils.h>
#include <fstream>
#include <vector>
#include "armadillo"
#include <gadgetron/mri_core_utility.h>
#include <gadgetron/mri_core_def.h>
#include <gadgetron/mri_core_girf_correction.h>

using namespace Gadgetron;

    class WaveformToTrajectory
        : public Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>
    {
    public:
        using Core::ChannelGadget<Core::variant<Core::Acquisition, Core::Waveform>>::ChannelGadget;
        WaveformToTrajectory(const Core::Context &context, const Core::GadgetProperties &props);
        
        hoNDArray<std::complex<float>> girf_kernel;

        float newscaling=0.5; 
        float newscaling1=0.5;
        int counterData=0;

        std::map<size_t,hoNDArray<float>> trajectory_map;
        std::map<size_t,Core::Waveform> gradient_wave_store;
        
        size_t curAvg=0;
    protected:
        ISMRMRD::IsmrmrdHeader header;

        NODE_PROPERTY(perform_GIRF, bool, " Perform GIRF", false);
        NODE_PROPERTY(GIRF_folder, std::string, "Path where GIRF Data is stored", "/opt/data/GIRF/");

        int curr_avg = 0;
        float kspace_scaling = 0;
        void prepare_trajectory_from_waveforms(Core::Waveform &grad_waveform,
                                                           const ISMRMRD::AcquisitionHeader &head);
        
        hoNDArray<floatd2> sincInterpolation(const hoNDArray<floatd2> input, int zpadFactor);
        hoNDArray<floatd2> zeroHoldInterpolation(const hoNDArray<floatd2> input, int zpadFactor);

        // void send_data(Core::OutputChannel &out, std::map<unsigned short, AcquisitionBucket> &buckets,
        //                std::vector<Core::Waveform> &waveforms);
        
        void process(Core::InputChannel<Core::variant<Core::Acquisition, Core::Waveform>> &in,
                     Core::OutputChannel &out) override;

        
        void printGradtoFile(std::string fname_grad, hoNDArray<floatd2> grad_traj);
        void printTrajtoFile(std::string fname_grad, hoNDArray<float> grad_traj);

        hoNDArray<float> calculate_weights_Hoge(const hoNDArray<floatd2> &gradients, const hoNDArray<float> &trajectories);
    };


