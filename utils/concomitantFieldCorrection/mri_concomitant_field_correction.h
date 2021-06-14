/*
 * mri_concomitant_field_correction.h
 *
 *  Created on: November 10th, 2020
 *      Author: Ahsan Javed
 */
#pragma once

#include <gadgetron/vector_td.h>
#include <vector>
#include <complex>
#include <gadgetron/hoNDArray.h>
#include <gadgetron/hoNDFFT.h>
#include <gadgetron/hoNDArray_utils.h>
#include <gadgetron/hoNDArray_math.h>
#include <boost/optional.hpp>
#include <gadgetron/hoNDArray_fileio.h>
#include <boost/math/constants/constants.hpp>
#include <math.h>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/filesystem/fstream.hpp>
#include "armadillo"

#include <ismrmrd/xml.h>
#include <gadgetron/mri_core_utility.h> // Added MRD
#include "../../utils/util_functions.h"

using namespace Gadgetron;

namespace lit_sgncr_toolbox
{
    namespace corrections
    {
        class mri_concomitant_field_correction
        {
        public:
            mri_concomitant_field_correction() = default;
            mri_concomitant_field_correction(const ISMRMRD::IsmrmrdHeader &h);
            void set_offset(ISMRMRD::AcquisitionHeader acq_header);
            void set_rotationMatrix(ISMRMRD::AcquisitionHeader acq_header);
            void generate_mesh_grid(ISMRMRD::AcquisitionHeader acq_header);
            void calculate_fieldCoordinates();
            void calculate_scaledTime(hoNDArray<floatd3> gradients);
            void calculate_freqMap();
            void calculate_combinationWeights();
            void set_combinationWeights();
            void setup(hoNDArray<floatd3> gradients, ISMRMRD::AcquisitionHeader acq_header);

            hoNDArray<float> calculate_gradXYamp(hoNDArray<floatd3> gradients);
            hoNDArray<float> calculate_gradCsum(hoNDArray<float> gradients);

            int find_index(arma::vec in, float val);

            size_t numfreqbins;
            arma::vec demodulation_freqs;
            hoNDArray<std::complex<float>> combinationWeights;
            hoNDArray<float> scaled_time;

        private:
            float gamma = 42.57e6;
            float B0;
            float sampling_time = 2e-6;

        protected:
            arma::fvec3 fov;
            arma::fvec3 steps;
            arma::fvec3 res;
            arma::fvec3 offset;

            arma::cube X;
            arma::cube Y;
            arma::cube Z;
            arma::cube field_cords;
            arma::cube freqMap;

            arma::fvec3 logical_fov;
            arma::fvec3 logical_steps;
            arma::fvec3 logical_res;
            arma::fvec3 logical_offset;

            arma::fmat33 rotation_matrix;
            arma::fmat33 R_DCS_PCS;

            arma::cx_mat C;

            float maxGA; // max(vec(sqrt(gradients_logical(:,:,1).^2+gradients_logical(:,:,2).^2)))
        };
    } // namespace corrections
} // namespace lit_sgncr_toolbox
