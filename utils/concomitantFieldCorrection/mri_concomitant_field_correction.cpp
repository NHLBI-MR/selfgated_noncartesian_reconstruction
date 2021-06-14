/*
 * mri_concomitant_field_correction.cpp
 *
 *  Created on: November 10th, 2020
 *      Author: Ahsan Javed
 */

#include "mri_concomitant_field_correction.h"
#include <gadgetron/mri_core_kspace_filter.h>
#include <math.h>

namespace lit_sgncr_toolbox
{
    namespace corrections
    {
        mri_concomitant_field_correction::mri_concomitant_field_correction(const ISMRMRD::IsmrmrdHeader &head)
        {

            R_DCS_PCS = lit_sgncr_toolbox::utils::lookup_PCS_DCS(head.measurementInformation.get().patientPosition);

            B0 = 0.55; // head.acquisitionSystemInformation.get().systemFieldStrength_T; -> Not giving the correct field value bug !

            fov(0) = head.encoding[0].encodedSpace.fieldOfView_mm.x / 1000; // meters
            fov(1) = head.encoding[0].encodedSpace.fieldOfView_mm.y / 1000; // meters
            fov(2) = head.encoding[0].encodedSpace.fieldOfView_mm.z / 1000; // meters

            steps(0) = head.encoding[0].encodedSpace.matrixSize.x;
            steps(1) = head.encoding[0].encodedSpace.matrixSize.y;
            steps(2) = head.encoding[0].encodedSpace.matrixSize.z;

            res = fov / steps;
        }

        void mri_concomitant_field_correction::set_offset(ISMRMRD::AcquisitionHeader acq_header)
        {
            offset(0) = acq_header.position[0];
            offset(1) = acq_header.position[1];
            offset(2) = acq_header.position[2];
            offset = offset / 1000; // convert to m
        }

        void mri_concomitant_field_correction::set_rotationMatrix(ISMRMRD::AcquisitionHeader acq_header)
        {
            rotation_matrix(0, 0) = acq_header.read_dir[0];
            rotation_matrix(1, 0) = acq_header.read_dir[1];
            rotation_matrix(2, 0) = acq_header.read_dir[2];
            rotation_matrix(0, 1) = acq_header.phase_dir[0];
            rotation_matrix(1, 1) = acq_header.phase_dir[1];
            rotation_matrix(2, 1) = acq_header.phase_dir[2];
            rotation_matrix(0, 2) = acq_header.slice_dir[0];
            rotation_matrix(1, 2) = acq_header.slice_dir[1];
            rotation_matrix(2, 2) = acq_header.slice_dir[2];
        }

        void mri_concomitant_field_correction::generate_mesh_grid(ISMRMRD::AcquisitionHeader acq_header)
        {
            set_offset(acq_header);
            set_rotationMatrix(acq_header);

            logical_fov = fov.as_col();
            logical_steps = steps.as_col();
            logical_res = res.as_col();
            logical_offset = R_DCS_PCS * rotation_matrix * offset.as_col();

            unsigned int warp_size = cudaDeviceManager::Instance()->warp_size();
            auto newzsize = warp_size * int(logical_steps(2) / warp_size + 1);

            auto logical_coords_x = arma::linspace(-1 * (logical_fov(0) * 0.5 - logical_res(0) * 0.5) + logical_offset(0), (logical_fov(0) * 0.5 - logical_res(0) * 0.5) + logical_offset(0), abs(logical_steps(0)));
            auto logical_coords_y = arma::linspace(-1 * (logical_fov(1) * 0.5 - logical_res(1) * 0.5) + logical_offset(1), (logical_fov(1) * 0.5 - logical_res(1) * 0.5) + logical_offset(1), abs(logical_steps(1)));
            auto logical_coords_z = arma::linspace(-1 * (logical_fov(2) * 0.5 - logical_res(2) * 0.5) + logical_offset(2), (logical_fov(2) * 0.5 - logical_res(2) * 0.5) + logical_offset(2), abs(logical_steps(2)));

            // auto newLogical_cord_z = arma::vec(newzsize,arma::fill::zeros);
            // auto lenToappend = (newzsize-logical_steps(2))/2.0;

            // newLogical_cord_z(arma::span(0,lenToappend-1)) = logical_coords_z(arma::span(logical_steps(2)-lenToappend,logical_steps(2)-1));
            // newLogical_cord_z(arma::span(newzsize-lenToappend,newzsize-1)) = logical_coords_z(arma::span(0,lenToappend-1));

            //logical_coords_z = newLogical_cord_z;
            //logical_steps(2) = newzsize;

            X = arma::cube(logical_steps(0), logical_steps(1), logical_steps(2), arma::fill::zeros);
            Y = arma::cube(logical_steps(0), logical_steps(1), logical_steps(2), arma::fill::zeros);
            Z = arma::cube(logical_steps(0), logical_steps(1), logical_steps(2), arma::fill::zeros);

            auto X_slice = arma::mat(arma::repmat(logical_coords_x.as_col(), 1, logical_steps(0)));
            auto Y_slice = arma::mat(arma::repmat(logical_coords_y.as_row(), logical_steps(1), 1));
            //auto Z_slice = arma::mat(logical_steps(0), logical_steps(1), arma::fill::ones); // Doesnt work getting creative
            auto Z_slice = arma::mat(arma::repmat(logical_coords_z.as_row(), logical_steps(1), 1));
            // Creates the Mesh Grid Not very clean but should work
            for (auto ii = 0; ii < logical_steps(2); ii++)
            {
                X.slice(ii) = X_slice;
            }
            for (auto ii = 0; ii < logical_steps(2); ii++)
            {
                Y.slice(ii) = Y_slice;
            }
            for (auto ii = 0; ii < logical_steps(0); ii++)
            {
                Z.row(ii) = Z_slice; // * logical_coords_z(ii);
            }
        }
        void mri_concomitant_field_correction::calculate_fieldCoordinates()
        {
            GDEBUG_STREAM("Calculating Field coords");

            auto A = arma::fmat33(R_DCS_PCS * rotation_matrix);

            auto a1 = A(0, 0);
            auto a2 = A(0, 1);
            auto a3 = A(0, 2);
            auto a4 = A(1, 0);
            auto a5 = A(1, 1);
            auto a6 = A(1, 2);
            auto a7 = A(2, 0);
            auto a8 = A(2, 1);
            auto a9 = A(2, 2);

            auto F1 = float(0.25) * (a1 * a1 + a4 * a4) * (a7 * a7 + a8 * a8) + a7 * a7 * (a2 * a2 + a5 * a5) - a7 * a8 * (a1 * a2 + a4 * a5);
            auto F2 = float(0.25) * (a2 * a2 + a5 * a5) * (a7 * a7 + a8 * a8) + a8 * a8 * (a1 * a1 + a4 * a4) - a7 * a8 * (a1 * a2 + a4 * a5);
            auto F3 = float(0.25) * (a3 * a3 + a6 * a6) * (a7 * a7 + a8 * a8) + a9 * a9 * (a1 * a1 + a2 * a2 + a4 * a4 + a5 * a5) - a7 * a9 * (a1 * a3 + a4 * a6) - a8 * a9 * (a2 * a2 + a5 * a6);
            auto F4 = float(0.5) * (a2 * a3 + a5 * a6) * (a7 * a7 - a8 * a8) + a8 * a9 * (2 * a1 * a1 + a2 * a2 + 2 * a4 * a4 + a5 * a5) - a7 * a8 * (a1 * a3 + a4 * a6) - a7 * a9 * (a1 * a2 + a4 * a5);
            auto F5 = float(0.5) * (a1 * a3 + a4 * a6) * (a8 * a8 - a7 * a7) + a7 * a9 * (a1 * a1 + 2 * a2 * a2 + a4 * a4 + 2 * a5 * a5) - a7 * a8 * (a2 * a3 + a5 * a6) - a8 * a9 * (a1 * a2 + a4 * a5);
            auto F6 = float(-0.5) * (a1 * a2 + a4 * a5) * (a8 * a8 - a7 * a7) + a7 * a8 * (a1 * a1 + a2 * a2 + a4 * a4 + a5 * a5);

            field_cords = (F1 * X % X + F2 * Y % Y + F3 * Z % Z +
                           F4 * Y % Z + F5 * X % Z + F6 * X % Y);
        }
        //        tc_t  = @(gradients_logical,dt) 1/max(vec(sqrt(gradients_logical(:,:,1).^2+gradients_logical(:,:,2).^2))).^2 * cumsum(gradients_logical(:,:,1).^2+gradients_logical(:,:,2).^2,1)*dt;
        void mri_concomitant_field_correction::calculate_scaledTime(hoNDArray<floatd3> gradients)
        {
            GDEBUG_STREAM("Calculating Scaled time");

            gradients /= float(1000.0);
            auto gradientAmplitude = calculate_gradXYamp(gradients);
            auto csumGradamp = calculate_gradCsum(gradientAmplitude);

            sqrt_inplace(&gradientAmplitude);
            maxGA = max(&gradientAmplitude);
            csumGradamp *= float(1.0) / (maxGA * maxGA) * sampling_time;
            scaled_time = csumGradamp;
            GDEBUG_STREAM("max Grad Am : " << maxGA);
        }

        hoNDArray<float> mri_concomitant_field_correction::calculate_gradCsum(hoNDArray<float> gradients)
        {
            hoNDArray<float> csumGrad({gradients.get_size(0), gradients.get_size(1)});

            for (auto jj = 0; jj < gradients.get_size(1); jj++)
            {
                //  auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
                // auto csumGrad_ptr = csumGrad.get_data_ptr() + jj * csumGrad.get_size(0);
                for (auto ii = 0; ii < gradients.get_size(0); ii++)
                {
                    if (ii > 0)
                    {
                        csumGrad(ii, jj) = gradients(ii, jj) + csumGrad(ii - 1, jj);
                    }
                    else
                    {
                        csumGrad(ii, jj) = gradients(ii, jj);
                    }
                }
            }
            return csumGrad;
        }

        hoNDArray<float> mri_concomitant_field_correction::calculate_gradXYamp(hoNDArray<floatd3> gradients)
        {
            hoNDArray<float> gradAmp({gradients.get_size(0), gradients.get_size(1)});

            //GDEBUG_STREAM("gradients Size 0 " << gradients.get_size(0));
            //GDEBUG_STREAM("gradients Size 1 " << gradients.get_size(1));

            for (auto jj = 0; jj < gradients.get_size(1); jj++)
            {
                //auto grad_ptr = gradients.get_data_ptr() + jj * gradients.get_size(0);
                //auto gradAmp_ptr = gradAmp.get_data_ptr() + jj * gradAmp.get_size(0);
                for (auto ii = 0; ii < gradients.get_size(0); ii++)
                {
                    gradAmp(ii, jj) = (gradients(ii, jj)[0] * gradients(ii, jj)[0] + gradients(ii, jj)[1] * gradients(ii, jj)[1]);
                }
            }
            return gradAmp;
        }
        void mri_concomitant_field_correction::calculate_freqMap()
        {
            // There may be a slight discrepancy with matlab please explore further later
            freqMap = float(1.0 / (float(4.0) * B0)) * float(gamma) * (maxGA * maxGA) * field_cords;
         }
        void mri_concomitant_field_correction::calculate_combinationWeights()
        {
            GDEBUG_STREAM("Calculating combination weights");
            GDEBUG_STREAM("scaled_time dims 0 : " << scaled_time.get_size(0));
            GDEBUG_STREAM("scaled_time dims 1 : " << scaled_time.get_size(1));
            auto t = sum(scaled_time, 1);
            t /= float(scaled_time.get_size(1));

            auto time = as_arma_col(t);

            //  GDEBUG_STREAM("demodulation_freqs : " << this->demodulation_freqs);
            //  GDEBUG_STREAM("time : " << time);
            //  GDEBUG_STREAM("Aimag : " << time * this->demodulation_freqs.as_row());
            //  GDEBUG_STREAM("Aimag : " << float(M_PI) * time * this->demodulation_freqs.as_row());
            //  GDEBUG_STREAM("Aimag : " << float(2.0) * float(M_PI) * time * this->demodulation_freqs.as_row());
            // GDEBUG_STREAM("Aimag : " << argA_imag);

            auto A = arma::cx_mat(arma::cos(float(2.0) * float(M_PI) * time * this->demodulation_freqs.as_row()), arma::sin(float(2.0) * float(M_PI) * time * this->demodulation_freqs.as_row()));
            auto Ah = A.t();
            arma::cx_mat Q; 
            arma::cx_mat R; 
            arma::qr(Q,R,A);
            auto Qh = Q.t();
            // GDEBUG_STREAM("A rows: " << A.n_rows);
            // GDEBUG_STREAM("A cols: " << A.n_cols);

            auto freqs = arma::vec(arma::unique(arma::trunc(arma::vectorise(freqMap))));

            GDEBUG_STREAM("freqs unique size : " << freqs.n_elem);
            auto C_r = arma::cx_mat(freqs.n_elem, demodulation_freqs.n_elem, arma::fill::zeros);
            C = arma::cx_mat(freqMap.n_elem, demodulation_freqs.n_elem);
            //arma::fmat C_i = arma::fmat(freqs.n_elem, demodulation_freqs.n_elem,arma::fill::zeros);
            //             GDEBUG_STREAM("A : " << A);
            // GDEBUG_STREAM("Ah : " << Ah);
//#pragma omp parallel for
            for (auto ii = 0; ii < freqs.n_elem; ii++)
            {
                auto b = arma::cx_fcolvec(arma::cos(float(2.0) * float(M_PI) * freqs(ii) * time), arma::sin(float(2.0) * float(M_PI) * freqs(ii) * time));
                //  GDEBUG_STREAM("b : " << b);
                //auto x = arma::cx_vec(arma::pinv(Ah * A) * Ah * b);
                auto x = arma::cx_vec(arma::pinv(R) * Qh * b);
                //  GDEBUG_STREAM("x : " << x);

                //   arma::cx_vec x;
                //  arma::solve(x,A,b);

                C_r.row(ii) = x.as_row();
            }
            // GDEBUG_STREAM("Freqs: " << freqs);
            auto vec_freqmap = arma::vec(arma::trunc(arma::vectorise(freqMap)));
//#pragma omp parallel for            
            for (auto ii = 0; ii < freqMap.n_elem; ii++)
            {
                //GDEBUG_STREAM("Freq Map: " << vec_freqmap(ii));

                // auto index = arma::find(freqs.as_col() == vec_freqmap(ii));
                auto index = find_index(freqs, vec_freqmap(ii));
                //  GDEBUG_STREAM("Cw @ ii: " << C_r.row(index));

                C.row(ii) = C_r.row(index);
            }
        }
        int mri_concomitant_field_correction::find_index(arma::vec in, float val)
        {
            for (auto ii = 0; ii < in.n_elem; ii++)
            {
                if (in(ii) == val)
                    return ii;
            }
            return -1;
        }
        void mri_concomitant_field_correction::set_combinationWeights()
        {
            hoNDArray<std::complex<float>> output(C.n_rows, C.n_cols);

#pragma omp parallel for
            for (auto ii = 0; ii < C.n_rows; ii++)
            {
                for (auto jj = 0; jj < C.n_cols; jj++)
                {
                    output(ii, jj) = C(ii, jj);
                }
            }
            output.reshape(freqMap.n_rows, freqMap.n_cols, freqMap.n_slices, -1);
            combinationWeights = output;
        }
        void mri_concomitant_field_correction::setup(hoNDArray<floatd3> gradients, ISMRMRD::AcquisitionHeader acq_header)
        {
            GDEBUG_STREAM("Setting up concomitant Field Correction");

            generate_mesh_grid(acq_header);
            calculate_fieldCoordinates();
            calculate_scaledTime(gradients);
            calculate_freqMap();

            numfreqbins = std::ceil(4 * freqMap.max() * gradients.get_size(0) * sampling_time);
            demodulation_freqs = arma::linspace(freqMap.min(), freqMap.max() * 1.1, numfreqbins);

            GDEBUG_STREAM("Number of Demodulation bin " << numfreqbins);
            GDEBUG_STREAM("Freq Range Min: " << demodulation_freqs.min());
            GDEBUG_STREAM("Freq Range Max: " << demodulation_freqs.max());

            calculate_combinationWeights();
            set_combinationWeights();
        }

    } // namespace corrections
} // namespace lit_sgncr_toolbox