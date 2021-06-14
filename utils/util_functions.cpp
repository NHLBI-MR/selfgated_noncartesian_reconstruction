
#include "util_functions.h"
#include <gadgetron/GadgetronTimer.h>

using namespace Gadgetron;
using namespace Gadgetron::Core;

namespace lit_sgncr_toolbox
{
    namespace utils
    {
        cuNDArray<float_complext> estimateCoilmaps_slice(cuNDArray<float_complext> &data)
        {
            auto RO = data.get_size(0);
            auto E1 = data.get_size(1);
            auto E2 = data.get_size(2);
            auto CHA = data.get_size(3);

            data = permute(data, {0, 1, 3, 2});

            std::vector<size_t> recon_dims = {RO, E1, CHA}; // Cropped to size of Recon Matrix
            cuNDArray<float_complext> tempcsm(recon_dims);
            recon_dims.push_back(E2);
            cuNDArray<float_complext> csm(recon_dims);

            for (int iSL = 0; iSL < E2; iSL++)
            {
                cudaMemcpy(tempcsm.get_data_ptr(),
                           data.get_data_ptr() + RO * E1 * iSL * CHA,
                           RO * E1 * CHA * sizeof(float_complext), cudaMemcpyDeviceToDevice);

                cudaMemcpy(csm.get_data_ptr() + RO * E1 * iSL * CHA,
                           (estimate_b1_map<float, 2>(tempcsm)).get_data_ptr(),
                           RO * E1 * CHA * sizeof(float_complext), cudaMemcpyDeviceToDevice);
            }

            data = permute(data, {0, 1, 3, 2});

            return permute(csm, {0, 1, 3, 2});
        }
        float correlation(hoNDArray<float>a,hoNDArray<float>b)
        {
            float r = -1;

            float ma, mb;
            ma = Gadgetron::mean( &a );
            mb = Gadgetron::mean( &b );

            size_t N = a.get_number_of_elements();

            const float* pA = a.begin();
            const float* pB = b.begin();

            size_t n;

            double x(0), y(0), z(0);
            for ( n=0; n<N; n++ )
            {
                x += (pA[n]-ma)*(pA[n]-ma);
                y += (pB[n]-mb)*(pB[n]-mb);
                z += (pA[n]-ma)*(pB[n]-mb);
            }

            double p = std::sqrt(x*y);
            if ( p > 0 )
            {
                r = (float)(z/p);
            }
            return r;
        }
        void attachHeadertoImageArray(ImageArray &imarray, ISMRMRD::AcquisitionHeader acqhdr, const ISMRMRD::IsmrmrdHeader &h)
        {

            int n = 0;
            int s = 0;
            int loc = 0;
            std::vector<size_t> header_dims(3);
            header_dims[0] = 1;
            header_dims[1] = 1;
            header_dims[2] = 1;
            imarray.headers_.create(header_dims);
            imarray.meta_.resize(1);

            auto fov = h.encoding.front().encodedSpace.fieldOfView_mm;
            auto val = h.encoding.front().encodingLimits.kspace_encoding_step_0;
            imarray.headers_(n, s, loc).matrix_size[0] = h.encoding.front().encodedSpace.matrixSize.x;
            imarray.headers_(n, s, loc).matrix_size[1] = h.encoding.front().encodedSpace.matrixSize.y;
            imarray.headers_(n, s, loc).matrix_size[2] = h.encoding.front().reconSpace.matrixSize.z;
            imarray.headers_(n, s, loc).field_of_view[0] = fov.x;
            imarray.headers_(n, s, loc).field_of_view[1] = fov.y;
            imarray.headers_(n, s, loc).field_of_view[2] = fov.z;
            imarray.headers_(n, s, loc).channels = 1;
            imarray.headers_(n, s, loc).average = acqhdr.idx.average;
            imarray.headers_(n, s, loc).slice = acqhdr.idx.slice;
            imarray.headers_(n, s, loc).contrast = acqhdr.idx.contrast;
            imarray.headers_(n, s, loc).phase = acqhdr.idx.phase;
            imarray.headers_(n, s, loc).repetition = acqhdr.idx.repetition;
            imarray.headers_(n, s, loc).set = acqhdr.idx.set;
            imarray.headers_(n, s, loc).acquisition_time_stamp = acqhdr.acquisition_time_stamp;
            imarray.headers_(n, s, loc).position[0] = acqhdr.position[0];
            imarray.headers_(n, s, loc).position[1] = acqhdr.position[1];
            imarray.headers_(n, s, loc).position[2] = acqhdr.position[2];
            imarray.headers_(n, s, loc).read_dir[0] = acqhdr.read_dir[0];
            imarray.headers_(n, s, loc).read_dir[1] = acqhdr.read_dir[1];
            imarray.headers_(n, s, loc).read_dir[2] = acqhdr.read_dir[2];
            imarray.headers_(n, s, loc).phase_dir[0] = acqhdr.phase_dir[0];
            imarray.headers_(n, s, loc).phase_dir[1] = acqhdr.phase_dir[1];
            imarray.headers_(n, s, loc).phase_dir[2] = acqhdr.phase_dir[2];
            imarray.headers_(n, s, loc).slice_dir[0] = acqhdr.slice_dir[0];
            imarray.headers_(n, s, loc).slice_dir[1] = acqhdr.slice_dir[1];
            imarray.headers_(n, s, loc).slice_dir[2] = acqhdr.slice_dir[2];
            imarray.headers_(n, s, loc).patient_table_position[0] = acqhdr.patient_table_position[0];
            imarray.headers_(n, s, loc).patient_table_position[1] = acqhdr.patient_table_position[1];
            imarray.headers_(n, s, loc).patient_table_position[2] = acqhdr.patient_table_position[2];
            imarray.headers_(n, s, loc).data_type = ISMRMRD::ISMRMRD_CXFLOAT;
            imarray.headers_(n, s, loc).image_index = 1;
        }

        void filterImagealongSlice(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma)
        {
            hoNDArray<std::complex<float>> fx(image.get_size(0)),
                fy(image.get_size(1)),
                fz(image.get_size(2));

            generate_symmetric_filter(image.get_size(0), fx, ISMRMRD_FILTER_NONE);
            generate_symmetric_filter(image.get_size(1), fy, ISMRMRD_FILTER_NONE);
            generate_symmetric_filter(image.get_size(2), fz, ftype, fsigma, (size_t)std::ceil(fwidth * image.get_size(2)));

            hoNDArray<std::complex<float>> fxyz(image.get_dimensions());
            compute_3d_filter(fx, fy, fz, fxyz);

            Gadgetron::hoNDFFT<float>::instance()->ifft3c(image);
            multiply(&image, &fxyz, &image);
            Gadgetron::hoNDFFT<float>::instance()->fft3c(image);
        }

        void filterImage(hoNDArray<std::complex<float>> &image, ISMRMRDKSPACEFILTER ftype, size_t fwidth, double fsigma)
        {
            hoNDArray<std::complex<float>> fx(image.get_size(0)),
                fy(image.get_size(1)),
                fz(image.get_size(2));

            generate_symmetric_filter(image.get_size(0), fx, ftype, fsigma, size_t(std::ceil(image.get_size(0) * fwidth)));
            generate_symmetric_filter(image.get_size(1), fy, ftype, fsigma, size_t(std::ceil(image.get_size(0) * fwidth)));
            generate_symmetric_filter(image.get_size(2), fz, ISMRMRD_FILTER_NONE, fsigma, fwidth);

            hoNDArray<std::complex<float>> fxyz(image.get_dimensions());
            compute_3d_filter(fx, fy, fz, fxyz);

            Gadgetron::hoNDFFT<float>::instance()->ifft3c(image);
            multiply(&image, &fxyz, &image);
            Gadgetron::hoNDFFT<float>::instance()->fft3c(image);
        }

        template <typename T>
        void write_gpu_nd_array(cuNDArray<T> &data, std::string filename)
        {
            boost::shared_ptr<hoNDArray<T>> data_host = data.to_host();
            write_nd_array<T>(data_host.get(), filename.c_str());
        }

        template <typename T>
        void write_cpu_nd_array(hoNDArray<T> &data, std::string filename)
        {
            auto d = &data;
            write_nd_array<T>(d, filename.c_str());
        }

        template <typename T>
        cuNDArray<T> concat(std::vector<cuNDArray<T>> &arrays)
        {
            if (arrays.empty())
                return cuNDArray<T>();

            const cuNDArray<T> &first = *std::begin(arrays);

            auto dims = first.dimensions();
            auto size = first.size();

            if (!std::all_of(begin(arrays), end(arrays), [&](auto &array) { return dims == array.dimensions(); }) ||
                !std::all_of(begin(arrays), end(arrays), [&](auto &array) { return size == array.size(); }))
            {
                throw std::runtime_error("Array size or dimensions do not match.");
            }

            dims.push_back(arrays.size());
            cuNDArray<T> output(dims);

            auto slice_dimensions = output.dimensions();
            slice_dimensions.pop_back();

            size_t stride = std::accumulate(slice_dimensions.begin(), slice_dimensions.end(), 1, std::multiplies<size_t>());

            for (size_t iter = 0; iter < arrays.size(); iter++)
            {
                cudaMemcpy(output.get_data_ptr() + iter * stride,
                           arrays.at(iter).get_data_ptr(),
                           stride * sizeof(T), cudaMemcpyDeviceToDevice);
            }

            return output;
        }

        template <typename T>
        hoNDArray<T> concat(std::vector<hoNDArray<T>> &arrays)
        {
            GadgetronTimer timer("Concat: ");

            if (arrays.empty())
                return hoNDArray<T>();

            const hoNDArray<T> &first = *std::begin(arrays);

            auto dims = first.dimensions();
            auto size = first.size();
            auto num_dims = first.get_number_of_dimensions();
            for (auto ii = 1; ii < arrays.size(); ii++)
            {
                dims[num_dims - 1] += arrays[ii].dimensions()[num_dims - 1];
            }

            hoNDArray<T> output(dims);
            size_t stride = 0;

            for (auto ii = 0; ii < arrays.size(); ii++)
            {

                std::move(arrays[ii].begin(), arrays[ii].end(), output.begin() + stride);
                stride += arrays[ii].size();
            }
            // for (size_t iter = 0; iter < arrays.size(); iter++)
            // {
            //     // auto strideCur = arrays[iter].size();
            //     // memmove(output.get_data_ptr() + stride,
            //     //        arrays[iter].get_data_ptr(),
            //     //        strideCur * sizeof(T));
            //     // stride += strideCur;

            // }

            return output;
        }

        template <typename T>
        std::vector<T> sliceVec(std::vector<T> &v, int start, int end, int stride)
        {
            int oldlen = v.size();
            int newlen;

            if (end == -1 or end >= oldlen)
            {
                newlen = (oldlen - start) / stride;
            }
            else
            {
                newlen = (end - start) / stride;
            }

            std::vector<T> nv(newlen);

            for (int i = 0; i < newlen; i++)
            {
                nv[i] = v[start + i * stride];
            }
            return nv;
        }

        int selectCudaDevice()
        {
            int totalNumberofDevice = cudaDeviceManager::Instance()->getTotalNumberOfDevice();
            int selectedDevice = 0;
            size_t freeMemory = 0;

            for (int dno = 0; dno < totalNumberofDevice; dno++)
            {
                cudaSetDevice(dno);
                if (cudaDeviceManager::Instance()->getFreeMemory(dno) > freeMemory)
                {
                    freeMemory = cudaDeviceManager::Instance()->getFreeMemory(dno);
                    selectedDevice = dno;
                }
            }
            //GDEBUG_STREAM("Selected Device: " << selectedDevice);
            return selectedDevice;
        }

        std::vector<int> FindCudaDevices(unsigned long req_size)
        {
            int totalNumberofDevice = cudaDeviceManager::Instance()->getTotalNumberOfDevice();
            size_t freeMemory = 0;
            std::vector<int> gpus;
            struct cudaDeviceProp properties;

            /* machines with no GPUs can still report one emulation device */

           // GDEBUG_STREAM("req_size# " << req_size);

            if (req_size < (long)6 * std::pow(1024, 3))
                req_size = (long)6 * std::pow(1024, 3);

            //GDEBUG_STREAM("req_size# " << req_size);

            for (int dno = 0; dno < totalNumberofDevice; dno++)
            {
                cudaSetDevice(dno);
               // GDEBUG_STREAM("Free_memory# " << cudaDeviceManager::Instance()->getFreeMemory(dno));
                 cudaGetDeviceProperties(&properties, dno);
               //  GDEBUG_STREAM("MajorMode# " << properties.major);
               //  GDEBUG_STREAM("Minor# " << properties.minor);
                
                if (cudaDeviceManager::Instance()->getFreeMemory(dno) > req_size && properties.major >=6)
                {
                    gpus.push_back(dno);
                }
            }

            return gpus;
        }

        void setNumthreadsonGPU(int Number)
        {
            omp_set_num_threads(Number);
            int id = omp_get_num_threads();
        }

        std::vector<hoNDArray<float>> estimateDCF_slice(std::vector<std::vector<hoNDArray<floatd3>>> trajectories, std::vector<std::vector<hoNDArray<float>>> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                                        std::vector<size_t> image_dims_, bool fullySampled)
        {
            boost::shared_ptr<cuNDArray<float>> dcw;
            boost::shared_ptr<cuNDArray<floatd3>> traj;
            boost::shared_ptr<cuNDArray<float>> tdcf;

            std::vector<hoNDArray<float>> dcw_vec;
#pragma omp parallel
            omp_set_num_threads(cudaDeviceManager::Instance()->getTotalNumberOfDevice() - 1);
            int id = omp_get_num_threads();
#pragma omp for

            for (auto ii = 0; ii < trajectories.size(); ii++)
            {
                if (!fullySampled || ii == 0)
                {
                    cudaSetDevice(lit_sgncr_toolbox::utils::selectCudaDevice());

                    auto temp = Gadgetron::concat(trajectories[ii]);
                    auto temp_dcf = Gadgetron::concat(dcf[ii]);

                    tdcf = boost::make_shared<cuNDArray<float>>(temp_dcf);
                    traj = boost::make_shared<cuNDArray<floatd3>>(temp);
                    std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
                    cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
                    traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

                    auto t = *(Gadgetron::estimate_dcw<float, 3>((*traj),
                                                                 *tdcf,
                                                                 from_std_vector<size_t, 3>(image_dims_),
                                                                 oversampling_factor_,
                                                                 size_t(iterations), 5.5, ConvolutionType::ATOMIC));

                    //float scale_factor = float(image_dims_os_[0]*image_dims_os_[1]*image_dims_os_[2]) / asum((&t));
                    //t *= scale_factor;
                    dcw_vec.push_back(*(t.to_host()));
                }
                else
                {
                    //auto temp = dcw_vec.at(ii-1);
                    dcw_vec.push_back(dcw_vec.at(ii - 1));
                }
            }
            return dcw_vec;
        }

        hoNDArray<float> estimateDCF(hoNDArray<floatd3> trajectories, hoNDArray<float> dcf, bool useIterativeDCWEstimated, float oversampling_factor_, size_t iterations,
                                     std::vector<size_t> image_dims_, bool fullySampled)
        {
            GadgetronTimer timer("DCF : ");

            boost::shared_ptr<cuNDArray<float>> dcw;
            boost::shared_ptr<cuNDArray<floatd3>> traj;
            boost::shared_ptr<cuNDArray<float>> tdcf;

            // std::vector<hoNDArray<float>> dcw_vec;

            cudaSetDevice(lit_sgncr_toolbox::utils::selectCudaDevice());

            // auto temp = lit_sgncr_toolbox::utils::concat<floatd3>(trajectories);
            // auto temp_dcf = lit_sgncr_toolbox::utils::concat<float>(dcf);

            tdcf = boost::make_shared<cuNDArray<float>>(dcf);
            traj = boost::make_shared<cuNDArray<floatd3>>(trajectories);
            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
            traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);

            auto t = *(Gadgetron::estimate_dcw<float, 3>((*traj),
                                                         *tdcf,
                                                         from_std_vector<size_t, 3>(image_dims_),
                                                         oversampling_factor_,
                                                         size_t(iterations), 5.5, ConvolutionType::ATOMIC));

            //float scale_factor = float(image_dims_os_[0]*image_dims_os_[1]*image_dims_os_[2]) / asum((&t));
            //t *= scale_factor;
            //dcw_vec.push_back(*(t.to_host()));
            auto output = *(t.to_host());
            return output;
        }

        template <template <class> class ARRAY>
        void
        set_data(ARRAY<float_complext> &sp_data, ARRAY<floatd3> &sp_traj, ARRAY<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                 boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, int currDev, bool useIterativeDCW, bool useIterativeDCWEstimated)
        {
            cudaPointerAttributes attributes;
            cudaPointerGetAttributes(&attributes, sp_data.get_data_ptr());

            cuData = boost::make_shared<cuNDArray<float_complext>>((sp_data));
            cuData.get()->squeeze();

            traj = boost::make_shared<cuNDArray<floatd3>>((sp_traj));

            std::vector<size_t> flat_dims = {traj->get_number_of_elements()};
            cuNDArray<floatd3> flat_traj(flat_dims, traj->get_data_ptr());
            traj = boost::make_shared<cuNDArray<floatd3>>(flat_traj);
            dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            cuNDArray<float> flat_dcw(flat_dims, dcw.get()->get_data_ptr());
            dcw = boost::make_shared<cuNDArray<float>>(flat_dcw);
            // if (~useIterativeDCW)
            // {
            //     dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            // }
            // if (useIterativeDCWEstimated && useIterativeDCW)
            // {
            //     dcw = boost::make_shared<cuNDArray<float>>((sp_dcw));
            //     cuNDArray<float> flat_dcw(flat_dims, dcw->get_data_ptr());
            //     dcw = boost::make_shared<cuNDArray<float>>(flat_dcw);
            // }

            // cuNDArray<float> flat_dcw(flat_dims, dcw.get_data_ptr());
            // dcw = cuNDArray<float>(flat_dcw);
        }

        arma::fmat33 lookup_PCS_DCS(std::string PCS_description)
        {
            arma::fmat33 A;

            if (PCS_description == "HFP")
            {
                A(0, 0) = 0;
                A(0, 1) = 1;
                A(0, 2) = 0;
                A(1, 0) = -1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFS")
            {
                A(0, 0) = 0;
                A(0, 1) = -1;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFDR")
            {
                A(0, 0) = 1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = 1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = -1;
            }
            if (PCS_description == "HFDL")
            {
                A(0, 0) = -1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFP")
            {
                A(0, 0) = 0;
                A(0, 1) = 1;
                A(0, 2) = 0;
                A(1, 0) = 1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFS")
            {
                A(0, 0) = 0;
                A(0, 1) = -1;
                A(0, 2) = 0;
                A(1, 0) = -1;
                A(1, 1) = 0;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFDR")
            {
                A(0, 0) = 1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = -1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }
            if (PCS_description == "FFDL")
            {
                A(0, 0) = -1;
                A(0, 1) = 0;
                A(0, 2) = 0;
                A(1, 0) = 0;
                A(1, 1) = 1;
                A(1, 2) = 0;
                A(2, 0) = 0;
                A(2, 1) = 0;
                A(2, 2) = 1;
            }

            return A;
        } // namespace corrections

        template void set_data(cuNDArray<float_complext> &sp_data, cuNDArray<floatd3> &sp_traj, cuNDArray<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                               boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, int currDev, bool useIterativeDCW, bool useIterativeDCWEstimated);
        template void set_data(hoNDArray<float_complext> &sp_data, hoNDArray<floatd3> &sp_traj, hoNDArray<float> &sp_dcw, std::vector<ISMRMRD::AcquisitionHeader> &sp_headers,
                               boost::shared_ptr<cuNDArray<float_complext>> &cuData, boost::shared_ptr<cuNDArray<floatd3>> &traj, boost::shared_ptr<cuNDArray<float>> &dcw, int currDev, bool useIterativeDCW, bool useIterativeDCWEstimated);

        template void write_gpu_nd_array(cuNDArray<float> &data, std::string filename);
        template void write_gpu_nd_array(cuNDArray<float_complext> &data, std::string filename);
        template void write_gpu_nd_array(cuNDArray<floatd3> &data, std::string filename);

        template void write_cpu_nd_array(hoNDArray<float> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<float_complext> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<floatd3> &data, std::string filename);
        template void write_cpu_nd_array(hoNDArray<complex_float_t> &data, std::string filename);

        template cuNDArray<float_complext> concat(std::vector<cuNDArray<float_complext>> &arrays);
        template cuNDArray<floatd3> concat(std::vector<cuNDArray<floatd3>> &arrays);
        template cuNDArray<floatd2> concat(std::vector<cuNDArray<floatd2>> &arrays);
        template cuNDArray<float> concat(std::vector<cuNDArray<float>> &arrays);

        template hoNDArray<float_complext> concat(std::vector<hoNDArray<float_complext>> &arrays);
        template hoNDArray<floatd3> concat(std::vector<hoNDArray<floatd3>> &arrays);
        template hoNDArray<floatd2> concat(std::vector<hoNDArray<floatd2>> &arrays);
        template hoNDArray<float> concat(std::vector<hoNDArray<float>> &arrays);

        template std::vector<cuNDArray<float>> sliceVec(std::vector<cuNDArray<float>> &v, int start, int end, int stride);
        template std::vector<cuNDArray<floatd2>> sliceVec(std::vector<cuNDArray<floatd2>> &v, int start, int end, int stride);
        template std::vector<cuNDArray<float_complext>> sliceVec(std::vector<cuNDArray<float_complext>> &v, int start, int end, int stride);
        template std::vector<ISMRMRD::AcquisitionHeader> sliceVec(std::vector<ISMRMRD::AcquisitionHeader> &v, int start, int end, int stride);

    } // namespace utils
} // namespace lit_sgncr_toolbox
