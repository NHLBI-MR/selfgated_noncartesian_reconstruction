/*
 * window_filter.cpp
 *
 *  Created on: March 1st, 2021
 *      Author: Ahsan Javed
 */

#include "window_filter.h"

namespace lit_sgncr_toolbox
{
    kaiserFilter::kaiserFilter(arma::vec bands, arma::vec errors, float Fs, filterType ftype)
    {
        this->bands = bands;
        this->errors = errors;
        this->Fs = Fs;
        this->ftype = ftype;
    }

    float kaiserFilter::factorial(float n)
    {
        float sum = 1;
        for (float ii = 1; ii <= n; ii++)
            sum = sum * n;

        return sum;
    }
    float kaiserFilter::I0(float n, float x)
    {
        float I0_x = 1.0;
        for (float ii = 1; ii <= n; ii++)
            I0_x = I0_x + pow((x / pow(2, ii)) / factorial(ii), 2);
        return I0_x;
    }
    // function I0_x = I0(n,x)

    // I0_x = 1.0 ;
    // for ii=1:n
    //     I0_x = I0_x+ ((x/2^ii)/ n_jiecheng(ii))^2;
    // end
    // end

    // function sum = n_jiecheng(n)
    // sum = 1;
    // for ii=1:n
    //     sum = sum * n;
    // end
    // end
    hoNDArray<std::complex<float>> kaiserFilter::kaiserWindow(float beta, size_t length)
    {
        hoNDArray<std::complex<float>> window(length);

        for (float ii = 0; ii < length; ii++)
        {
            window(ii) = I0(20, beta * sqrt(1 - pow((2.0 * (ii + 1) / (float(length))) - 1, 2))) / I0(20, beta);
        }
        return window;
    }
    hoNDArray<std::complex<float>> kaiserFilter::generateRect(arma::vec bands, filterType ftype, float Fs)
    {
        hoNDArray<std::complex<float>> rect(this->filterLength);
        switch (ftype)
        {
        case filterType::highPass:
        {
            auto stIn = std::ceil(filterLength / 2 + filterLength * bands[0] / Fs);
            auto endIn = double(filterLength);

            for (auto ii = stIn; ii < endIn; ii++)
                rect(ii) = 1;

            stIn = ceil(filterLength / 2 - filterLength * bands[0] / Fs);
            endIn = 0;
            for (auto ii = std::min(stIn, endIn); ii < std::max(stIn, endIn); ii++)
                rect(ii) = 1;

            break;
        }
        case filterType::bandPass:
        {
            auto stIn = std::ceil(filterLength / 2 + filterLength * bands[1] / Fs);
            auto endIn = std::ceil(filterLength / 2 + filterLength * bands[2] / Fs);

            for (auto ii = stIn; ii < endIn; ii++)
                rect(ii) = 1;

            stIn = std::ceil(filterLength / 2 - filterLength * bands[1] / Fs);
            endIn = std::ceil(filterLength / 2 - filterLength * bands[2] / Fs);
            for (auto ii = std::min(stIn, endIn); ii < std::max(stIn, endIn); ii++)
                rect(ii) = 1;

            break;
        }
        }
        return rect;
    }
    hoNDArray<std::complex<float>> kaiserFilter::generateWindow(arma::vec bands, arma::vec errors, filterType ftype, float Fs)
    {
        switch (ftype)
        {
        case filterType::highPass:
            transitionBand = arma::min(bands);
            break;

        case filterType::bandPass:
            transitionBand = arma::min(arma::diff(bands));
            break;
        }
        alpha = 20 * std::abs(std::log10(arma::min(errors)));

        if (alpha > 50)
        {
            beta = 0.1102 * (alpha - 8.7);
        }
        else if (alpha >= 26 && alpha <= 50)
        {
            beta = 0.1102 * (alpha - 8.7);
        }
        else
        {
            beta = 0.1102 * (alpha - 8.7);
        }

        M = ceil((alpha - 7.95) / (2.285 * transitionBand));

        // if (filterLength % 2 == 1 && M % 2 == 0)
        if (M % 2 == 0)
            M = M + 1;

        this->filterLength = M;
        auto window = kaiserWindow(beta, M);

        return window;
    }

    hoNDArray<std::complex<float>> kaiserFilter::sincInterpolation(hoNDArray<std::complex<float>> input, size_t output_size)
    {
        hoNDArray<std::complex<float>> coutput(output_size);

        std::fill(coutput.begin(), coutput.end(), 0);
        hoNDFFT<float>::instance()->fft1c(input);

#pragma omp parallel
#pragma omp for
        for (int ii = 0; ii < output_size; ii++)
        {
            if (ii > output_size / 2 - input.get_size(0) / 2 - 1 && ii < output_size / 2 + (input.get_size(0) / 2))
            {
                coutput(ii) = input(ii - (output_size / 2 - input.get_size(0) / 2));
            }
        }

        hoNDFFT<float>::instance()->ifft1c(coutput);
        coutput *= sqrt(output_size / input.get_size(0));
        hoNDFFT<float>::instance()->ifft1c(input);
        // output *= sqrt(zpadFactor);
        return coutput;
    }

    // Add function for zero padding and applying the filter to the data.
    void kaiserFilter::filterData(hoNDArray<std::complex<float>> &data)
    {
        auto E0 = data.get_size(0);
        auto E1 = data.get_size(1);
        using namespace Gadgetron::Indexing;
        auto windowFilter = this->windowedFilter;
        if (windowFilter.get_size(0) < E0)
        {
            hoNDArray<std::complex<float>> paddedFilter({E0, E1}); // dimension to zero pad should be the first one
#pragma omp parallel
#pragma omp for
            for (auto jj = 0; jj < E1; jj++)
                paddedFilter(slice, jj) = sincInterpolation(windowFilter, E0);

            hoNDFFT<float>::instance()->fft1c(data);
            multiply(&data, &paddedFilter, &data);
            hoNDFFT<float>::instance()->ifft1c(data);
            // hoNDArray<std::complex<float>> paddedFilter({E0, E1}); // dimension to zero pad should be the first one
            // auto window = this->windowedFilter;

            //  hoNDFFT<float>::instance()->fft1c(window);

            // for (int jj = 0; jj < E1; jj++)
            // {
            //     for (int ii = 0; ii < E0; ii++)
            //     {
            //         if (ii > E0 / 2 - window.size() / 2 - 1 && ii < E0/ 2 + (window.size() / 2))
            //         {
            //             paddedFilter(ii,jj) = window(ii - (E0 / 2 - E0 / 2));
            //         }
            //     }
            // }
            // hoNDFFT<float>::instance()->ifft1c(paddedFilter);
        }
        else if (windowFilter.get_size(0) >= E0)
        {
            hoNDArray<std::complex<float>> paddedData({windowFilter.get_size(0), E1}); // dimension to zero pad should be the first one

            hoNDFFT<float>::instance()->fft1c(data);

#pragma omp parallel
#pragma omp for
            for (auto jj = 0; jj < E1; jj++)
                paddedData(slice, jj) = sincInterpolation(data, windowFilter.get_size(0));

            // lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(data, "/opt/data/gt_data/data2.complex");
            //  lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(paddedData, "/opt/data/gt_data/paddedData2.complex");
            //  lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(windowFilter, "/opt/data/gt_data/windowFilter2.complex");

            multiply(&paddedData, &windowFilter, &paddedData);
            hoNDFFT<float>::instance()->ifft1c(paddedData);

            //  lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(paddedData, "/opt/data/gt_data/paddedData22.complex");

            // double check may be off by one sample
            data = crop<std::complex<float>, 2>(vector_td<size_t, 2>((paddedData.get_size(0) - E0) / 2, 0),
                                                vector_td<size_t, 2>(E0, E1),
                                                paddedData);
            // lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(data, "/opt/data/gt_data/data22.complex");
        }

        // return data;
    }
    void kaiserFilter::generateFilter()
    {
        auto win = generateWindow(this->bands, this->errors, this->ftype, this->Fs);
        auto rect = generateRect(this->bands, this->ftype, this->Fs);
        // lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(win, "/opt/data/gt_data/win.complex");
        // lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(rect, "/opt/data/gt_data/rect.complex");

        hoNDFFT<float>::instance()->ifft1c(rect);
        //lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(rect, "/opt/data/gt_data/rectfft.complex");

        multiply(win, rect, win);
        //lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(win, "/opt/data/gt_data/winrect.complex");

        hoNDFFT<float>::instance()->fft1c(win);
        //lit_sgncr_toolbox::utils::write_cpu_nd_array<std::complex<float>>(win, "/opt/data/gt_data/winfftc.complex");

        this->windowedFilter = win;
    }

} // namespace lit_sgncr_toolbox