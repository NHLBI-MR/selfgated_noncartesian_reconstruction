/*
 * window_filter.cpp
 *
 *  Created on: March 1st, 2021
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
#include <math.h>
#include <stdio.h>
#include <boost/algorithm/string.hpp>
#include "util_functions.h"
#include "armadillo"

using namespace Gadgetron;
using namespace Gadgetron::Core;

namespace lit_sgncr_toolbox
{

    class kaiserFilter
    {
    public:
        enum class filterType
        {
            highPass,
            bandPass
        };

        kaiserFilter() = default;
        kaiserFilter(arma::vec bands, arma::vec errors, float Fs, filterType ftype);

        hoNDArray<std::complex<float>> getWindow();
        hoNDArray<std::complex<float>> getFilter();
        void filterData(hoNDArray<std::complex<float>> &data);

        void generateFilter();

    private:
        hoNDArray<std::complex<float>> generateWindow(arma::vec bands, arma::vec errors, filterType ftype, float Fs);
        hoNDArray<std::complex<float>> generateRect(arma::vec bands, filterType ftype, float Fs);
        hoNDArray<std::complex<float>> sincInterpolation( hoNDArray<std::complex<float>> input, size_t output_size);

        hoNDArray<std::complex<float>> kaiserWindow(float beta, size_t M);
        float I0(float n, float x);
        float factorial(float n);

        float transitionBand;
        float alpha;
        float beta;
        arma::vec bands;
        arma::vec errors;
        float Fs;
        filterType ftype;
        size_t filterLength;
        size_t M; // Filter Order
        hoNDArray<std::complex<float>> window;
        hoNDArray<std::complex<float>> windowedFilter;
    };
}