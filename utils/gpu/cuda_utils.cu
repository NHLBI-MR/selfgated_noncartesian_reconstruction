
#include "cuda_utils.h"
#include <gadgetron/cuNDArray_elemwise.h>
#include <gadgetron/cuNDArray_operators.h>
#include <gadgetron/cuNDArray_blas.h>
#include <gadgetron/cuNDArray_math.h>
#include <gadgetron/complext.h>

#include <complex>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/complex.h>
#include <math.h>
namespace lit_sgncr_toolbox
{
  namespace cuda_utils
  {
    using namespace Gadgetron;
    template<typename T> struct cuNDA_exp : public thrust::unary_function<T,T>
    {
      __device__ T operator()(const T &x) const {return exp(x);}
    };

    template<typename T> struct cuNDA_cos : public thrust::unary_function<T,T>
    {
      __device__  T operator()(const T &x) const {return cos(x);}
    };
    
    template<typename T> struct cuNDA_sin: public thrust::unary_function<T,T>
    {
      __device__  T operator()(const T &x) const {return sin(x);}
    };

  template<typename T> cuNDArray<T> cuexp( cuNDArray<T> &x )
  {  
    cuNDArray<T> results(x.get_dimensions());
    thrust::transform(x.begin(),x.end(),results.begin(),cuNDA_exp<T>());
    return results;
  }
  
template<typename T> cuNDArray<T> cucos( cuNDArray<T> &x )
{ 
  cuNDArray<T> results(x.get_dimensions());
  thrust::transform(x.begin(),x.end(),results.begin(),cuNDA_cos<T>());
  return results;

}  

template<typename T> cuNDArray<T> cusin( cuNDArray<T> &x )
{ 
  cuNDArray<T> results(x.get_dimensions());
  thrust::transform(x.begin(),x.end(),results.begin(),cuNDA_sin<T>());
  return results;

}  

  }
}
using namespace Gadgetron;

template Gadgetron::cuNDArray<float_complext>  lit_sgncr_toolbox::cuda_utils::cuexp<float_complext>( Gadgetron::cuNDArray<float_complext>& );
template Gadgetron::cuNDArray<float>           lit_sgncr_toolbox::cuda_utils::cucos<float>( Gadgetron::cuNDArray<float>& );
template Gadgetron::cuNDArray<float>           lit_sgncr_toolbox::cuda_utils::cusin<float>( Gadgetron::cuNDArray<float>& );