#include "module_hamilt_pw/hamilt_pwdft/kernels/vec_mul_vec_complex_op.h"
#include "module_parameter/parameter.h"
#include "module_psi/psi.h"
namespace hamilt{
template <typename FPTYPE>
struct vec_mul_vec_complex_op<std::complex<FPTYPE>, base_device::DEVICE_CPU>
{
    using T = std::complex<FPTYPE>;
    void operator()(const T *vec1, const T *vec2, T *out, int n)
    {
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < n; i++)
        {
            out[i] = vec1[i] * vec2[i];
        }
    }

};
template struct vec_mul_vec_complex_op<std::complex<float>, base_device::DEVICE_CPU>;
template struct vec_mul_vec_complex_op<std::complex<double>, base_device::DEVICE_CPU>;
} // hamilt