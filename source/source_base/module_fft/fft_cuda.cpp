#include "fft_cuda.h"
#include "source_base/module_device/memory_op.h"
#include "source_base/module_device/device_check.h"

namespace ModuleBase
{
template <typename FPTYPE>
void FFT_CUDA<FPTYPE>::initfft(int nx_in, int ny_in, int nz_in)
{
    this->nx = nx_in;
    this->ny = ny_in;
    this->nz = nz_in;
}
template <>
void FFT_CUDA<float>::setupFFT()
{
    cufftPlan3d(&c_handle, this->nx, this->ny, this->nz, CUFFT_C2C);
    resmem_cd_op()(this->c_auxr_3d, this->nx * this->ny * this->nz);
}
template <>
void FFT_CUDA<double>::setupFFT()
{
    cufftPlan3d(&z_handle, this->nx, this->ny, this->nz, CUFFT_Z2Z);
    resmem_zd_op()(this->z_auxr_3d, this->nx * this->ny * this->nz);
}
template <>
void FFT_CUDA<float>::cleanFFT()
{
    if (c_handle)
    {
        cufftDestroy(c_handle);
        c_handle = {};
    }
}
template <>
void FFT_CUDA<double>::cleanFFT()
{
    if (z_handle)
    {
        cufftDestroy(z_handle);
        z_handle = {};
    }
}
template <>
void FFT_CUDA<float>::clear()
{
    this->cleanFFT();
    if (c_handle_batch)
    {
        cufftDestroy(c_handle_batch);
        c_handle_batch = {};
    }
    batch_initialized = false;
    planned_batch_size = 0;
    if (c_auxr_3d != nullptr)
    {
        delmem_cd_op()(c_auxr_3d);
        c_auxr_3d = nullptr;
    }
}
template <>
void FFT_CUDA<double>::clear()
{
    this->cleanFFT();
    if (z_handle_batch)
    {
        cufftDestroy(z_handle_batch);
        z_handle_batch = {};
    }
    batch_initialized = false;
    planned_batch_size = 0;
    if (z_auxr_3d != nullptr)
    {
        delmem_zd_op()(z_auxr_3d);
        z_auxr_3d = nullptr;
    }
}

template <>
void FFT_CUDA<float>::fft3D_forward(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<double>::fft3D_forward(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_FORWARD));
}
template <>
void FFT_CUDA<float>::fft3D_backward(std::complex<float>* in, std::complex<float>* out) const
{
    CHECK_CUFFT(cufftExecC2C(this->c_handle,
                             reinterpret_cast<cufftComplex*>(in),
                             reinterpret_cast<cufftComplex*>(out),
                             CUFFT_INVERSE));
}

template <>
void FFT_CUDA<double>::fft3D_backward(std::complex<double>* in, std::complex<double>* out) const
{
    CHECK_CUFFT(cufftExecZ2Z(this->z_handle,
                             reinterpret_cast<cufftDoubleComplex*>(in),
                             reinterpret_cast<cufftDoubleComplex*>(out),
                             CUFFT_INVERSE));
}
template <>
std::complex<float>* FFT_CUDA<float>::get_auxr_3d_data() const
{
    return this->c_auxr_3d;
}
template <>
std::complex<double>* FFT_CUDA<double>::get_auxr_3d_data() const
{
    return this->z_auxr_3d;
}

template <>
void FFT_CUDA<float>::setupFFT_batched(int batch_size)
{
    if (batch_size <= 1) {
        return;
    }

    if (batch_initialized && planned_batch_size == batch_size) {
        return;
    }

    if (c_handle_batch) {
        cufftDestroy(c_handle_batch);
        c_handle_batch = {};
    }
    batch_initialized = false;
    planned_batch_size = 0;

    int n[3] = {this->nx, this->ny, this->nz};
    const int dist = this->nx * this->ny * this->nz;

    CHECK_CUFFT(cufftPlanMany(&c_handle_batch,
                              3,
                              n,
                              nullptr,
                              1,
                              dist,
                              nullptr,
                              1,
                              dist,
                              CUFFT_C2C,
                              batch_size));

    planned_batch_size = batch_size;
    batch_initialized = true;
}

template <>
void FFT_CUDA<double>::setupFFT_batched(int batch_size)
{
    if (batch_size <= 1) {
        return;
    }

    if (batch_initialized && planned_batch_size == batch_size) {
        return;
    }

    if (z_handle_batch) {
        cufftDestroy(z_handle_batch);
        z_handle_batch = {};
    }
    batch_initialized = false;
    planned_batch_size = 0;

    int n[3] = {this->nx, this->ny, this->nz};
    const int dist = this->nx * this->ny * this->nz;

    CHECK_CUFFT(cufftPlanMany(&z_handle_batch,
                              3,
                              n,
                              nullptr,
                              1,
                              dist,
                              nullptr,
                              1,
                              dist,
                              CUFFT_Z2Z,
                              batch_size));

    planned_batch_size = batch_size;
    batch_initialized = true;
}

template <>
void FFT_CUDA<float>::cleanFFT_batched()
{
    if (c_handle_batch) {
        cufftDestroy(c_handle_batch);
        c_handle_batch = {};
        batch_initialized = false;
        planned_batch_size = 0;
    }
}

template <>
void FFT_CUDA<double>::cleanFFT_batched()
{
    if (z_handle_batch) {
        cufftDestroy(z_handle_batch);
        z_handle_batch = {};
        batch_initialized = false;
        planned_batch_size = 0;
    }
}

template <>
void FFT_CUDA<float>::fft3D_forward_batched(std::complex<float>* in,
                                            std::complex<float>* out,
                                            int batch_size) const
{
    if (batch_size > 1 && c_handle_batch && batch_initialized) {
        CHECK_CUFFT(cufftExecC2C(c_handle_batch,
                                 reinterpret_cast<cufftComplex*>(in),
                                 reinterpret_cast<cufftComplex*>(out),
                                 CUFFT_FORWARD));
    } else {
        for (int b = 0; b < batch_size; b++) {
            fft3D_forward(in + b * this->nx * this->ny * this->nz,
                         out + b * this->nx * this->ny * this->nz);
        }
    }
}

template <>
void FFT_CUDA<double>::fft3D_forward_batched(std::complex<double>* in,
                                             std::complex<double>* out,
                                             int batch_size) const
{
    if (batch_size > 1 && z_handle_batch && batch_initialized) {
        CHECK_CUFFT(cufftExecZ2Z(z_handle_batch,
                                 reinterpret_cast<cufftDoubleComplex*>(in),
                                 reinterpret_cast<cufftDoubleComplex*>(out),
                                 CUFFT_FORWARD));
    } else {
        for (int b = 0; b < batch_size; b++) {
            fft3D_forward(in + b * this->nx * this->ny * this->nz,
                         out + b * this->nx * this->ny * this->nz);
        }
    }
}

template <>
void FFT_CUDA<float>::fft3D_backward_batched(std::complex<float>* in,
                                             std::complex<float>* out,
                                             int batch_size) const
{
    if (batch_size > 1 && c_handle_batch && batch_initialized) {
        CHECK_CUFFT(cufftExecC2C(c_handle_batch,
                                 reinterpret_cast<cufftComplex*>(in),
                                 reinterpret_cast<cufftComplex*>(out),
                                 CUFFT_INVERSE));
    } else {
        for (int b = 0; b < batch_size; b++) {
            fft3D_backward(in + b * this->nx * this->ny * this->nz,
                          out + b * this->nx * this->ny * this->nz);
        }
    }
}

template <>
void FFT_CUDA<double>::fft3D_backward_batched(std::complex<double>* in,
                                              std::complex<double>* out,
                                              int batch_size) const
{
    if (batch_size > 1 && z_handle_batch && batch_initialized) {
        CHECK_CUFFT(cufftExecZ2Z(z_handle_batch,
                                 reinterpret_cast<cufftDoubleComplex*>(in),
                                 reinterpret_cast<cufftDoubleComplex*>(out),
                                 CUFFT_INVERSE));
    } else {
        for (int b = 0; b < batch_size; b++) {
            fft3D_backward(in + b * this->nx * this->ny * this->nz,
                          out + b * this->nx * this->ny * this->nz);
        }
    }
}

template FFT_CUDA<float>::FFT_CUDA();
template FFT_CUDA<float>::~FFT_CUDA();
template FFT_CUDA<double>::FFT_CUDA();
template FFT_CUDA<double>::~FFT_CUDA();
} // namespace ModuleBase
