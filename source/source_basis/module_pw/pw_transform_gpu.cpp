#include "source_base/timer.h"
#include "source_basis/module_pw/kernels/pw_op.h"
#include "pw_basis.h"
namespace ModulePW
{
#if (defined(__CUDA) || defined(__ROCM))
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const FPTYPE* in, std::complex<FPTYPE>* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    const size_t size = this->nrxx;
    base_device::memory::cast_memory_op<std::complex<FPTYPE>, FPTYPE,base_device::DEVICE_GPU, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        in,
        size);

    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}
template <typename FPTYPE>
void PW_Basis::real2recip_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
    assert(this->poolnproc == 1);
    base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                               base_device::DEVICE_GPU,
                                               base_device::DEVICE_GPU>()(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                          in,
                                                                          this->nrxx);
    this->fft_bundle.fft3D_forward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                   this->nxyz,
                                                                   add,
                                                                   factor,
                                                                   this->ig2ixyz_gpu,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);
    ModuleBase::timer::tick(this->classname, "real_to_recip gpu");
}

template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in, FPTYPE* out, const bool add, const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<FPTYPE>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);
    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}
template <typename FPTYPE>
void PW_Basis::recip2real_gpu(const std::complex<FPTYPE>* in,
                              std::complex<FPTYPE>* out,
                              const bool add,
                              const FPTYPE factor) const
{
    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
    assert(this->poolnproc == 1);
    // ModuleBase::GlobalFunc::ZEROS(fft_bundle.get_auxr_3d_data<double>(), this->nxyz);
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(
        this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
        0,
        this->nxyz);

    set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                         this->ig2ixyz_gpu,
                                                         in,
                                                         this->fft_bundle.get_auxr_3d_data<FPTYPE>());
    this->fft_bundle.fft3D_backward(this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                    this->fft_bundle.get_auxr_3d_data<FPTYPE>());

    set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(this->nrxx,
                                                                   add,
                                                                   factor,
                                                                   this->fft_bundle.get_auxr_3d_data<FPTYPE>(),
                                                                   out);

    ModuleBase::timer::tick(this->classname, "recip_to_real gpu");
}

template <typename FPTYPE>
void PW_Basis::real2recip_batched_gpu(const std::complex<FPTYPE>* in,
                                      std::complex<FPTYPE>* out,
                                      const int batch_size,
                                      const bool add,
                                      const FPTYPE factor) const
{
    if (batch_size <= 1)
    {
        this->real2recip_gpu(in, out, add, factor);
        return;
    }

    ModuleBase::timer::tick(this->classname, "real_to_recip batched gpu");
    assert(this->poolnproc == 1);

    const int nxyz = this->nxyz;
    const int nrxx = this->nrxx;
    const int npw = this->npw;
    const size_t total = static_cast<size_t>(batch_size) * nxyz;

    std::complex<FPTYPE>* auxr_batched = nullptr;
    if constexpr (std::is_same<FPTYPE, double>::value)
    {
        if (this->batched_auxr_double_size_ < total)
        {
            base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->batched_auxr_double_);
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->batched_auxr_double_, total);
            this->batched_auxr_double_size_ = total;
        }
        auxr_batched = reinterpret_cast<std::complex<FPTYPE>*>(this->batched_auxr_double_);
    }
    else
    {
        if (this->batched_auxr_float_size_ < total)
        {
            base_device::memory::delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(this->batched_auxr_float_);
            base_device::memory::resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(this->batched_auxr_float_, total);
            this->batched_auxr_float_size_ = total;
        }
        auxr_batched = reinterpret_cast<std::complex<FPTYPE>*>(this->batched_auxr_float_);
    }

    for (int ib = 0; ib < batch_size; ++ib)
    {
        base_device::memory::synchronize_memory_op<std::complex<FPTYPE>,
                                                   base_device::DEVICE_GPU,
                                                   base_device::DEVICE_GPU>()(auxr_batched + static_cast<size_t>(ib) * nxyz,
                                                                              in + static_cast<size_t>(ib) * nrxx,
                                                                              nrxx);
    }

    this->fft_bundle.fft3D_forward_batched(auxr_batched, auxr_batched, batch_size);

    for (int ib = 0; ib < batch_size; ++ib)
    {
        set_real_to_recip_output_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                                        nxyz,
                                                                        add,
                                                                        factor,
                                                                        this->ig2ixyz_gpu,
                                                                        auxr_batched + static_cast<size_t>(ib) * nxyz,
                                                                        out + static_cast<size_t>(ib) * npw);
    }

    ModuleBase::timer::tick(this->classname, "real_to_recip batched gpu");
}

template <typename FPTYPE>
void PW_Basis::recip2real_batched_gpu(const std::complex<FPTYPE>* in,
                                      std::complex<FPTYPE>* out,
                                      const int batch_size,
                                      const bool add,
                                      const FPTYPE factor) const
{
    if (batch_size <= 1)
    {
        this->recip2real_gpu(in, out, add, factor);
        return;
    }

    ModuleBase::timer::tick(this->classname, "recip_to_real batched gpu");
    assert(this->poolnproc == 1);

    const int nxyz = this->nxyz;
    const int nrxx = this->nrxx;
    const int npw = this->npw;
    const size_t total = static_cast<size_t>(batch_size) * nxyz;

    std::complex<FPTYPE>* auxr_batched = nullptr;
    if constexpr (std::is_same<FPTYPE, double>::value)
    {
        if (this->batched_auxr_double_size_ < total)
        {
            base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->batched_auxr_double_);
            base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(this->batched_auxr_double_, total);
            this->batched_auxr_double_size_ = total;
        }
        auxr_batched = reinterpret_cast<std::complex<FPTYPE>*>(this->batched_auxr_double_);
    }
    else
    {
        if (this->batched_auxr_float_size_ < total)
        {
            base_device::memory::delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(this->batched_auxr_float_);
            base_device::memory::resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(this->batched_auxr_float_, total);
            this->batched_auxr_float_size_ = total;
        }
        auxr_batched = reinterpret_cast<std::complex<FPTYPE>*>(this->batched_auxr_float_);
    }
    base_device::memory::set_memory_op<std::complex<FPTYPE>, base_device::DEVICE_GPU>()(auxr_batched, 0, total);

    for (int ib = 0; ib < batch_size; ++ib)
    {
        set_3d_fft_box_op<FPTYPE, base_device::DEVICE_GPU>()(npw,
                                                              this->ig2ixyz_gpu,
                                                              in + static_cast<size_t>(ib) * npw,
                                                              auxr_batched + static_cast<size_t>(ib) * nxyz);
    }

    this->fft_bundle.fft3D_backward_batched(auxr_batched, auxr_batched, batch_size);

    for (int ib = 0; ib < batch_size; ++ib)
    {
        set_recip_to_real_output_op<FPTYPE, base_device::DEVICE_GPU>()(nrxx,
                                                                        add,
                                                                        factor,
                                                                        auxr_batched + static_cast<size_t>(ib) * nxyz,
                                                                        out + static_cast<size_t>(ib) * nrxx);
    }

    ModuleBase::timer::tick(this->classname, "recip_to_real batched gpu");
}
template void PW_Basis::real2recip_gpu<double>(const double* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const float* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::real2recip_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::real2recip_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               double* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              float* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::recip2real_gpu<double>(const std::complex<double>* in,
                                               std::complex<double>* out,
                                               const bool add,
                                               const double factor) const;
template void PW_Basis::recip2real_gpu<float>(const std::complex<float>* in,
                                              std::complex<float>* out,
                                              const bool add,
                                              const float factor) const;

template void PW_Basis::real2recip_batched_gpu<double>(const std::complex<double>* in,
                                                       std::complex<double>* out,
                                                       const int batch_size,
                                                       const bool add,
                                                       const double factor) const;
template void PW_Basis::real2recip_batched_gpu<float>(const std::complex<float>* in,
                                                      std::complex<float>* out,
                                                      const int batch_size,
                                                      const bool add,
                                                      const float factor) const;

template void PW_Basis::recip2real_batched_gpu<double>(const std::complex<double>* in,
                                                       std::complex<double>* out,
                                                       const int batch_size,
                                                       const bool add,
                                                       const double factor) const;
template void PW_Basis::recip2real_batched_gpu<float>(const std::complex<float>* in,
                                                      std::complex<float>* out,
                                                      const int batch_size,
                                                      const bool add,
                                                      const float factor) const;

#endif
} // namespace ModulePW
