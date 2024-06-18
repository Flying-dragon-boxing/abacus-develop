#include "module_base/global_variable.h"
#include "module_cell/klist.h"
#include "module_elecstate/elecstate_getters.h"
#include "module_hamilt_general/operator.h"
#include "module_psi/psi.h"
#include "module_base/tool_quit.h"

#include <cassert>
#include <complex>
#include <cstdlib>
#include <memory>

#ifdef __EXX
#define __EXX_PW
#endif

#define __EXX_PW

#ifdef __EXX_PW
#include "op_exx_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace hamilt
{
// init update_psi
template <typename T, typename Device>
bool OperatorEXXPW<T, Device>::update_psi = true;

template <typename T, typename Device>
OperatorEXXPW<T, Device>::OperatorEXXPW(const int* isk_in,
                                        const ModulePW::PW_Basis_K* wfcpw_in,
                                        const ModulePW::PW_Basis* rhopw_in,
                                        K_Vectors *kv_in)
    : isk(isk_in), wfcpw(wfcpw_in), rhopw(rhopw_in), kv(kv_in)
{

    if (GlobalV::KPAR != 1)
    {
        // GlobalV::ofs_running << "EXX Calculation does not support k-point parallelism" << std::endl;
        ModuleBase::WARNING_QUIT("OperatorEXXPW", "EXX Calculation does not support k-point parallelism");
    }

    this->classname = "OperatorEXXPW";
    this->ctx = nullptr;
    this->cpu_ctx = nullptr;
    this->cal_type = pw_exx;

    // allocate real space memory
    assert(wfcpw->nrxx == rhopw->nrxx);
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw->nrxx);
    // allocate density recip space memory
    resmem_complex_op()(this->ctx, density_recip, rhopw->npw);
    // allocate h_psi recip space memory
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);

    update_psi = true;
}

template <typename T, typename Device>
OperatorEXXPW<T, Device>::~OperatorEXXPW()
{
    // use delete_memory_op to delete the allocated pws
    delmem_complex_op()(this->ctx, psi_nk_real);
    delmem_complex_op()(this->ctx, psi_mq_real);
    delmem_complex_op()(this->ctx, density_real);
    delmem_complex_op()(this->ctx, h_psi_real);
    delmem_complex_op()(this->ctx, density_recip);
    delmem_complex_op()(this->ctx, h_psi_recip);
}

template <typename T>
inline bool is_finite(const T &val)
{
    return std::isfinite(val);
}

template <>
inline bool is_finite(const std::complex<float> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <>
inline bool is_finite(const std::complex<double> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik) const
{
    if (GlobalC::exx_helper.first_iter)
    {
        // if (this->ik == wfcpw->nks - 1)
        //     first_called = false;
        return;
    }
    // if (update_psi)
    {
        // allocate pws
        if (pws.empty())
        {
            for (int ik = 0; ik < wfcpw->nks; ik++)
            {
                pws.push_back(std::make_unique<T[]>(wfcpw->npwk[ik] * GlobalV::NBANDS));

            }
        }

        if (GlobalC::exx_helper.exx_energy.empty())
        {
            GlobalC::exx_helper.exx_energy.resize(wfcpw->nks);
        }
        
        Real max_error = 0;
        Real max = 0;
        
        for (int ik = 0; ik < wfcpw->nks; ik++)
        {
            T* pw_ik = pws[ik].get();
            
            psi->fix_k(ik);
            const T* psi_ik = psi->get_pointer();
            for (int ib = 0; ib < GlobalV::NBANDS; ib++)
            {
                #ifdef _OPENMP
                #pragma omp parallel for reduction(max : max_error, max)
                #endif
                for (int ig = 0; ig < wfcpw->npwk[ik]; ig++)
                {
                    max_error = std::max(max_error, std::abs(std::norm(psi_ik[ig + ib * wfcpw->npwk[ik]] - pw_ik[ig + ib * wfcpw->npwk[ik]])));
                    max = std::max(max, std::norm(psi_ik[ig + ib * wfcpw->npwk[ik]]));
                    pw_ik[ig + ib * wfcpw->npwk[ik]] = psi_ik[ig + ib * wfcpw->npwk[ik]];
                }
            }
        }

        psi->fix_k(this->ik);
        // update_psi = false;
    }


    setmem_complex_op()(this->ctx, h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(this->ctx, h_psi_real, 0, rhopw->nrxx);
    
    // ik fixed here, select band n
    for (int n_iband = 0; n_iband < nbands; n_iband++)
    {
        const T *psi_nk = tmpsi_in + n_iband * nbasis;
        // retrieve \psi_nk in real space
        wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);

        // for \psi_nk, get the pw of iq and band m
        auto q_points = get_q_points(this->ik);
        for (int iq: q_points)
        {
            for (int m_iband = 0; m_iband < GlobalV::NBANDS; m_iband++)
            {
                double wg_f = GlobalC::exx_helper.wg(iq, m_iband);
                T wg_iqb = wg_f;
                if (wg_f < 1e-12)
                {
                    continue;
                }
                const T* psi_mq = get_pw(m_iband, iq);
                wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                
                // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    assert(is_finite(psi_nk_real[ir]));
                    assert(is_finite(psi_mq_real[ir]));
                    density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]);
                }
                // to be changed into kernel function
                
                // bring the density to recip space
                rhopw->real2recip(density_real, density_recip);

                // multiply the density with the potential in recip space
                multiply_potential(density_recip, this->ik, iq);

                // bring the potential back to real space
                rhopw->recip2real(density_recip, density_real);

                // get the h|psi_ik>(r), save in density_real
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    assert(is_finite(psi_mq_real[ir]));
                    assert(is_finite(density_real[ir]));
                    density_real[ir] *= psi_mq_real[ir];
                }

                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    h_psi_real[ir] += density_real[ir] * wg_iqb;
                }

            }

        }
        
        wfcpw->real_to_recip(ctx, h_psi_real, h_psi_recip, this->ik);

        // add the h|psi_ik> to tmhpsi
        T *h_psi_nk = tmhpsi + n_iband * nbasis;
        T hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        T Eexx_ik = 0.0;
        T wg_ikb = GlobalC::exx_helper.wg(this->ik, n_iband);
        T ucell_nxyz = wfcpw->nxyz;
        #ifdef _OPENMP
        #pragma omp parallel for
        #endif
        for (int ig = 0; ig < wfcpw->npwk_max; ig++)
        {
            // assert no nan
            assert(is_finite(h_psi_recip[ig]));
            h_psi_nk[ig] += h_psi_recip[ig] * hybrid_alpha;
        }

        // #ifdef _OPENMP
        // #pragma omp parallel for reduction(+ : Eexx_ik)
        // #endif
        for (int ir = 0; ir < wfcpw->nrxx; ir++)
        {
            Eexx_ik += wg_ikb * h_psi_real[ir] * std::conj(psi_nk_real[ir]) / ucell_nxyz;
        }
        // to be changed into kernel function
        // assert(std::abs(std::imag(Eexx_ik)) < 1e-3);
        Eexx += std::real(Eexx_ik);
        
    }
    
    

    

    // evaluate the Eexx
    T Eexx_ik = 0.0;
    for (int ik = 0; ik < wfcpw->nks; ik++)
    {
        for (int n_iband = 0; n_iband < GlobalV::NBANDS; n_iband++)
        {
            setmem_complex_op()(this->ctx, h_psi_recip, 0, wfcpw->npwk_max);
            setmem_complex_op()(this->ctx, h_psi_real, 0, rhopw->nrxx);    

            const T *psi_nk = psi->get_pointer() + n_iband * wfcpw->npwk[ik];
            // retrieve \psi_nk in real space
            wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);

            // for \psi_nk, get the pw of iq and band m
            auto q_points = get_q_points(this->ik);
            for (int iq: q_points)
            {
                for (int m_iband = 0; m_iband < GlobalV::NBANDS; m_iband++)
                {
                    double wg_f = GlobalC::exx_helper.wg(iq, m_iband);
                    T wg_iqb = wg_f;
                    if (wg_f < 1e-12)
                    {
                        continue;
                    }
                    const T* psi_mq = psi->get_pointer() + m_iband * wfcpw->npwk[iq];
                    wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                    
                    // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int ir = 0; ir < wfcpw->nrxx; ir++)
                    {
                        assert(is_finite(psi_nk_real[ir]));
                        assert(is_finite(psi_mq_real[ir]));
                        density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]);
                    }
                    // to be changed into kernel function
                    
                    // bring the density to recip space
                    rhopw->real2recip(density_real, density_recip);

                    // multiply the density with the potential in recip space
                    multiply_potential(density_recip, this->ik, iq);

                    // bring the potential back to real space
                    rhopw->recip2real(density_recip, density_real);

                    // get the h|psi_ik>(r), save in density_real
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int ir = 0; ir < wfcpw->nrxx; ir++)
                    {
                        assert(is_finite(psi_mq_real[ir]));
                        assert(is_finite(density_real[ir]));
                        density_real[ir] *= psi_mq_real[ir];
                    }

                    for (int ir = 0; ir < wfcpw->nrxx; ir++)
                    {
                        h_psi_real[ir] += density_real[ir] * wg_iqb;
                    }

                }

            }

            T wg_ikb = GlobalC::exx_helper.wg(this->ik, n_iband);
            // T t_nxyz = wfcpw->nxyz;
            // T t_omega = elecstate::get_ucell_omega();
            
            // for (int ir = 0; ir < wfcpw->nrxx; ir++)
            // {
            //     Eexx_ik += wg_ikb * h_psi_real[ir] * std::conj(psi_nk_real[ir]) / t_nxyz;
            // }
            
            wfcpw->real_to_recip(ctx, h_psi_real, h_psi_recip, ik);

            for (int ig = 0; ig < wfcpw->npwk_max; ig++)
            {
                Eexx_ik += h_psi_recip[ig] * std::conj(psi_nk[ig]) * wg_ikb;
            }

        }
    }
    
    Real Eexx_ik_real = std::real(Eexx_ik);

    // if (Eexx != 0)
    GlobalC::exx_lip.set_exx_energy(Eexx_ik_real);
}

template <typename T, typename Device>
std::vector<int> OperatorEXXPW<T, Device>::get_q_points(const int ik) const
{
    // stored in q_points
    if (q_points.find(ik) != q_points.end())
    {
        return q_points.find(ik)->second;
    }

    std::vector<int> q_points_ik;

    // if () // downsampling
    {
        for (int iq = 0; iq < wfcpw->nks; iq++)
        {
            q_points_ik.push_back(iq);
        }
    }
    // else
    // {    
    //     for (int iq = 0; iq < wfcpw->nks; iq++)
    //     {
    //         kv->
    //     }
    // }

    q_points[ik] = q_points_ik;
    return q_points_ik;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::multiply_potential(T *density_recip, int ik, int iq) const
{
    int npw = rhopw->npw;
    auto k = wfcpw->kvec_c[ik];
    auto q = wfcpw->kvec_c[iq];
    double ucell_omega = elecstate::get_ucell_omega();
    double e2 = 2.0;
    // screen not yet implemented
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int ig = 0; ig < npw; ig++)
    {
        // |k - q + G| ^ 2
        auto kqgcar2 = (k - q + rhopw->gcar[ig]).norm2();
        auto hse_omega2 = GlobalC::exx_info.info_global.hse_omega * GlobalC::exx_info.info_global.hse_omega;
        if (kqgcar2 > 1e-12) // vasp uses 1/40 of the smallest (k spacing)**2
        {
            density_recip[ig] *= -4.0 * M_PI * e2 / kqgcar2 / ucell_omega;
            if (GlobalV::DFT_FUNCTIONAL == "hse")
            {
                density_recip[ig] *= (1 - std::exp(-kqgcar2 / 4.0 / hse_omega2));
            }
        }
        else 
        {
            density_recip[ig] = 0.0;
        }
        assert(is_finite(density_recip[ig]));
    }
}

template <typename T, typename Device>
const T *OperatorEXXPW<T, Device>::get_pw(const int m, const int iq) const
{
    return pws[iq].get() + m * wfcpw->npwk[iq];
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
OperatorEXXPW<T, Device>::OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op)
{
    // copy all the datas
    this->isk = op->isk;
    this->wfcpw = op->wfcpw;
    this->rhopw = op->rhopw;
    this->psi = op->psi;
    this->ctx = op->ctx;
    this->cpu_ctx = op->cpu_ctx;
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, density_recip, rhopw->npw);
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);
    this->pws.resize(wfcpw->nks);

    for (int ik = 0; ik < wfcpw->nks; ik++)
    {
        psi->fix_k(ik);
        this->pws[ik] = std::make_unique<T[]>(wfcpw->npwk[ik] * psi->get_nbands());
    }
    psi->fix_k(this->ik);

}

template class OperatorEXXPW<std::complex<float>, psi::DEVICE_CPU>;
template class OperatorEXXPW<std::complex<double>, psi::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
// to be implemented
#endif

} // namespace hamilt

#endif