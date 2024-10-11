#include "module_base/constants.h"
#include "module_base/global_variable.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_cell/klist.h"
#include "module_elecstate/elecstate_getters.h"
#include "module_hamilt_general/operator.h"
#include "module_psi/psi.h"
#include "module_base/tool_quit.h"

// #include <cassert>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <memory>
#include <utility>

#ifdef __EXX
#define __EXX_PW
#endif

#define __EXX_PW

#ifdef __EXX_PW
#include "op_exx_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace hamilt
{
template <typename T, typename Device>
void OperatorEXXPW<T, Device>::exx_divergence()
{
    if (GlobalC::exx_info.info_lip.lambda == 0.0)
    {
        return;
    }

    // here we follow the exx_divergence subroutine in q-e (PW/src/exx_base.f90)
    double alpha = 10.0 / wfcpw->gk_ecut;
    std::cout << "alpha: " << alpha << std::endl;
    // double alpha = GlobalC::exx_info.info_lip.lambda; // alternative way set by user
    double tpiba2 = elecstate::get_ucell_tpiba() * elecstate::get_ucell_tpiba();
    double div = 0;
    
    // this is the \sum_q F(q) part
    // temporarily for all k points, should be replaced to q points later
    for (int ik = 0; ik < wfcpw->nks; ik++)
    {
        auto k = wfcpw->kvec_c[ik];
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:div)
        #endif
        for (int ig = 0; ig < rhopw->npw; ig++)
        {
            auto q = k + rhopw->gcar[ig];
            double qq = q.norm2();
            if (qq <= 1e-8) continue;
            else if (PARAM.inp.dft_functional == "hse")
            {
                double omega = GlobalC::exx_info.info_global.hse_omega;
                double omega2 = omega * omega;
                div += std::exp(-alpha * qq) / qq * (1.0 - std::exp(-qq*tpiba2 / 4.0 / omega2));
            }
            else
            {
                div += std::exp(-alpha * qq) / qq;
            }
        }
    }

    Parallel_Reduce::reduce_pool(div);
    // std::cout << "EXX div: " << div << std::endl;

    if (PARAM.inp.dft_functional == "hse")
    {
        double omega = GlobalC::exx_info.info_global.hse_omega;
        div += tpiba2 / 4.0 / omega / omega; // compensate for the finite value when qq = 0
    }
    else 
    {
        div -= alpha;
    }

    div *= ModuleBase::e2 * ModuleBase::FOUR_PI / tpiba2 / wfcpw->nks;

    // numerically value the nean value of F(q) in the reciprocal space
    // This means we need to calculate the average of F(q) in the first brillouin zone
    alpha /= tpiba2;
    int nqq = 100000;
    double dq = 5.0 / std::sqrt(alpha) / nqq;
    double aa = 0.0;
    if (PARAM.inp.dft_functional == "hse")
    {
        double omega = GlobalC::exx_info.info_global.hse_omega;
        double omega2 = omega * omega;
        #ifdef _OPENMP
        #pragma omp parallel for reduction(+:aa)
        #endif
        for (int i = 0; i < nqq; i++)
        {
            double q = dq * (i+0.5);
            aa -= exp(-alpha * q * q) * exp(-q*q / 4.0 / omega2) * dq;
        }
    }
    aa *= 8 / ModuleBase::FOUR_PI;
    aa += 1.0 / std::sqrt(alpha * ModuleBase::PI);

    double omega = elecstate::get_ucell_omega();
    div -= ModuleBase::e2 * omega * aa;
    exx_div = div * wfcpw->nks;
    // std::cout << "EXX divergence: " << exx_div << std::endl;

    return;

}

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
    this->cal_type = hamilt::calculation_type::pw_exx;

    // allocate real space memory
    // assert(wfcpw->nrxx == rhopw->nrxx);
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw->nrxx);
    // allocate density recip space memory
    resmem_complex_op()(this->ctx, density_recip, rhopw->npw);
    // allocate h_psi recip space memory
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);
    // resmem_complex_op()(this->ctx, psi_all_real, wfcpw->nrxx * GlobalV::NBANDS);

    update_psi = true;
    exx_divergence();
    // GlobalC::exx_helper.op_exx = this;
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
//    if (GlobalC::exx_helper.first_iter) return;
    // return;

    ModuleBase::timer::tick("OperatorEXXPW", "act");

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
        Real nqs = q_points.size();
        for (int iq: q_points)
        {
            for (int m_iband = 0; m_iband < GlobalV::NBANDS; m_iband++)
            {
                // double wg_mqb_real = GlobalC::exx_helper.wg(iq, m_iband);
                double wg_mqb_real = (*(GlobalC::exx_helper.wg))(this->ik, m_iband);
                T wg_mqb = wg_mqb_real;
                if (wg_mqb_real < 1e-12)
                {
                    continue;
                }

                // if (has_real.find({iq, m_iband}) == has_real.end())
                // {
                    const T* psi_mq = get_pw(m_iband, iq);
                    wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                //     syncmem_complex_op()(this->ctx, this->ctx, psi_all_real + m_iband * wfcpw->nrxx, psi_mq_real, wfcpw->nrxx);
                //     has_real[{iq, m_iband}] = true;
                // }
                // else
                // {
                //     // const T* psi_mq = get_pw(m_iband, iq);
                //     // wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                //     syncmem_complex_op()(this->ctx, this->ctx, psi_mq_real, psi_all_real + m_iband * wfcpw->nrxx, wfcpw->nrxx);
                // }
                
                // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    // assert(is_finite(psi_nk_real[ir]));
                    // assert(is_finite(psi_mq_real[ir]));
                    Real ucell_omega = elecstate::get_ucell_omega();
                    density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]) / ucell_omega; // Phase e^(i(q-k)r)
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
                    // assert(is_finite(psi_mq_real[ir]));
                    // assert(is_finite(density_real[ir]));
                    density_real[ir] *= psi_mq_real[ir];
                }

                T wk_iq = kv->wk[iq];
                T wk_ik = kv->wk[this->ik];

                #ifdef _OPENMP
                #pragma omp parallel for
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    h_psi_real[ir] += density_real[ir] * wg_mqb / wk_iq / nqs;
                }

            } // end of m_iband
            setmem_complex_op()(this->ctx, density_real, 0, rhopw->nrxx);
            setmem_complex_op()(this->ctx, density_recip, 0, rhopw->npw);
            setmem_complex_op()(this->ctx, psi_mq_real, 0, wfcpw->nrxx);

        } // end of iq
        auto h_psi_nk = tmhpsi + n_iband * nbasis;
        Real hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        // wfcpw->real_to_recip(ctx, h_psi_real, h_psi_recip, this->ik);
        wfcpw->real_to_recip(ctx, h_psi_real, h_psi_nk, this->ik, true, hybrid_alpha);
        setmem_complex_op()(this->ctx, h_psi_real, 0, rhopw->nrxx);

        // add the h|psi_ik> to tmhpsi
        
        // T Eexx_ik = 0.0;
        // T ucell_nxyz = wfcpw->nxyz, ucell_omega = elecstate::get_ucell_omega();
        // T e2 = ModuleBase::e2;
        // #ifdef _OPENMP
        // #pragma omp parallel for
        // #endif
        // for (int ig = 0; ig < wfcpw->npwk_max; ig++)
        // {
        //     // assert no nan
        //     // assert(is_finite(h_psi_recip[ig]));
        //     // T wkik = kv->wk[this->ik];
        //     h_psi_nk[ig] += h_psi_recip[ig] * hybrid_alpha;// / wkik/* * ucell_omega * e2*/;
        // }
        
    }

    ModuleBase::timer::tick("OperatorEXXPW", "act");
    
}

template <typename  T, typename Device>
double OperatorEXXPW<T, Device>::get_Eexx() const
{
    return 0;
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
    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
    int npw = rhopw->npw;
    auto k = wfcpw->kvec_c[ik];
    auto q = wfcpw->kvec_c[iq];
    double ucell_omega = elecstate::get_ucell_omega();
    double tpiba2 = elecstate::get_ucell_tpiba() * elecstate::get_ucell_tpiba();
    // double e2 = 2.0;
    // screen not yet implemented
    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int ig = 0; ig < npw; ig++)
    {
        // |k - q + G| ^ 2
        Real gg = (k - q + rhopw->gcar[ig]).norm2() * tpiba2;
        // double gg = (rhopw->gg[ig] * tpiba2);
        Real hse_omega2 = GlobalC::exx_info.info_global.hse_omega * GlobalC::exx_info.info_global.hse_omega;
        // if (kqgcar2 > 1e-12) // vasp uses 1/40 of the smallest (k spacing)**2
        // {
//        density_recip[ig] *= 2;
        if (gg >= 1e-8)
        {
            Real fac = -ModuleBase::FOUR_PI * ModuleBase::e2 / gg;
            // density_recip[ig] *= fac;
            if (PARAM.inp.dft_functional == "hse")
            {
                density_recip[ig] *= fac * (1.0 - std::exp(-gg/ 4.0 / hse_omega2));
            }
            else 
            {
                density_recip[ig] *= fac;
            }
        }
        // }
        else 
        {
            // std::cout << "div at " << ig << std::endl;
            if (PARAM.inp.dft_functional == "hse")
            {
                // std::cout << "Factor: " << -ModuleBase::PI * ModuleBase::e2 / hse_omega2 << std::endl;
                density_recip[ig] *= exx_div - ModuleBase::PI * ModuleBase::e2 / hse_omega2;
            }
            else 
            {
                density_recip[ig] *= exx_div;
            }
        }
        // assert(is_finite(density_recip[ig]));
    }
    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
}

template <typename T, typename Device>
const T *OperatorEXXPW<T, Device>::get_pw(const int m, const int iq) const
{
    // return pws[iq].get() + m * wfcpw->npwk[iq];
    psi->fix_kb(iq, m);
    auto psi_mq = psi->get_pointer();
    return psi_mq;
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

template class OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>;
template class OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
// to be implemented
#endif

} // namespace hamilt

#endif