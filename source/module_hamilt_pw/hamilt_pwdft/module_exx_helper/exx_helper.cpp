#include "exx_helper.h"

template <typename T, typename Device>
double Exx_Helper<T, Device>::cal_exx_energy(const Device *ctx, psi::Psi<T, Device>& psi, ModulePW::PW_Basis_K* pw_wfc, ModulePW::PW_Basis* pw_rho, UnitCell* ucell, K_Vectors *kv)
{
    ModuleBase::timer::tick("ESolver_KS_PW", "cal_exx_energy");

    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    T* psi_nk_real = new T[pw_wfc->nrxx];
    T* psi_mq_real = new T[pw_wfc->nrxx];
    T* h_psi_recip = new T[pw_wfc->npwk_max];
    T* h_psi_real = new T[pw_wfc->nrxx];
    T* density_real = new T[pw_wfc->nrxx];
    auto rhopw = pw_rho;
    T* density_recip = new T[rhopw->npw];

    double exx_div = div;

    if (wf_wg == nullptr) return 0.0;
    // evaluate the Eexx
    // T Eexx_ik = 0.0;
    double Eexx_ik_real = 0.0;
    for (int ik = 0; ik < pw_wfc->nks; ik++)
    {
        //        auto k = this->pw_wfc->kvec_c[ik];
        //        std::cout << k << std::endl;
        for (int n_iband = 0; n_iband < psi.get_nbands(); n_iband++)
        {
            setmem_complex_op()(h_psi_recip, 0, pw_wfc->npwk_max);
            setmem_complex_op()(h_psi_real, 0, rhopw->nrxx);
            setmem_complex_op()(density_real, 0, rhopw->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw->npw);

            // double wg_ikb_real = GlobalC::exx_helper.wg(this->ik, n_iband);
            double wg_ikb_real = (*wf_wg)(ik, n_iband);
            T wg_ikb = wg_ikb_real;
            if (wg_ikb_real < 1e-12)
            {
                continue;
            }

            //            std::cout << "ik = " << ik << " nb = " << n_iband << " wg_ikb = " << wg_ikb_real << std::endl;

            // const T *psi_nk = get_pw(n_iband, ik);
            psi.fix_kb(ik, n_iband);
            const T* psi_nk = psi.get_pointer();
            // retrieve \psi_nk in real space
            pw_wfc->recip_to_real(ctx, psi_nk, psi_nk_real, ik);

            // for \psi_nk, get the pw of iq and band m
            // q_points is a vector of integers, 0 to nks-1
            std::vector<int> q_points;
            for (int iq = 0; iq < pw_wfc->nks; iq++)
            {
                q_points.push_back(iq);
            }
            double nqs = q_points.size();

            //            std::cout << "ik = " << ik << " ib = " << n_iband << " wg_kb = " << wg_ikb_real << " wk_ik = " << kv->wk[ik] << std::endl;
            for (int iq: q_points)
            {
                for (int m_iband = 0; m_iband < psi.get_nbands(); m_iband++)
                {
                    // double wg_f = GlobalC::exx_helper.wg(iq, m_iband);
                    double wg_iqb_real = (*wf_wg)(iq, m_iband);
                    T wg_iqb = wg_iqb_real;
                    if (wg_iqb_real < 1e-12)
                    {
                        continue;
                    }

                    //                    std::cout << "iq = " << iq << " mb = " << m_iband << " wg_iqb = " << wg_iqb_real << std::endl;

                    psi.fix_kb(iq, m_iband);
                    const T* psi_mq = psi.get_pointer();
                    // const T* psi_mq = get_pw(m_iband, iq);
                    pw_wfc->recip_to_real(ctx, psi_mq, psi_mq_real, iq);

                    T omega_inv = 1.0 / ucell->omega;

                    // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int ir = 0; ir < pw_wfc->nrxx; ir++)
                    {
                        // assert(is_finite(psi_nk_real[ir]));
                        // assert(is_finite(psi_mq_real[ir]));
                        density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]) * omega_inv;
                    }
                    // to be changed into kernel function

                    // bring the density to recip space
                    rhopw->real2recip(density_real, density_recip);

                    #ifdef _OPENMP
                    #pragma omp parallel for reduction(+:Eexx_ik_real)
                    #endif
                    for (int ig = 0; ig < rhopw->npw; ig++)
                    {
                        int nks = pw_wfc->nks;
                        int npw = rhopw->npw;
                        Real Fac = pot[ik * nks * npw + iq * npw + ig];
                        Eexx_ik_real += Fac * (density_recip[ig] * std::conj(density_recip[ig])).real()
                                        * wg_iqb_real / nqs * wg_ikb_real / kv->wk[ik];
                    }

               } // m_iband

            } // iq

        } // n_iband

    } // ik
    Eexx_ik_real *= 0.5 * ucell->omega;
    Parallel_Reduce::reduce_pool(Eexx_ik_real);
//    std::cout << "omega = " << this_->pelec->omega << " tpiba = " << this_->pw_rho->tpiba2 << " exx_div = " << exx_div << std::endl;

    double Eexx = Eexx_ik_real;
    ModuleBase::timer::tick("ESolver_KS_PW", "cal_exx_energy");
    return Eexx;
}

template class Exx_Helper<std::complex<float>, base_device::DEVICE_CPU>;
template class Exx_Helper<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class Exx_Helper<std::complex<float>, base_device::DEVICE_GPU>;
template class Exx_Helper<std::complex<double>, base_device::DEVICE_GPU>;
#endif

#ifndef __EXX
#include "module_hamilt_general/module_xc/exx_info.h"
namespace GlobalC
{
    Exx_Info exx_info;
}
#endif