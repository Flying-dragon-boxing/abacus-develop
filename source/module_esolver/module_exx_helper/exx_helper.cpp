#include "module_esolver/esolver_ks_pw.h"

template <typename T, typename Device>
double ModuleESolver::ESolver_KS_PW<T, Device>::Exx_Helper::cal_exx_energy(psi::Psi<T, Device>& psi, ESolver_KS_PW<T, Device>* this_)
{
    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    T* psi_nk_real = new T[this_->pw_wfc->nrxx];
    T* psi_mq_real = new T[this_->pw_wfc->nrxx];
    T* h_psi_recip = new T[this_->pw_wfc->npwk_max];
    T* h_psi_real = new T[this_->pw_wfc->nrxx];
    T* density_real = new T[this_->pw_wfc->nrxx];
    auto rhopw = this_->pelec->charge->rhopw;
    T* density_recip = new T[rhopw->npw];
    auto *kv = &this_->kv;

    // lambda
    auto exx_divergence = [&]() -> double
    {
        auto wfcpw = this_->pw_wfc;
        // if (GlobalC::exx_info.info_lip.lambda == 0.0)
        // {
        //     return 0;
        // }

        // here we follow the exx_divergence subroutine in q-e (PW/src/exx_base.f90)
        // double alpha = GlobalC::exx_info.info_lip.lambda;
        double alpha = 10.0 / wfcpw->gk_ecut;
        double tpiba2 = this_->pw_rhod->tpiba2;
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
                // else if (PARAM.inp.dft_functional == "hse")
                else if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
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

//        if (PARAM.inp.dft_functional == "hse")
        if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
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
        alpha /= tpiba2;
        int nqq = 100000;
        double dq = 5.0 / std::sqrt(alpha) / nqq;
        double aa = 0.0;
//        if (PARAM.inp.dft_functional == "hse")

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

        double omega = this_->pelec->omega;
        div -= ModuleBase::e2 * omega * aa;
        return div * wfcpw->nks;


    };

    if (div == DIV_UNDEFINED)
        div = exx_divergence();

    double exx_div = div;

    if (wf_wg == nullptr) return 0.0;
    ModuleBase::timer::tick("OperatorEXXPW", "get_Eexx");
    // evaluate the Eexx
    // T Eexx_ik = 0.0;
    Real Eexx_ik_real = 0.0;
    for (int ik = 0; ik < this_->pw_wfc->nks; ik++)
    {
        //        auto k = this->pw_wfc->kvec_c[ik];
        //        std::cout << k << std::endl;
        for (int n_iband = 0; n_iband < psi.get_nbands(); n_iband++)
        {
            setmem_complex_op()(h_psi_recip, 0, this_->pw_wfc->npwk_max);
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
            this_->pw_wfc->recip_to_real(this_->ctx, psi_nk, psi_nk_real, ik);

            // for \psi_nk, get the pw of iq and band m
            // q_points is a vector of integers, 0 to nks-1
            std::vector<int> q_points;
            for (int iq = 0; iq < this_->pw_wfc->nks; iq++)
            {
                q_points.push_back(iq);
            }
            Real nqs = q_points.size();

            //            std::cout << "ik = " << ik << " ib = " << n_iband << " wg_kb = " << wg_ikb_real << " wk_ik = " << kv->wk[ik] << std::endl;
            for (int iq: q_points)
            {
                double min_gg = 200;
                double max_gg = -1e8;
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
                    this_->pw_wfc->recip_to_real(this_->ctx, psi_mq, psi_mq_real, iq);

                    Real omega_inv = 1.0 / this_->pelec->omega;

                    // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                    #ifdef _OPENMP
                    #pragma omp parallel for
                    #endif
                    for (int ir = 0; ir < this_->pw_wfc->nrxx; ir++)
                    {
                        // assert(is_finite(psi_nk_real[ir]));
                        // assert(is_finite(psi_mq_real[ir]));
                        density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]);
                    }
                    // to be changed into kernel function

                    // bring the density to recip space
                    rhopw->real2recip(density_real, density_recip);

                    Real tpiba2 = this_->pw_rhod->tpiba2;
                    //                    std::cout << tpiba2 << std::endl;
                    Real hse_omega2 = GlobalC::exx_info.info_global.hse_omega * GlobalC::exx_info.info_global.hse_omega;

                    #ifdef _OPENMP
                    #pragma omp parallel for reduction(+:Eexx_ik_real) reduction(min:min_gg) reduction(max:max_gg)
                    #endif
                    for (int ig = 0; ig < rhopw->npw; ig++)
                    {
                        auto k = this_->pw_wfc->kvec_c[ik];// * latvec;
                        auto q = this_->pw_wfc->kvec_c[iq];// * latvec;
                        auto gcar = rhopw->gcar[ig];
                        double gg = (k - q + gcar).norm2() * tpiba2;

                        double Fac = 0.0;
                        if (gg >= 1e-8)
                        {
                            Fac = -ModuleBase::FOUR_PI * ModuleBase::e2 / gg;// * 2.57763;
//                            if (PARAM.inp.dft_functional == "hse")
                            if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
                            {

                                Fac *= (1 - std::exp(-gg/ 4.0 / hse_omega2));
                            }
                        }
                        else
                        {
//                            if (PARAM.inp.dft_functional == "hse")
                            if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
                            {
                                Fac =(-ModuleBase::PI * ModuleBase::e2 / hse_omega2 + exx_div);
                            }
                            else
                            {
                                // double exx_div = -4448.8824478350289 ;
                                Fac = exx_div;
                            }
                        }

                        Eexx_ik_real += Fac * (density_recip[ig] * std::conj(density_recip[ig])).real()
                                        * wg_iqb_real / nqs * wg_ikb_real / kv->wk[ik];
                    }

               } // m_iband

            } // iq

        } // n_iband

    } // ik
    Eexx_ik_real *= 0.5 / this_->pelec->omega;
    Parallel_Reduce::reduce_pool(Eexx_ik_real);
    //    std::cout << "Eexx: " << Eexx_ik_real << std::endl;

    Real Eexx = Eexx_ik_real;
    ModuleBase::timer::tick("OperatorEXXPW", "get_Eexx");
    return Eexx;
}

template class ModuleESolver::ESolver_KS_PW<std::complex<float>, base_device::DEVICE_CPU>;
template class ModuleESolver::ESolver_KS_PW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
template class ModuleESolver::ESolver_KS_PW<std::complex<float>, base_device::DEVICE_GPU>;
template class ModuleESolver::ESolver_KS_PW<std::complex<double>, base_device::DEVICE_GPU>;
#endif