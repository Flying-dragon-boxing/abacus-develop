#ifndef ESOLVER_KS_PW_H
#define ESOLVER_KS_PW_H
#include "./esolver_ks.h"
#include "module_hamilt_pw/hamilt_pwdft/operator_pw/velocity_pw.h"
#include "module_psi/psi_init.h"

#include "module_hamilt_pw/hamilt_pwdft/global.h"

#include <memory>
#include <module_base/macros.h>

namespace ModuleESolver
{

template <typename T, typename Device = base_device::DEVICE_CPU>
class ESolver_KS_PW : public ESolver_KS<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;

  public:
    ESolver_KS_PW();

    ~ESolver_KS_PW();

    void before_all_runners(UnitCell& ucell, const Input_para& inp) override;

    double cal_energy() override;

    void cal_force(UnitCell& ucell, ModuleBase::matrix& force) override;

    void cal_stress(UnitCell& ucell, ModuleBase::matrix& stress) override;

    void after_all_runners(UnitCell& ucell) override;

#ifdef __EXX
    struct Exx_Helper
    {
      public:
        Exx_Helper() = default;
        ModuleBase::matrix * wf_wg;
        psi::Psi<T, base_device::DEVICE_CPU> psi;
        static constexpr double DIV_UNDEFINED = 0x0d000721;
        double div = DIV_UNDEFINED;

        bool exx_after_converge(int &iter)
        {
            if (first_iter)
            {
                first_iter = false;
            }
            else if (!GlobalC::exx_info.info_global.separate_loop)
            {
                return true;
            }
            else if (iter == 1)
            {
                return true;
            }
            GlobalV::ofs_running << "Updating EXX and rerun SCF" << std::endl;
            iter = 0;
            return false;

        }

        void set_psi(psi::Psi<T, Device> &psi_)
        {
            this->psi = psi_;
        }

        double cal_exx_energy(psi::Psi<T, Device> &psi, ESolver_KS_PW<T, Device> *this_)
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
                                density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]) * omega_inv;
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
                                    if (PARAM.inp.dft_functional == "hse")
                                    {
                                        Fac *= (1 - std::exp(-gg/ 4.0 / hse_omega2));
                                    }
                                }
                                else
                                {
                                    if (PARAM.inp.dft_functional == "hse")
                                    {
                                        Fac =(-ModuleBase::PI * ModuleBase::e2 / hse_omega2 + exx_div);
                                    }
                                    else
                                    {
                                        // double exx_div = -4448.8824478350289 ;
                                        Fac = exx_div;
                                    }
                                }
                                min_gg = std::min(min_gg, Fac);
                                max_gg = std::max(max_gg, Fac);
                                Eexx_ik_real += Fac * (density_recip[ig] * std::conj(density_recip[ig])).real()
                                                * wg_iqb_real / nqs * wg_ikb_real / kv->wk[ik];
                            }
                            MPI_Allreduce(&min_gg, &min_gg, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
                            MPI_Allreduce(&max_gg, &max_gg, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
                        } // m_iband

                    } // iq

                } // n_iband

            } // ik
            Eexx_ik_real *= 0.5 * this_->pelec->omega;
            Parallel_Reduce::reduce_pool(Eexx_ik_real);
            //    std::cout << "Eexx: " << Eexx_ik_real << std::endl;

            Real Eexx = Eexx_ik_real;
            ModuleBase::timer::tick("OperatorEXXPW", "get_Eexx");
            return Eexx;
        }


        bool first_iter = false;
    };
#endif

  protected:
    virtual void before_scf(UnitCell& ucell, const int istep) override;

    virtual void iter_init(UnitCell& ucell, const int istep, const int iter) override;

    virtual void update_pot(UnitCell& ucell, const int istep, const int iter) override;

    virtual void iter_finish(UnitCell& ucell, const int istep, int& iter) override;

    virtual void after_scf(UnitCell& ucell, const int istep) override;

    virtual void others(UnitCell& ucell, const int istep) override;

    virtual void hamilt2density_single(UnitCell& ucell, const int istep, const int iter, const double ethr) override;
    // EXX Todo: verify current implementation for after_converge
    // virtual bool do_after_converge(int &iter) override;
#ifdef __EXX
    Exx_Helper exx_helper;
#endif

    virtual void allocate_hamilt(const UnitCell& ucell);
    virtual void deallocate_hamilt();

    //! hide the psi in ESolver_KS for tmp use
    psi::Psi<std::complex<double>, base_device::DEVICE_CPU>* psi = nullptr;

    // psi_initializer controller
    psi::PSIInit<T, Device>* p_psi_init = nullptr;

    Device* ctx = {};

    base_device::AbacusDevice_t device = {};

    psi::Psi<T, Device>* kspw_psi = nullptr;

    psi::Psi<std::complex<double>, Device>* __kspw_psi = nullptr;

    bool already_initpsi = false;

    using castmem_2d_d2h_op
        = base_device::memory::cast_memory_op<std::complex<double>, T, base_device::DEVICE_CPU, Device>;

};
} // namespace ModuleESolver
#endif
