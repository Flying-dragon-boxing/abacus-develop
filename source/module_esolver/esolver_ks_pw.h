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
    class Exx_Helper
    {
      public:
        Exx_Helper() = default;
        ModuleBase::matrix *wg;
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
            // if (std::abs(exx_energy[0] - exx_lip.get_exx_energy()) < 1e-5)
            else if (iter == 1)
            {
                return true;
            }
            GlobalV::ofs_running << "Updating EXX and rerun SCF" << std::endl;
            iter = 0;
            return false;

        }
        bool first_iter = true;
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

    double cal_exx_energy(psi::Psi<T, Device> psi);

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
