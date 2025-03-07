//
// For EXX in PW.
//
#include "module_psi/psi.h"
#include "module_base/matrix.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

#ifndef EXX_HELPER_H
#define EXX_HELPER_H
template <typename T, typename Device>
struct Exx_Helper
{
  public:
    Exx_Helper() = default;
    ModuleBase::matrix * wf_wg;
    psi::Psi<T, Device> psi;
    static constexpr double DIV_UNDEFINED = 0x0d000721;
    double div = DIV_UNDEFINED;
    bool construct_ace = false;

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
        construct_ace = true;
    }

    void reset_div()
    {
        this->div = DIV_UNDEFINED;
    }

    double cal_exx_energy(const Device *ctx,
                          psi::Psi<T, Device>& psi,
                          ModulePW::PW_Basis_K* pw_wfc,
                          ModulePW::PW_Basis* pw_rho,
                          UnitCell* ucell,
                          K_Vectors *kv);


    bool first_iter = false;
};
#endif // EXX_HELPER_H
