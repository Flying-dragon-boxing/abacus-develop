#ifndef E_EXX_PW_H
#define E_EXX_PW_H

#include "module_basis/module_pw/pw_basis.h"
#include "module_basis/module_pw/pw_basis_k.h"

namespace elecstate
{

class E_exx
{
  public:
    E_exx();
    ~E_exx();

    static double cal_exx(const ModulePW::PW_Basis_K* wfcpw,
                        const ModulePW::PW_Basis* rhopw,
                        const int* isk,
                        const double* const* const rho,
                        const double& Eexx);

    static double get_exx_energy() { return exx_energy; }

  private:
    static double exx_energy;
    
};

} // namespace elecstate

#endif // E_EXX_PW_H