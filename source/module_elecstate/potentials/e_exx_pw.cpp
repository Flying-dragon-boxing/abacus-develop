#include "e_exx_pw.h"

namespace elecstate
{

double E_exx::exx_energy = 0.0;

E_exx::E_exx()
{
}

E_exx::~E_exx()
{
}

double E_exx::cal_exx(const ModulePW::PW_Basis_K* wfcpw,
                      const ModulePW::PW_Basis* rhopw,
                      const int* isk,
                      const double* const* const rho,
                      const double& Eexx)
{
    double exx = 0.0;
    return exx;
}

} // namespace elecstate