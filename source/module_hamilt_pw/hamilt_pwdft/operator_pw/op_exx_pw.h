#ifndef OPEXXPW_H
#define OPEXXPW_H

#include "module_base/matrix.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_cell/klist.h"
#include "module_psi/kernels/types.h"
#include "module_psi/psi.h"
#include "operator_pw.h"
#include "module_psi/kernels/memory_op.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_base/macros.h"

#include <memory>

#ifdef __EXX
#define __EXX_PW
#endif

#define __EXX_PW
#ifdef __EXX_PW
namespace hamilt
{

template <typename T, typename Device>
class OperatorEXXPW : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
  public:
    OperatorEXXPW(const int* isk_in,
                  const ModulePW::PW_Basis_K* wfcpw_in,
                  const ModulePW::PW_Basis* rhopw_in,
                  K_Vectors* kv_in);

    template <typename T_in, typename Device_in = Device>
    explicit OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op_exx);

    virtual ~OperatorEXXPW();

    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T *tmpsi_in,
                     T *tmhpsi,
                     const int ngk_ik = 0) const override;

    void set_psi(const psi::Psi<T, Device> *psi_in) const { this->psi = psi_in; }

    double get_Eexx() const { return this->Eexx; }

  private:
    const int *isk = nullptr;
    const ModulePW::PW_Basis_K *wfcpw = nullptr;
    const ModulePW::PW_Basis   *rhopw = nullptr;
    
    std::vector<int> get_q_points(const int ik) const;
    const T *get_pw(const int m, const int iq) const;

    void multiply_potential(T *density_recip, int ik, int iq) const;
    
    // pws
    mutable std::vector<std::unique_ptr<T[]>> pws;

    // k vectors
    K_Vectors *kv = nullptr;

    // psi memory
    mutable const psi::Psi<T, Device> *psi = nullptr;

    // // real space memory
    // std::unique_ptr<T[]> psi_nk_real;
    // std::unique_ptr<T[]> psi_mq_real;
    // std::unique_ptr<T[]> density_real;
    // // density recip space memory
    // std::unique_ptr<T[]> density_recip;
    // // h_psi recip space memory
    // std::unique_ptr<T[]> h_psi_recip;

    // real space memory
    T *psi_nk_real = nullptr;
    T *psi_mq_real = nullptr;
    T *density_real = nullptr;
    T *h_psi_real = nullptr;
    // density recip space memory
    T *density_recip = nullptr;
    // h_psi recip space memory
    T *h_psi_recip = nullptr;

    mutable std::map<int, std::vector<int>> q_points;

    // Energy
    mutable double Eexx = 0.0;

    mutable bool first_called = true;

    // occupational number
    ModuleBase::matrix wg;

    static bool update_psi;

    Device *ctx = {};
    psi::DEVICE_CPU* cpu_ctx = {};
    psi::AbacusDevice_t device = {};

    using setmem_complex_op = psi::memory::set_memory_op<T, Device>;
    using resmem_complex_op = psi::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = psi::memory::delete_memory_op<T, Device>;

};

} // namespace hamilt

#endif // __EXX_PW

#endif // OPEXXPW_H