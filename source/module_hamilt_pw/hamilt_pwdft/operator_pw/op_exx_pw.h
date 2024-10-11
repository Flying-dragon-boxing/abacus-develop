#ifndef OPEXXPW_H
#define OPEXXPW_H

#include "module_base/matrix.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_cell/klist.h"
//#include "module_psi/kernels/types.h"
#include "module_psi/psi.h"
#include "operator_pw.h"
//#include "module_psi/kernels/memory_op.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_base/macros.h"

#include <memory>
#include <utility>
#include <vector>

#ifdef __EXX
#define __EXX_PW
#endif

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

    void set_psi(const psi::Psi<T, Device> *psi_in) const 
    { 
        this->psi = psi_in;
        // get_Eexx();
    }

    double get_Eexx() const;
    double get_exx_div() const { return exx_div; }

    void set_update_psi(bool update_psi_in) const { update_psi = update_psi_in; }

  private:
    const int *isk = nullptr;
    const ModulePW::PW_Basis_K *wfcpw = nullptr;
    const ModulePW::PW_Basis   *rhopw = nullptr;
    Real exx_div = 0;
    
    std::vector<int> get_q_points(const int ik) const;
    const T *get_pw(const int m, const int iq) const;

    void multiply_potential(T *density_recip, int ik, int iq) const;

    void exx_divergence();

    mutable int cnt = 0;
    
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
    T* psi_all_real = nullptr;

    mutable std::map<int, std::vector<int>> q_points;
    // has real calced
    mutable std::map<std::pair<int, int>, bool> has_real;

    // Energy
    mutable double Eexx = 0.0;

    mutable bool first_called = true;

    // occupational number
    ModuleBase::matrix wg;

    mutable bool update_psi = false;

    Device *ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = {};

    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_complex_op = base_device::memory::synchronize_memory_op<T, Device, Device>;

};

} // namespace hamilt

#endif // __EXX_PW

#endif // OPEXXPW_H