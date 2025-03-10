#ifndef OPEXXPW_H
#define OPEXXPW_H

#include "module_base/matrix.h"
#include "module_basis/module_pw/pw_basis.h"
#include "module_cell/klist.h"
#include "module_psi/psi.h"
#include "operator_pw.h"
#include "module_basis/module_pw/pw_basis_k.h"
#include "module_base/macros.h"
#include "module_esolver/esolver_ks_pw.h"
#include "module_base/kernels/math_kernel_op.h"
#include "module_base/blas_connector.h"

#include <memory>
#include <utility>
#include <vector>

namespace hamilt
{

template <typename T, typename Device>
class OperatorEXXPW : public OperatorPW<T, Device>
{
  private:
    using Real = typename GetTypeReal<T>::type;
    using ExxHelper = Exx_Helper<T, Device>;
  public:
    OperatorEXXPW(const int* isk_in,
                  const ModulePW::PW_Basis_K* wfcpw_in,
                  const ModulePW::PW_Basis* rhopw_in,
                  K_Vectors* kv_in,
                  const UnitCell* ucell);

    template <typename T_in, typename Device_in = Device>
    explicit OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op_exx);

    virtual ~OperatorEXXPW();

    virtual void act(const int nbands,
                     const int nbasis,
                     const int npol,
                     const T *tmpsi_in,
                     T *tmhpsi,
                     const int ngk_ik = 0,
                     const bool is_first_node = false) const override;

    void act_op(const int nbands,
                const int nbasis,
                const int npol,
                const T *tmpsi_in,
                T *tmhpsi,
                const int ngk_ik = 0,
                const bool is_first_node = false) const;

    void act_op_ace(const int nbands,
                    const int nbasis,
                    const int npol,
                    const T *tmpsi_in,
                    T *tmhpsi,
                    const int ngk_ik = 0,
                    const bool is_first_node = false) const;

    void set_exx_helper(ExxHelper *p_exx_helper_in) const
    {
        this->p_exx_helper = p_exx_helper_in;
    }

    double cal_exx_energy() const;

  private:
    const int *isk = nullptr;
    const ModulePW::PW_Basis_K *wfcpw = nullptr;
    const ModulePW::PW_Basis   *rhopw = nullptr;
    const UnitCell *ucell = nullptr;
    Real exx_div = 0;
    Real tpiba = 0;
    
    std::vector<int> get_q_points(const int ik) const;
    const T *get_pw(const int m, const int iq) const;

    void multiply_potential(T *density_recip, int ik, int iq) const;

    void exx_divergence();

    void get_potential() const;

    void construct_ace() const;

    mutable int cnt = 0;

    mutable bool potential_got = false;
    
    // pws
    mutable std::vector<std::unique_ptr<T[]>> pws;

    // k vectors
    K_Vectors *kv = nullptr;

    mutable ExxHelper *p_exx_helper = nullptr;

    // real space memory
    T *psi_nk_real = nullptr;
    T *psi_mq_real = nullptr;
    T *density_real = nullptr;
    T *h_psi_real = nullptr;
    // density recip space memory
    T *density_recip = nullptr;
    // h_psi recip space memory
    T *h_psi_recip = nullptr;
    Real *pot = nullptr;

    // Lin Lin's ACE memory, 10.1021/acs.jctc.6b00092
    mutable T* h_psi_ace = nullptr; // H \Psi, W in the paper
    mutable T* psi_h_psi_ace = nullptr; // \Psi^{\dagger} H \Psi, M in the paper
    mutable T* L_ace = nullptr; // cholesky(-M).L, L in the paper
    mutable std::vector<T*> Xi_ace_k; // L^{-1} (H \Psi)^{\dagger}, \Xi in the paper
//    mutable T* Xi_ace = nullptr; // L^{-1} (H \Psi)^{\dagger}, \Xi in the paper

    mutable std::map<int, std::vector<int>> q_points;

    // occupational number
    ModuleBase::matrix wg;

//    mutable bool update_psi = false;

    Device *ctx = {};
    base_device::DEVICE_CPU* cpu_ctx = {};
    base_device::AbacusDevice_t device = {};

    using setmem_complex_op = base_device::memory::set_memory_op<T, Device>;
    using resmem_complex_op = base_device::memory::resize_memory_op<T, Device>;
    using delmem_complex_op = base_device::memory::delete_memory_op<T, Device>;
    using syncmem_complex_op = base_device::memory::synchronize_memory_op<T, Device, Device>;
    using resmem_real_op = base_device::memory::resize_memory_op<Real, Device>;
    using gemm_complex_op = ModuleBase::gemm_op<T, Device>;
    using vec_add_vec_complex_op = ModuleBase::constantvector_addORsub_constantVector_op<T, Device>;

};

} // namespace hamilt

#endif // OPEXXPW_H