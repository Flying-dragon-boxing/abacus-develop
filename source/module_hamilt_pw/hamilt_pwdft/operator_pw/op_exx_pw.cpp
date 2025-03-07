#include "module_base/constants.h"
#include "module_base/global_variable.h"
#include "module_base/parallel_reduce.h"
#include "module_base/timer.h"
#include "module_cell/klist.h"
#include "module_hamilt_general/operator.h"
#include "module_psi/psi.h"
#include "module_base/tool_quit.h"

#include <cmath>
#include <complex>
#include <cstdlib>
#include <memory>
#include <utility>

extern "C"
{
    void ztrtri_(char *uplo, char *diag, int *n, std::complex<double> *a, int *lda, int *info);
    void ctrtri_(char *uplo, char *diag, int *n, std::complex<float> *a, int *lda, int *info);
}

//extern "C" void zpotrf_(char* uplo, const int* n, std::complex<double>* A, const int* lda, int* info);
//extern "C" void cpotrf_(char* uplo, const int* n, std::complex<float>* A, const int* lda, int* info);

#ifdef __EXX
#define __EXX_PW
#endif

#ifdef __EXX_PW
#include "op_exx_pw.h"
#include "module_hamilt_pw/hamilt_pwdft/global.h"

namespace hamilt
{
template <typename T, typename Device>
OperatorEXXPW<T, Device>::OperatorEXXPW(const int* isk_in,
                                        const ModulePW::PW_Basis_K* wfcpw_in,
                                        const ModulePW::PW_Basis* rhopw_in,
                                        K_Vectors *kv_in,
                                        const UnitCell *ucell)
    : isk(isk_in), wfcpw(wfcpw_in), rhopw(rhopw_in), kv(kv_in), ucell(ucell)
{

    if (GlobalV::KPAR != 1)
    {
        // GlobalV::ofs_running << "EXX Calculation does not support k-point parallelism" << std::endl;
        ModuleBase::WARNING_QUIT("OperatorEXXPW", "EXX Calculation does not support k-point parallelism");
    }

    this->classname = "OperatorEXXPW";
    this->ctx = nullptr;
    this->cpu_ctx = nullptr;
    this->cal_type = hamilt::calculation_type::pw_exx;

    // allocate real space memory
    // assert(wfcpw->nrxx == rhopw->nrxx);
    resmem_complex_op()(psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(density_real, rhopw->nrxx);
    resmem_complex_op()(h_psi_real, rhopw->nrxx);
    // allocate density recip space memory
    resmem_complex_op()(density_recip, rhopw->npw);
    // allocate h_psi recip space memory
    resmem_complex_op()(h_psi_recip, wfcpw->npwk_max);
    // resmem_complex_op()(this->ctx, psi_all_real, wfcpw->nrxx * GlobalV::NBANDS);
    int nks = wfcpw->nks;
//    std::cout << "nks: " << nks << std::endl;
    resmem_real_op()(pot, rhopw->npw * nks * nks);

    tpiba = ucell->tpiba;
    Real tpiba2 = tpiba * tpiba;
    // calculate the exx_divergence
    exx_divergence();

}

template <typename T, typename Device>
OperatorEXXPW<T, Device>::~OperatorEXXPW()
{
    // use delete_memory_op to delete the allocated pws
    delmem_complex_op()(psi_nk_real);
    delmem_complex_op()(psi_mq_real);
    delmem_complex_op()(density_real);
    delmem_complex_op()(h_psi_real);
    delmem_complex_op()(density_recip);
    delmem_complex_op()(h_psi_recip);
}

template <typename T>
inline bool is_finite(const T &val)
{
    return std::isfinite(val);
}

template <>
inline bool is_finite(const std::complex<float> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <>
inline bool is_finite(const std::complex<double> &val)
{
    return std::isfinite(val.real()) && std::isfinite(val.imag());
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik,
                                   const bool is_first_node) const
{
    if (p_exx_helper->first_iter) return;

    if (p_exx_helper->construct_ace)
    {
        construct_ace();
        std::cout << "ACE constructed" << std::endl;
        p_exx_helper->construct_ace = false;
    }

    if (is_first_node)
    {
        setmem_complex_op()(tmhpsi, 0, nbasis*nbands/npol);
    }

    if (ace)
    {
        act_op_ace(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
    }
    else
    {
        act_op(nbands, nbasis, npol, tmpsi_in, tmhpsi, ngk_ik, is_first_node);
    }
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op(const int nbands,
                                   const int nbasis,
                                   const int npol,
                                   const T *tmpsi_in,
                                   T *tmhpsi,
                                   const int ngk_ik,
                                   const bool is_first_node) const
{
    if (!potential_got)
    {
        get_potential();
        potential_got = true;
    }

//    set_psi(&p_exx_helper->psi);

    ModuleBase::timer::tick("OperatorEXXPW", "act_op");

    setmem_complex_op()(h_psi_recip, 0, wfcpw->npwk_max);
    setmem_complex_op()(h_psi_real, 0, rhopw->nrxx);
    setmem_complex_op()(density_real, 0, rhopw->nrxx);
    setmem_complex_op()(density_recip, 0, rhopw->npw);
    // setmem_complex_op()(psi_all_real, 0, wfcpw->nrxx * GlobalV::NBANDS);
    // std::map<std::pair<int, int>, bool> has_real;
    setmem_complex_op()(psi_nk_real, 0, wfcpw->nrxx);
    setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

    // ik fixed here, select band n
    for (int n_iband = 0; n_iband < nbands; n_iband++)
    {
        const T *psi_nk = tmpsi_in + n_iband * nbasis;
        // retrieve \psi_nk in real space
        wfcpw->recip_to_real(ctx, psi_nk, psi_nk_real, this->ik);

        // for \psi_nk, get the pw of iq and band m
        auto q_points = get_q_points(this->ik);
        Real nqs = q_points.size();
        for (int iq: q_points)
        {
            for (int m_iband = 0; m_iband < p_exx_helper->psi.get_nbands(); m_iband++)
            {
                // double wg_mqb_real = GlobalC::exx_helper.wg(iq, m_iband);
                double wg_mqb_real = (*p_exx_helper->wf_wg)(this->ik, m_iband);
                T wg_mqb = wg_mqb_real;
                if (wg_mqb_real < 1e-12)
                {
                    continue;
                }

                // if (has_real.find({iq, m_iband}) == has_real.end())
                // {
                    const T* psi_mq = get_pw(m_iband, iq);
                    wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                //     syncmem_complex_op()(this->ctx, this->ctx, psi_all_real + m_iband * wfcpw->nrxx, psi_mq_real, wfcpw->nrxx);
                //     has_real[{iq, m_iband}] = true;
                // }
                // else
                // {
                //     // const T* psi_mq = get_pw(m_iband, iq);
                //     // wfcpw->recip_to_real(ctx, psi_mq, psi_mq_real, iq);
                //     syncmem_complex_op()(this->ctx, this->ctx, psi_mq_real, psi_all_real + m_iband * wfcpw->nrxx, wfcpw->nrxx);
                // }
                
                // direct multiplication in real space, \psi_nk(r) * \psi_mq(r)
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    // assert(is_finite(psi_nk_real[ir]));
                    // assert(is_finite(psi_mq_real[ir]));
                    Real ucell_omega = ucell->omega;
                    density_real[ir] = psi_nk_real[ir] * std::conj(psi_mq_real[ir]) / ucell_omega; // Phase e^(i(q-k)r)
                }
                // to be changed into kernel function
                
                // bring the density to recip space
                rhopw->real2recip(density_real, density_recip);

                // multiply the density with the potential in recip space
                multiply_potential(density_recip, this->ik, iq);

                // bring the potential back to real space
                rhopw->recip2real(density_recip, density_real);

                // get the h|psi_ik>(r), save in density_real
                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    // assert(is_finite(psi_mq_real[ir]));
                    // assert(is_finite(density_real[ir]));
                    density_real[ir] *= psi_mq_real[ir];
                }

                T wk_iq = kv->wk[iq];
                T wk_ik = kv->wk[this->ik];

                #ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                #endif
                for (int ir = 0; ir < wfcpw->nrxx; ir++)
                {
                    h_psi_real[ir] += density_real[ir] * wg_mqb / wk_iq / nqs;
                }

            } // end of m_iband
            setmem_complex_op()(density_real, 0, rhopw->nrxx);
            setmem_complex_op()(density_recip, 0, rhopw->npw);
            setmem_complex_op()(psi_mq_real, 0, wfcpw->nrxx);

        } // end of iq
        auto h_psi_nk = tmhpsi + n_iband * nbasis;
        Real hybrid_alpha = GlobalC::exx_info.info_global.hybrid_alpha;
        wfcpw->real_to_recip(ctx, h_psi_real, h_psi_nk, this->ik, true, hybrid_alpha);
        setmem_complex_op()(h_psi_real, 0, rhopw->nrxx);
        
    }

    ModuleBase::timer::tick("OperatorEXXPW", "act_op");
    
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::act_op_ace(const int nbands,
                                          const int nbasis,
                                          const int npol,
                                          const T *tmpsi_in,
                                          T *tmhpsi,
                                          const int ngk_ik,
                                          const bool is_first_node) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "act_op_ace");

//    std::cout << "act_op_ace" << std::endl;
    // hpsi += -Xi^\dagger * Xi * psi
    auto Xi_ace = Xi_ace_k[this->ik];
    int nbands_tot = p_exx_helper->psi.get_nbands();
    int nbasis_max = p_exx_helper->psi.get_nbasis();
//    T* hpsi = nullptr;
//    resmem_complex_op()(hpsi, nbands_tot * nbasis);
//    setmem_complex_op()(hpsi, 0, nbands_tot * nbasis);
    T* Xi_psi = nullptr;
    resmem_complex_op()(Xi_psi, nbands_tot * nbands);
    setmem_complex_op()(Xi_psi, 0, nbands_tot * nbands);

    char trans_N = 'N', trans_T = 'T', trans_C = 'C';
    T intermediate_one = 1.0, intermediate_zero = 0.0, intermediate_minus_one = -1.0;
    // Xi * psi
    gemm_complex_op()(trans_N,
                      trans_N,
                      nbands_tot,
                      nbands,
                      nbasis,
                      &intermediate_one,
                      Xi_ace,
                      nbands_tot,
                      tmpsi_in,
                      nbasis,
                      &intermediate_zero,
                      Xi_psi,
                      nbands_tot
        );

    Parallel_Reduce::reduce_pool(Xi_psi, nbands_tot * nbands);

    // Xi^\dagger * (Xi * psi)
    gemm_complex_op()(trans_C,
                      trans_N,
                      nbasis,
                      nbands,
                      nbands_tot,
                      &intermediate_minus_one,
                      Xi_ace,
                      nbands_tot,
                      Xi_psi,
                      nbands_tot,
                      &intermediate_one,
                      tmhpsi,
                      nbasis
        );


//    // negative sign, add to hpsi
//    vec_add_vec_complex_op()(this->ctx, nbands * nbasis, tmhpsi, hpsi, -1, tmhpsi, 1);
//    delmem_complex_op()(hpsi);
    delmem_complex_op()(Xi_psi);
    ModuleBase::timer::tick("OperatorEXXPW", "act_op");

}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::construct_ace() const
{
//    int nkb = p_exx_helper->psi.get_nbands() * p_exx_helper->psi.get_nk();
    int nbands = p_exx_helper->psi.get_nbands();
    int nbasis = p_exx_helper->psi.get_nbasis();
    int nk = p_exx_helper->psi.get_nk();

    int ik_store = this->ik;
    int *ik_ptr = const_cast<int*>(&this->ik);

    T intermediate_one = 1.0, intermediate_zero = 0.0;

    if (h_psi_ace == nullptr)
    {
        resmem_complex_op()(h_psi_ace, nbands * nbasis);
        setmem_complex_op()(h_psi_ace, 0, nbands * nbasis);
    }

    if (Xi_ace_k.size() != nk)
    {
        Xi_ace_k.resize(nk);
        for (int i = 0; i < nk; i++)
        {
            resmem_complex_op()(Xi_ace_k[i], nbands * nbasis);
        }
    }

    for (int i = 0; i < nk; i++)
    {
        setmem_complex_op()(Xi_ace_k[i], 0, nbands * nbasis);
    }

    if (L_ace == nullptr)
    {
        resmem_complex_op()(L_ace, nbands * nbands);
        setmem_complex_op()(L_ace, 0, nbands * nbands);
    }

    if (psi_h_psi_ace == nullptr)
    {
        resmem_complex_op()(psi_h_psi_ace, nbands * nbands);
    }

    for (int ik = 0; ik < nk; ik++)
    {
        int npwk = wfcpw->npwk[ik];

        T* Xi_ace = Xi_ace_k[ik];
        p_exx_helper->psi.fix_kb(ik, 0);
        T* p_psi = p_exx_helper->psi.get_pointer();

        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {
        //     std::ofstream ofs_psi("psi.dat", std::ios::binary);
        //     // p_exx_helper->psi.fix_kb(0, 0);
        //     ofs_psi.write(reinterpret_cast<char*>(p_psi), nbands * nbasis * sizeof(T));
        //     ofs_psi.close();
        // }

        setmem_complex_op()(h_psi_ace, 0, nbands * nbasis);

        *ik_ptr = ik;

        act_op(
            nbands,
            nbasis,
            1,
            p_psi,
            h_psi_ace,
            nbasis,
            false
            );

        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {
        //     std::ofstream ofs_hpsi("hpsi.dat", std::ios::binary);
        //     ofs_hpsi.write(reinterpret_cast<char*>(h_psi_ace), nbands * nbasis * sizeof(T));
        //     ofs_hpsi.close();
        // }

        // psi_h_psi_ace = psi^\dagger * h_psi_ace
        // p_exx_helper->psi.fix_kb(0, 0);
        gemm_complex_op()('C',
                          'N',
                          nbands,
                          nbands,
                          npwk,
                          &intermediate_one,
                          p_psi,
                          nbasis,
                          h_psi_ace,
                          nbasis,
                          &intermediate_zero,
                          psi_h_psi_ace,
                          nbands);

        // reduction of psi_h_psi_ace, due to distributed memory
        Parallel_Reduce::reduce_pool(psi_h_psi_ace, nbands * nbands); 
        
        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {  // save psi_h_psi_ace to disk
        //    std::ofstream ofs_psi_hpsi("psihpsi.dat", std::ios::binary);
        //    ofs_psi_hpsi.write(reinterpret_cast<char*>(psi_h_psi_ace), nbands * nbands * sizeof(T));
        //    ofs_psi_hpsi.close();
        // }

        // L_ace = cholesky(-psi_h_psi_ace)
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < nbands; i++)
        {
            for (int j = 0; j < nbands; j++)
            {
                L_ace[i * nbands + j] = -psi_h_psi_ace[i * nbands + j];
            }
        }

        int info = 0;
        char up = 'U', lo = 'L';

        if constexpr (std::is_same<T, std::complex<float>>::value)
        {
            cpotrf_(&lo, &nbands, L_ace, &nbands, &info);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            zpotrf_(&lo, &nbands, L_ace, &nbands, &info);
        }

        // expand for-loop
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (int i = 0; i < nbands; i++)
        {
            for (int j = 0; j < nbands; j++)
            {
                if (j < i)
                {
                    // L_ace[j * nkb + i] = std::conj(L_ace[i * nkb + j]);
                    L_ace[i * nbands + j] = 0.0;
                }
            }
        }

        // // save L_ace to disk
        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {
        //     std::ofstream ofs_L("L.dat", std::ios::binary);
        //     ofs_L.write(reinterpret_cast<char*>(L_ace), nbands * nbands * sizeof(T));
        //     ofs_L.close();
        // }

        // L_ace inv in place
        // T == std::complex<float> or std::complex<double>
        if constexpr (std::is_same<T, std::complex<float>>::value)
        {
            char non_unitary = 'N';

            ctrtri_(&lo, &non_unitary, &nbands, L_ace, &nbands, &info);
        }
        else if constexpr (std::is_same<T, std::complex<double>>::value)
        {
            char non_unitary = 'N';

            ztrtri_(&lo, &non_unitary, &nbands, L_ace, &nbands, &info);
        }

        // // save L_ace inv to disk
        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {   
        //     std::ofstream ofs_L_inv("L_inv.dat", std::ios::binary);
        //     ofs_L_inv.write(reinterpret_cast<char*>(L_ace), nbands * nbands * sizeof(T));
        //     ofs_L_inv.close();
        // }

        // Xi_ace = L_ace^-1 * h_psi_ace^dagger
        gemm_complex_op()('N',
                          'C',
                          nbands,
                          npwk,
                          nbands,
                          &intermediate_one,
                          L_ace,
                          nbands,
                          h_psi_ace,
                          nbasis,
                          &intermediate_zero,
                          Xi_ace,
                          nbands);

            // save Xi_ace to disk
        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {   
        //     std::ofstream ofs_Xi("Xi.dat", std::ios::binary);
        //     ofs_Xi.write(reinterpret_cast<char*>(Xi_ace), nbands * nbasis * sizeof(T));
        //     ofs_Xi.close();
        // }
        //
        // //    std::cout << "nkb: " << nkb << std::endl;
        // if (ik == 1 && GlobalV::RANK_IN_POOL == 0)
        // {
        //     std::cout << "nbands: " << p_exx_helper->psi.get_nbands() << std::endl;
        //     std::cout << "nbasis: " << nbasis << std::endl;
        //     std::cout << "npwk: " << npwk << std::endl;
        // }

        // clear mem
        setmem_complex_op()(h_psi_ace, 0, nbands * nbasis);
        setmem_complex_op()(psi_h_psi_ace, 0, nbands * nbands);
        setmem_complex_op()(L_ace, 0, nbands * nbands);

    }

    *ik_ptr = ik_store;

//    // save h_psi_ace to disk
//    std::ofstream ofs_hpsi("hpsi.dat", std::ios::binary);
//    ofs_hpsi.write(reinterpret_cast<char*>(h_psi_ace), nkb * nbasis * sizeof(T));
//    ofs_hpsi.close();




}

template <typename T, typename Device>
std::vector<int> OperatorEXXPW<T, Device>::get_q_points(const int ik) const
{
    // stored in q_points
    if (q_points.find(ik) != q_points.end())
    {
        return q_points.find(ik)->second;
    }

    std::vector<int> q_points_ik;

    // if () // downsampling
    {
        for (int iq = 0; iq < wfcpw->nks; iq++)
        {
            q_points_ik.push_back(iq);
        }
    }
    // else
    // {    
    //     for (int iq = 0; iq < wfcpw->nks; iq++)
    //     {
    //         kv->
    //     }
    // }

    q_points[ik] = q_points_ik;
    return q_points_ik;
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::multiply_potential(T *density_recip, int ik, int iq) const
{
    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
    int npw = rhopw->npw;
    int nks = wfcpw->nks;

    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int ig = 0; ig < npw; ig++)
    {
        density_recip[ig] *= pot[ik * nks * npw + iq * npw + ig];

    }

    ModuleBase::timer::tick("OperatorEXXPW", "multiply_potential");
}

template <typename T, typename Device>
const T *OperatorEXXPW<T, Device>::get_pw(const int m, const int iq) const
{
    // return pws[iq].get() + m * wfcpw->npwk[iq];
    p_exx_helper->psi.fix_kb(iq, m);
    auto psi_mq = p_exx_helper->psi.get_pointer();
    return psi_mq;
}

template <typename T, typename Device>
template <typename T_in, typename Device_in>
OperatorEXXPW<T, Device>::OperatorEXXPW(const OperatorEXXPW<T_in, Device_in> *op)
{
    // copy all the datas
    this->isk = op->isk;
    this->wfcpw = op->wfcpw;
    this->rhopw = op->rhopw;
    this->psi = op->psi;
    this->ctx = op->ctx;
    this->cpu_ctx = op->cpu_ctx;
    resmem_complex_op()(this->ctx, psi_nk_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, psi_mq_real, wfcpw->nrxx);
    resmem_complex_op()(this->ctx, density_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, h_psi_real, rhopw->nrxx);
    resmem_complex_op()(this->ctx, density_recip, rhopw->npw);
    resmem_complex_op()(this->ctx, h_psi_recip, wfcpw->npwk_max);
    this->pws.resize(wfcpw->nks);


}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::get_potential() const
{
    int nks = wfcpw->nks, npw = rhopw->npw;
    double tpiba2 = tpiba * tpiba;
    // calculate the pot
    for (int ik = 0; ik < nks; ik++)
    {
        for (int iq = 0; iq < nks; iq++)
        {
            auto k = wfcpw->kvec_c[ik];
            auto q = wfcpw->kvec_c[iq];

            #ifdef _OPENMP
            #pragma omp parallel for schedule(static)
            #endif
            for (int ig = 0; ig < rhopw->npw; ig++)
            {
                Real gg = (k - q + rhopw->gcar[ig]).norm2() * tpiba2;
                Real hse_omega2 = GlobalC::exx_info.info_global.hse_omega * GlobalC::exx_info.info_global.hse_omega;
                // if (kqgcar2 > 1e-12) // vasp uses 1/40 of the smallest (k spacing)**2
                if (gg >= 1e-8)
                {
                    Real fac = -ModuleBase::FOUR_PI * ModuleBase::e2 / gg;
                    // if (PARAM.inp.dft_functional == "hse")
                    if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
                    {
                        pot[ik * nks * npw + iq * npw + ig] = fac * (1.0 - std::exp(-gg / 4.0 / hse_omega2));
                    }
                    else
                    {
                        pot[ik * nks * npw + iq * npw + ig] = fac;
                    }
                }
                // }
                else
                {
                    // if (PARAM.inp.dft_functional == "hse")
                    if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
                    {
                        pot[ik * nks * npw + iq * npw + ig] = exx_div - ModuleBase::PI * ModuleBase::e2 / hse_omega2;
                    }
                    else
                    {
                        pot[ik * nks * npw + iq * npw + ig] = exx_div;
                    }
                }
                // assert(is_finite(density_recip[ig]));
            }
        }
    }
}

template <typename T, typename Device>
void OperatorEXXPW<T, Device>::exx_divergence()
{
    if (GlobalC::exx_info.info_lip.lambda == 0.0)
    {
        return;
    }

    // here we follow the exx_divergence subroutine in q-e (PW/src/exx_base.f90)
    double alpha = 10.0 / wfcpw->gk_ecut;
    double tpiba2 = tpiba * tpiba;
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

    // if (PARAM.inp.dft_functional == "hse")
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

    // numerically value the mean value of F(q) in the reciprocal space
    // This means we need to calculate the average of F(q) in the first brillouin zone
    alpha /= tpiba2;
    int nqq = 100000;
    double dq = 5.0 / std::sqrt(alpha) / nqq;
    double aa = 0.0;
    // if (PARAM.inp.dft_functional == "hse")
    if (GlobalC::exx_info.info_global.ccp_type == Conv_Coulomb_Pot_K::Ccp_Type::Erfc)
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

    //    printf("ucell: %p\n", ucell);
    double omega = ucell->omega;
    div -= ModuleBase::e2 * omega * aa;
    exx_div = div * wfcpw->nks;
    // std::cout << "EXX divergence: " << exx_div << std::endl;

    return;

}

template class OperatorEXXPW<std::complex<float>, base_device::DEVICE_CPU>;
template class OperatorEXXPW<std::complex<double>, base_device::DEVICE_CPU>;
#if ((defined __CUDA) || (defined __ROCM))
// to be implemented
#endif

} // namespace hamilt

#endif