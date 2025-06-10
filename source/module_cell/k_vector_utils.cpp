//
// Created by rhx on 25-6-3.
//
#include "k_vector_utils.h"

#include "klist.h"
#include "module_base/global_variable.h"
#include "module_base/matrix3.h"

#include <module_base/formatter.h>
#include <module_parameter/parameter.h>
#include <module_base/parallel_common.h>
#include <module_base/parallel_reduce.h>

namespace KVectorUtils
{
void kvec_d2c(K_Vectors& kv, const ModuleBase::Matrix3& reciprocal_vec)
{
//    throw std::runtime_error("k_vec_d2c: This function is not implemented in the new codebase. Please use the new implementation.");
    if (kv.kvec_d.size() != kv.kvec_c.size())
    {
//        ModuleBase::WARNING_QUIT("k_vec_d2c", "Size of Cartesian and Direct K vectors mismatch. ");
        kv.kvec_c.resize(kv.kvec_d.size());
    }
    int nks = kv.kvec_d.size(); // always convert all k vectors

    for (int i = 0; i < nks; i++)
    {
        // wrong!!   kvec_c[i] = G * kvec_d[i];
        //  mohan fixed bug 2010-1-10
        if (std::abs(kv.kvec_d[i].x) < 1.0e-10)
        {
            kv.kvec_d[i].x = 0.0;
        }
        if (std::abs(kv.kvec_d[i].y) < 1.0e-10)
        {
            kv.kvec_d[i].y = 0.0;
        }
        if (std::abs(kv.kvec_d[i].z) < 1.0e-10)
        {
            kv.kvec_d[i].z = 0.0;
        }

        kv.kvec_c[i] = kv.kvec_d[i] * reciprocal_vec;

        // mohan add2012-06-10
        if (std::abs(kv.kvec_c[i].x) < 1.0e-10)
        {
            kv.kvec_c[i].x = 0.0;
        }
        if (std::abs(kv.kvec_c[i].y) < 1.0e-10)
        {
            kv.kvec_c[i].y = 0.0;
        }
        if (std::abs(kv.kvec_c[i].z) < 1.0e-10)
        {
            kv.kvec_c[i].z = 0.0;
        }
    }
}
void kvec_c2d(K_Vectors& kv, const ModuleBase::Matrix3& latvec)
{
    if (kv.kvec_d.size() != kv.kvec_c.size())
    {
        kv.kvec_d.resize(kv.kvec_c.size());
    }
    int nks = kv.kvec_d.size(); // always convert all k vectors

    ModuleBase::Matrix3 RT = latvec.Transpose();
    for (int i = 0; i < nks; i++)
    {
        //			std::cout << " ik=" << i
        //				<< " kvec.x=" << kvec_c[i].x
        //				<< " kvec.y=" << kvec_c[i].y
        //				<< " kvec.z=" << kvec_c[i].z << std::endl;
        // wrong!            kvec_d[i] = RT * kvec_c[i];
        // mohan fixed bug 2011-03-07
        kv.kvec_d[i] = kv.kvec_c[i] * RT;
    }
}

void set_both_kvec(K_Vectors& kv, const ModuleBase::Matrix3& G, const ModuleBase::Matrix3& R, std::string& skpt)
{
    if (true) // Originally GlobalV::FINAL_SCF, but we don't have this variable in the new code.
    {
        if (kv.get_k_nkstot() == 0)
        {
            kv.kd_done = true;
            kv.kc_done = false;
        }
        else
        {
            if (kv.get_k_kword() == "Cartesian" || kv.get_k_kword() == "C")
            {
                kv.kc_done = true;
                kv.kd_done = false;
            }
            else if (kv.get_k_kword() == "Direct" || kv.get_k_kword() == "D")
            {
                kv.kd_done = true;
                kv.kc_done = false;
            }
            else
            {
                GlobalV::ofs_warning << " Error : neither Cartesian nor Direct kpoint." << std::endl;
            }
        }
    }

    // set cartesian k vectors.
    if (!kv.kc_done && kv.kd_done)
    {
        KVectorUtils::kvec_d2c(kv, G);
        kv.kc_done = true;
    }

    // set direct k vectors
    else if (kv.kc_done && !kv.kd_done)
    {
        KVectorUtils::kvec_c2d(kv, R);
        kv.kd_done = true;
    }
    std::string table;
    table += " K-POINTS DIRECT COORDINATES\n";
    table += FmtCore::format("%8s%12s%12s%12s%8s\n", "KPOINTS", "DIRECT_X", "DIRECT_Y", "DIRECT_Z", "WEIGHT");
    for (int i = 0; i < kv.get_nkstot(); i++)
    {
        table += FmtCore::format("%8d%12.8f%12.8f%12.8f%8.4f\n",
                                 i + 1,
                                 kv.kvec_d[i].x,
                                 kv.kvec_d[i].y,
                                 kv.kvec_d[i].z,
                                 kv.wk[i]);
    }
    GlobalV::ofs_running << table << std::endl;
    if (GlobalV::MY_RANK == 0)
    {
        std::stringstream ss;
        ss << " " << std::setw(40) << "nkstot now"
           << " = " << kv.get_nkstot() << std::endl;
        ss << table << std::endl;
        skpt = ss.str();
    }
    return;
}

void set_after_vc(K_Vectors& kv, const int& nspin_in, const ModuleBase::Matrix3& reciprocal_vec)
{
    GlobalV::ofs_running << "\n SETUP K-POINTS" << std::endl;
//    kv.nspin = nspin_in;
    kv.set_nspin(nspin_in);
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "nspin", kv.get_nspin());

    // set cartesian k vectors.
    KVectorUtils::kvec_d2c(kv, reciprocal_vec);

    std::string table;
    table += "K-POINTS DIRECT COORDINATES\n";
    table += FmtCore::format("%8s%12s%12s%12s%8s\n", "KPOINTS", "DIRECT_X", "DIRECT_Y", "DIRECT_Z", "WEIGHT");
    for (int i = 0; i < kv.get_nks(); i++)
    {
        table += FmtCore::format("%8d%12.8f%12.8f%12.8f%8.4f\n",
                                 i + 1,
                                 kv.kvec_d[i].x,
                                 kv.kvec_d[i].y,
                                 kv.kvec_d[i].z,
                                 kv.wk[i]);
    }
    GlobalV::ofs_running << table << std::endl;

    kv.kd_done = true;
    kv.kc_done = true;

    print_klists(kv, GlobalV::ofs_running);
}

void print_klists(const K_Vectors& kv, std::ofstream& ofs)
{
    ModuleBase::TITLE("KVectorUtils", "print_klists");
    int nks = kv.get_nks();
    int nkstot = kv.get_nkstot();

    if (nkstot < nks)
    {
        std::cout << "\n nkstot=" << nkstot;
        std::cout << "\n nks=" << nks;
        ModuleBase::WARNING_QUIT("print_klists", "nkstot < nks");
    }
    std::string table;
    table += " K-POINTS CARTESIAN COORDINATES\n";
    table += FmtCore::format("%8s%12s%12s%12s%8s\n", "KPOINTS", "CARTESIAN_X", "CARTESIAN_Y", "CARTESIAN_Z", "WEIGHT");
    for (int i = 0; i < nks; i++)
    {
        table += FmtCore::format("%8d%12.8f%12.8f%12.8f%8.4f\n",
                                 i + 1,
                                 kv.kvec_c[i].x,
                                 kv.kvec_c[i].y,
                                 kv.kvec_c[i].z,
                                 kv.wk[i]);
    }
    GlobalV::ofs_running << "\n" << table << std::endl;

    table.clear();
    table += " K-POINTS DIRECT COORDINATES\n";
    table += FmtCore::format("%8s%12s%12s%12s%8s\n", "KPOINTS", "DIRECT_X", "DIRECT_Y", "DIRECT_Z", "WEIGHT");
    for (int i = 0; i < nks; i++)
    {
        table += FmtCore::format("%8d%12.8f%12.8f%12.8f%8.4f\n",
                                 i + 1,
                                 kv.kvec_d[i].x,
                                 kv.kvec_d[i].y,
                                 kv.kvec_d[i].z,
                                 kv.wk[i]);
    }
    GlobalV::ofs_running << "\n" << table << std::endl;
    return;
}

#ifdef __MPI
void kvec_mpi_k(K_Vectors& kv)
{
    ModuleBase::TITLE("KVectorUtils", "kvec_mpi_k");

    Parallel_Common::bcast_bool(kv.kc_done);

    Parallel_Common::bcast_bool(kv.kd_done);

    Parallel_Common::bcast_int(kv.nspin);

    Parallel_Common::bcast_int(kv.nkstot);

    Parallel_Common::bcast_int(kv.nkstot_full);

    Parallel_Common::bcast_int(kv.nmp, 3);

    kv.kl_segids.resize(kv.nkstot);
    Parallel_Common::bcast_int(kv.kl_segids.data(), kv.nkstot);

    Parallel_Common::bcast_double(kv.koffset, 3);

    kv.nks = kv.para_k.nks_pool[GlobalV::MY_POOL];

    GlobalV::ofs_running << std::endl;
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "k-point number in this process", kv.nks);
    int nks_minimum = kv.nks;

    Parallel_Reduce::gather_min_int_all(GlobalV::NPROC, nks_minimum);

    if (nks_minimum == 0)
    {
        ModuleBase::WARNING_QUIT("K_Vectors::mpi_k()", " nks == 0, some processor have no k point!");
    }
    else
    {
        ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "minimum distributed K point number", nks_minimum);
    }

    std::vector<int> isk_aux(kv.nkstot);
    std::vector<double> wk_aux(kv.nkstot);
    std::vector<double> kvec_c_aux(kv.nkstot * 3);
    std::vector<double> kvec_d_aux(kv.nkstot * 3);

    // collect and process in rank 0
    if (GlobalV::MY_RANK == 0)
    {
        for (int ik = 0; ik < kv.nkstot; ik++)
        {
            isk_aux[ik] = kv.isk[ik];
            wk_aux[ik] = kv.wk[ik];
            kvec_c_aux[3 * ik] = kv.kvec_c[ik].x;
            kvec_c_aux[3 * ik + 1] = kv.kvec_c[ik].y;
            kvec_c_aux[3 * ik + 2] = kv.kvec_c[ik].z;
            kvec_d_aux[3 * ik] = kv.kvec_d[ik].x;
            kvec_d_aux[3 * ik + 1] = kv.kvec_d[ik].y;
            kvec_d_aux[3 * ik + 2] = kv.kvec_d[ik].z;
        }
    }

    // broadcast k point data to all processors
    Parallel_Common::bcast_int(isk_aux.data(), kv.nkstot);

    Parallel_Common::bcast_double(wk_aux.data(), kv.nkstot);
    Parallel_Common::bcast_double(kvec_c_aux.data(), kv.nkstot * 3);
    Parallel_Common::bcast_double(kvec_d_aux.data(), kv.nkstot * 3);

    // process k point data in each processor
    kv.renew(kv.nks * kv.nspin);

    // distribute
    int k_index = 0;

    for (int i = 0; i < kv.nks; i++)
    {
        // 3 is because each k point has three value:kx, ky, kz
        k_index = i + kv.para_k.startk_pool[GlobalV::MY_POOL];
        kv.kvec_c[i].x = kvec_c_aux[k_index * 3];
        kv.kvec_c[i].y = kvec_c_aux[k_index * 3 + 1];
        kv.kvec_c[i].z = kvec_c_aux[k_index * 3 + 2];
        kv.kvec_d[i].x = kvec_d_aux[k_index * 3];
        kv.kvec_d[i].y = kvec_d_aux[k_index * 3 + 1];
        kv.kvec_d[i].z = kvec_d_aux[k_index * 3 + 2];
        kv.wk[i] = wk_aux[k_index];
        kv.isk[i] = isk_aux[k_index];
    }

#ifdef __EXX
    if (ModuleSymmetry::Symmetry::symm_flag == 1)
    {//bcast kstars
        kv.kstars.resize(kv.nkstot);
        for (int ikibz = 0;ikibz < kv.nkstot;++ikibz)
        {
            int starsize = kv.kstars[ikibz].size();
            Parallel_Common::bcast_int(starsize);
            GlobalV::ofs_running << "starsize: " << starsize << std::endl;
            auto ks = kv.kstars[ikibz].begin();
            for (int ik = 0;ik < starsize;++ik)
            {
                int isym = 0;
                ModuleBase::Vector3<double> ks_vec(0, 0, 0);
                if (GlobalV::MY_RANK == 0)
                {
                    isym = ks->first;
                    ks_vec = ks->second;
                    ++ks;
                }
                Parallel_Common::bcast_int(isym);
                Parallel_Common::bcast_double(ks_vec.x);
                Parallel_Common::bcast_double(ks_vec.y);
                Parallel_Common::bcast_double(ks_vec.z);
                GlobalV::ofs_running << "isym: " << isym << " ks_vec: " << ks_vec.x << " " << ks_vec.y << " " << ks_vec.z << std::endl;
                if (GlobalV::MY_RANK != 0)
                {
                    kv.kstars[ikibz].insert(std::make_pair(isym, ks_vec));
                }
            }
        }
    }
#endif
} // END SUBROUTINE
#endif
} // namespace KVectorUtils
