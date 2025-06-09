//
// Created by rhx on 25-6-3.
//
#include "k_vector_utils.h"

#include "klist.h"
#include "module_base/global_variable.h"
#include "module_base/matrix3.h"

#include <module_base/formatter.h>
#include <module_parameter/parameter.h>

namespace KVectorUtils
{
void k_vec_d2c(K_Vectors& kv, const ModuleBase::Matrix3& reciprocal_vec)
{
    if (kv.kvec_d.size() != kv.kvec_c.size())
    {
        ModuleBase::WARNING_QUIT("k_vec_d2c", "Size of Cartesian and Direct K vectors mismatch. ");
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
void k_vec_c2d(K_Vectors& kv, const ModuleBase::Matrix3& latvec)
{
    if (kv.kvec_d.size() != kv.kvec_c.size())
    {
        ModuleBase::WARNING_QUIT("k_vec_c2d", "Size of Cartesian and Direct K vectors mismatch. ");
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
    if (PARAM.inp.final_scf) // Todo: Any way to avoid directly using input variables?
    {
        if (kv.k_nkstot == 0)
        {
            kv.kd_done = true;
            kv.kc_done = false;
        }
        else
        {
            if (kv.k_kword == "Cartesian" || kv.k_kword == "C")
            {
                kv.kc_done = true;
                kv.kd_done = false;
            }
            else if (kv.k_kword == "Direct" || kv.k_kword == "D")
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
        KVectorUtils::k_vec_d2c(kv, G);
        kv.kc_done = true;
    }

    // set direct k vectors
    else if (kv.kc_done && !kv.kd_done)
    {
        KVectorUtils::k_vec_c2d(kv, R);
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
    kv.nspin = nspin_in;
    ModuleBase::GlobalFunc::OUT(GlobalV::ofs_running, "nspin", kv.nspin);

    // set cartesian k vectors.
    KVectorUtils::k_vec_d2c(kv, reciprocal_vec);

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

    kv.print_klists(GlobalV::ofs_running);
}
} // namespace KVectorUtils
