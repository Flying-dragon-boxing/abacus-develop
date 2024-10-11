#include "global.h"
//----------------------------------------------------------
// init "GLOBAL CLASS" object
//----------------------------------------------------------
namespace GlobalC
{
#ifdef __EXX
    Exx_Info exx_info;
    Exx_Helper exx_helper;
    bool Exx_Helper::exx_after_converge(int &iter)
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
#endif
pseudopot_cell_vnl ppcell;
UnitCell ucell;
Parallel_Grid Pgrid; //mohan add 2010-06-06 
Parallel_Kpoints Pkpoints; // mohan add 2010-06-07
Restart restart; // Peize Lin add 2020.04.04
}

//Magnetism mag;															
