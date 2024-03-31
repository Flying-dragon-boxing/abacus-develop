#include "module_base/parallel_global.h"
#ifdef __PEXSI
#include "pexsi_solver.h"

#include <mpi.h>
#include <cstring>

#include "module_base/global_variable.h"
#include "simple_pexsi.h"

extern MPI_Comm DIAG_WORLD;
namespace pexsi
{

int PEXSI_Solver::pexsi_npole = 0;
bool PEXSI_Solver::pexsi_inertia = 0;
int PEXSI_Solver::pexsi_nmax = 0;
// int PEXSI_Solver::pexsi_symbolic = 0;
bool PEXSI_Solver::pexsi_comm = 0;
bool PEXSI_Solver::pexsi_storage = 0;
int PEXSI_Solver::pexsi_ordering = 0;
int PEXSI_Solver::pexsi_row_ordering = 0;
int PEXSI_Solver::pexsi_nproc = 0;
bool PEXSI_Solver::pexsi_symm = 0;
bool PEXSI_Solver::pexsi_trans = 0;
int PEXSI_Solver::pexsi_method = 0;
int PEXSI_Solver::pexsi_nproc_pole = 0;
// double PEXSI_Solver::pexsi_spin = 2;
double PEXSI_Solver::pexsi_temp = 0.0;
double PEXSI_Solver::pexsi_gap = 0.0;
double PEXSI_Solver::pexsi_delta_e = 0.0;
double PEXSI_Solver::pexsi_mu_lower = 0.0;
double PEXSI_Solver::pexsi_mu_upper = 0.0;
double PEXSI_Solver::pexsi_mu = 0.0;
double PEXSI_Solver::pexsi_mu_thr = 0.0;
double PEXSI_Solver::pexsi_mu_expand = 0.0;
double PEXSI_Solver::pexsi_mu_guard = 0.0;
double PEXSI_Solver::pexsi_elec_thr = 0.0;
double PEXSI_Solver::pexsi_zero_thr = 0.0;

PEXSI_Solver::PEXSI_Solver(const int blacs_text,
                           const int nb,
                           const int nrow,
                           const int ncol,
                           const double* h,
                           const double* s,
                           double& totalEnergyH,
                           double& totalEnergyS,
                           double& totalFreeEnergy)
{
    this->blacs_text = blacs_text;
    this->nb = nb;
    this->nrow = nrow;
    this->ncol = ncol;
    this->h = new double[nrow * ncol];
    this->s = new double[nrow * ncol];
    std::memcpy(this->h, h, nrow * ncol * sizeof(double));
    std::memcpy(this->s, s, nrow * ncol * sizeof(double));
    this->DM = new double[nrow * ncol];
    this->EDM = new double[nrow * ncol];
    this->totalEnergyH = 0.0;
    this->totalEnergyS = 0.0;
    this->totalFreeEnergy = 0.0;
}

int PEXSI_Solver::solve(double mu0)
{
    MPI_Group grid_group;
    int myid, grid_np;
    MPI_Group world_group;
    MPI_Comm_rank(DIAG_WORLD, &myid);
    MPI_Comm_size(DIAG_WORLD, &grid_np);
    MPI_Comm_group(DIAG_WORLD, &world_group);

    int grid_proc_range[3]={0, (GlobalV::NPROC/grid_np)*grid_np-1, GlobalV::NPROC/grid_np};
    MPI_Group_range_incl(world_group, 1, &grid_proc_range, &grid_group);

    simplePEXSI(DIAG_WORLD,
                DIAG_WORLD,
                grid_group,
                this->blacs_text,
                GlobalV::NLOCAL,
                this->nb,
                this->nrow,
                this->ncol,
                'C',
                this->h,
                this->s,
                GlobalV::nelec,
                "PEXSIOPTION",
                this->DM,
                this->EDM,
                this->totalEnergyH,
                this->totalEnergyS,
                this->totalFreeEnergy,
                mu,
                mu0);
    return 0;
}

double* PEXSI_Solver::get_DM() const
{
    return DM;
}

double* PEXSI_Solver::get_EDM() const
{
    return EDM;
}

const double PEXSI_Solver::get_totalFreeEnergy() const
{
    return totalFreeEnergy;
}

const double PEXSI_Solver::get_totalEnergyH() const
{
    return totalEnergyH;
}

const double PEXSI_Solver::get_totalEnergyS() const
{
    return totalEnergyS;
}

const double PEXSI_Solver::get_mu() const
{
    return mu;
}

void PEXSI_Solver::set_pexsi_vars()
{
    pexsi_npole = 40;
    pexsi_inertia = true;
    pexsi_nmax = 80;
    // pexsi_symbolic = 1;
    pexsi_comm = true;
    pexsi_storage = true;
    pexsi_ordering = 0;
    pexsi_row_ordering = 1;
    pexsi_nproc = 1;
    pexsi_symm = true;
    pexsi_trans = false;
    pexsi_method = 1;
    pexsi_nproc_pole = 1;
    // pexsi_spin = 2;
    pexsi_temp = 0.015;
    pexsi_gap = 0;
    pexsi_delta_e = 20.0;
    pexsi_mu_lower = -10;
    pexsi_mu_upper = 10;
    pexsi_mu = 0.0;
    pexsi_mu_thr = 0.05;
    pexsi_mu_expand = 0.3;
    pexsi_mu_guard = 0.2;
    pexsi_elec_thr = 0.001;
    pexsi_zero_thr = 1e-10;
    return;
}

} // namespace pexsi
#endif