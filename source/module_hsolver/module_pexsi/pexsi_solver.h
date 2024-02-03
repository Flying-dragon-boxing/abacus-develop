#ifndef PEXSI_Solver_H
#define PEXSI_Solver_H

#include <vector>

namespace pexsi
{
class PEXSI_Solver
{
  public:
    PEXSI_Solver(const int blacs_text,
                 const int nb,
                 const int nrow,
                 const int ncol,
                 const double* h,
                 const double* s,
                 double& totalEnergyH,
                 double& totalEnergyS,
                 double& totalFreeEnergy);
    int solve();
    double* get_DM() const;
    double* get_EDM() const;
    const double get_totalFreeEnergy() const;
    const double get_totalEnergyH() const;
    const double get_totalEnergyS() const;

  private:
    int blacs_text;
    int nb;
    int nrow;
    int ncol;
    double* h;
    double* s;
    double* DM;
    double* EDM;
    double totalEnergyH;
    double totalEnergyS;
    double totalFreeEnergy;
};
} // namespace pexsi
#endif // PEXSI_Solver_H