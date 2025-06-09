//
// Created by rhx on 25-6-3.
//

#ifndef K_VECTOR_UTILS_H
#define K_VECTOR_UTILS_H

#include "module_base/global_variable.h"
#include "module_base/matrix3.h"

class K_Vectors;

namespace KVectorUtils
{
void k_vec_d2c(K_Vectors& kv, const ModuleBase::Matrix3& reciprocal_vec);

void k_vec_c2d(K_Vectors& kv, const ModuleBase::Matrix3& latvec);

/**
     * @brief Sets both the direct and Cartesian k-vectors.
     *
     * This function sets both the direct and Cartesian k-vectors based on the input parameters.
     * It also checks the k-point type and sets the corresponding flags.
     *
     * @param kv The K_Vectors object containing the k-point information.
     * @param G The reciprocal lattice matrix.
     * @param R The real space lattice matrix.
     * @param skpt A string to store the k-point table.
     *
     * @return void
     *
     * @note If the k-point type is neither "Cartesian" nor "Direct", an error message will be printed.
     * @note The function sets the flags kd_done and kc_done to indicate whether the direct and Cartesian k-vectors have
     * been set, respectively.
     * @note The function also prints a table of the direct k-vectors and their weights.
     * @note If the function is called by the master process (MY_RANK == 0), the k-point table is also stored in the
     * string skpt.
 */
void set_both_kvec(K_Vectors& kv, const ModuleBase::Matrix3& G, const ModuleBase::Matrix3& R, std::string& skpt);

/**
     * @brief Sets up the k-points after a volume change.
     *
     * This function sets up the k-points after a volume change in the system.
     * It sets the Cartesian and direct k-vectors based on the new reciprocal and real space lattice vectors.
     *
     * @param kv The K_Vectors object containing the k-point information.
     * @param nspin_in The number of spins. 1 for non-spin-polarized calculations and 2 for spin-polarized calculations.
     * @param reciprocal_vec The new reciprocal lattice matrix.
     *
     * @return void
     *
     * @note The function first sets the number of spins (nspin) to the input value.
     * @note The direct k-vectors have been set (kd_done = true) but the Cartesian k-vectors have not (kc_done =
     * false) after a volume change. The function calculates the Cartesian k-vectors by multiplying the direct k-vectors with the reciprocal
     * lattice matrix.
     * @note The function also prints a table of the direct k-vectors and their weights.
     * @note The function calls the print_klists function to print the k-points in both Cartesian and direct
     * coordinates.
 */
void set_after_vc(K_Vectors& kv, const int& nspin, const ModuleBase::Matrix3& G);
} // namespace KVectorUtils

#endif // K_VECTOR_UTILS_H
