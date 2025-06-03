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

void set_both_kvec(K_Vectors& kv, const ModuleBase::Matrix3& G, const ModuleBase::Matrix3& R, std::string& skpt);

void set_after_vc(K_Vectors& kv, const int& nspin, const ModuleBase::Matrix3& G);
} // namespace KVectorUtils

#endif // K_VECTOR_UTILS_H
