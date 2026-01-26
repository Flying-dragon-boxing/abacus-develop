#if defined __MPI

#include "mpi.h"
#include "parallel_global.h"

MPI_Comm POOL_WORLD; //groups for different plane waves. In this group, only plane waves are different. K-points and bands are the same.
MPI_Comm KP_WORLD;   // groups for differnt k. In this group, only k-points are different. Bands and plane waves are the same.
MPI_Comm BP_WORLD;   // groups for differnt bands. In this group, only bands are different. K-points and plane waves are the same.
MPI_Comm INT_BGROUP; // internal comm groups for same bands. In this group, only bands are the same. K-points and plane waves are different.
MPI_Comm GRID_WORLD; // mohan add 2012-01-13
MPI_Comm DIAG_WORLD; // mohan add 2012-01-13

#include <nccl.h>
#include <map>
#include <cuda_runtime.h>

ncclComm_t NCCL_POOL_WORLD;
ncclComm_t NCCL_KP_WORLD;
ncclComm_t NCCL_BP_WORLD;
ncclComm_t NCCL_INT_BGROUP;
ncclComm_t NCCL_GRID_WORLD;
ncclComm_t NCCL_DIAG_WORLD;

std::map<MPI_Comm, ncclComm_t*> mpi_to_nccl_map;

ncclComm_t create_nccl_comm_from_mpi(MPI_Comm mpi_comm)
{
    ncclComm_t nccl_comm;
    int rank, size;
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);

    // Get unique ID from rank 0
    ncclUniqueId id;
    if (rank == 0) {
        ncclGetUniqueId(&id);
    }
    MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, mpi_comm);

    // Each rank binds to one GPU
    int device_id = rank % size;  // This assumes one GPU per MPI rank
    cudaSetDevice(device_id);

    ncclCommInitRank(&nccl_comm, size, id, rank);
    return nccl_comm;
}

void init_nccl_comms()
{
    NCCL_POOL_WORLD = create_nccl_comm_from_mpi(POOL_WORLD);
    NCCL_KP_WORLD = create_nccl_comm_from_mpi(KP_WORLD);
    NCCL_BP_WORLD = create_nccl_comm_from_mpi(BP_WORLD);
    NCCL_INT_BGROUP = create_nccl_comm_from_mpi(INT_BGROUP);
    NCCL_GRID_WORLD = create_nccl_comm_from_mpi(GRID_WORLD);
    NCCL_DIAG_WORLD = create_nccl_comm_from_mpi(DIAG_WORLD);

    // Build mapping
    mpi_to_nccl_map[POOL_WORLD] = &NCCL_POOL_WORLD;
    mpi_to_nccl_map[KP_WORLD] = &NCCL_KP_WORLD;
    mpi_to_nccl_map[BP_WORLD] = &NCCL_BP_WORLD;
    mpi_to_nccl_map[INT_BGROUP] = &NCCL_INT_BGROUP;
    mpi_to_nccl_map[GRID_WORLD] = &NCCL_GRID_WORLD;
    mpi_to_nccl_map[DIAG_WORLD] = &NCCL_DIAG_WORLD;
}

void finalize_nccl_comms()
{
    ncclCommDestroy(NCCL_POOL_WORLD);
    ncclCommDestroy(NCCL_KP_WORLD);
    ncclCommDestroy(NCCL_BP_WORLD);
    ncclCommDestroy(NCCL_INT_BGROUP);
    ncclCommDestroy(NCCL_GRID_WORLD);
    ncclCommDestroy(NCCL_DIAG_WORLD);
    mpi_to_nccl_map.clear();
}

ncclComm_t get_nccl_comm(MPI_Comm mpi_comm)
{
    auto it = mpi_to_nccl_map.find(mpi_comm);
    if (it != mpi_to_nccl_map.end()) {
        return *(it->second);
    }
    // If not found, create a new one
    return create_nccl_comm_from_mpi(mpi_comm);
}

MPICommGroup::MPICommGroup(MPI_Comm parent_comm)
    : parent_comm(parent_comm)
{
    MPI_Comm_size(parent_comm, &this->gsize);
    MPI_Comm_rank(parent_comm, &this->grank);
}

MPICommGroup::~MPICommGroup()
{
    if (group_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&group_comm);
    }
    if (inter_comm != MPI_COMM_NULL)
    {
        MPI_Comm_free(&inter_comm);
    }
}

void MPICommGroup::divide_group_comm(const int& ngroup, const bool assert_even)
{
    this->ngroups = ngroup;
    Parallel_Global::divide_mpi_groups(this->gsize,
                                       ngroup,
                                       this->grank,
                                       this->nprocs_in_group,
                                       this->my_group,
                                       this->rank_in_group,
                                       assert_even);

    MPI_Comm_split(parent_comm, my_group, rank_in_group, &group_comm);
    if(this->gsize % ngroup == 0)
    {
        this->is_even = true;
    }

    if (this->is_even)
    {
        MPI_Comm_split(parent_comm, my_inter, rank_in_inter, &inter_comm);
    }
}

#endif