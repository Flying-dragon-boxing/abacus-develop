#include <cstdio>
#include <iostream>
#ifdef __PEXSI
#include "dist_ccs_matrix.h"

#include <mpi.h>

namespace pexsi
{
DistCCSMatrix::DistCCSMatrix(void)
{
    this->comm = MPI_COMM_WORLD;
    this->size = 0;
    this->nnz = 0;
    this->nnzLocal = 0;
    this->numColLocal = 0;
    this->colptrLocal = nullptr;
    this->rowindLocal = nullptr;
}

DistCCSMatrix::DistCCSMatrix(MPI_Comm comm_in)
{
    this->comm = comm_in;
    this->size = 0;
    this->nnz = 0;
    this->nnzLocal = 0;
    this->numColLocal = 0;
    this->colptrLocal = nullptr;
    this->rowindLocal = nullptr;
}

DistCCSMatrix::DistCCSMatrix(int size_in, int nnzLocal_in)
{
    this->comm = MPI_COMM_WORLD;
    this->size = size_in;
    this->nnzLocal = nnzLocal_in;
    MPI_Request req;
    MPI_Iallreduce(&nnzLocal, &this->nnz, 1, MPI_INT, MPI_SUM, this->comm, &req);
    this->numColLocal = 0;
    this->colptrLocal = new int[size];
    this->rowindLocal = new int[nnzLocal];

    MPI_Status req_status;
    MPI_Wait(&req, &req_status);
}

DistCCSMatrix::DistCCSMatrix(MPI_Comm comm_in, int nproc_data_in, int size_in)
{
    this->comm = comm_in;
    this->nproc_data = nproc_data_in;
    int nproc_data_range[3] = {0, this->nproc_data - 1, 1};
    // create processes group with data: this->group_data and associated communicator
    MPI_Comm_group(this->comm, &this->group);
    MPI_Group_range_incl(this->group, 1, &nproc_data_range, &this->group_data);
    this->comm_data = MPI_COMM_NULL;
    MPI_Comm_create(this->comm, this->group_data, &this->comm_data);
    int identical = 0;
    MPI_Comm_compare(comm_in, this->comm_data, &identical);
    // printf("identical = %d\n", identical);
    this->size = size_in;
    this->nnz = 0;
    this->nnzLocal = 0;
    int myproc;
    if (comm != MPI_COMM_NULL)
    {
        MPI_Comm_size(comm, &nprocs);
        MPI_Comm_rank(comm, &myproc);
        if (myproc < nproc_data - 1)
        {
            this->numColLocal = size / nproc_data;
            this->firstCol = size / nproc_data * myproc;
            this->colptrLocal = new int[this->numColLocal + 1];
            this->rowindLocal = nullptr;
        }
        else if (myproc == nproc_data - 1)
        {
            this->numColLocal = size - myproc * (size / nproc_data);
            this->firstCol = size / nproc_data * myproc;
            this->colptrLocal = new int[this->numColLocal + 1];
            this->rowindLocal = nullptr;
        }
        else
        {
            this->numColLocal = 0;
            this->firstCol = size - 1;
            this->colptrLocal = new int[this->numColLocal + 1];
            this->rowindLocal = nullptr;
        }
    }
}

int DistCCSMatrix::globalCol(int localCol)
{
    return this->firstCol + localCol;
}

// NOTE: the process id is 0-based
int DistCCSMatrix::localCol(int globalCol, int& mypcol)
{
    mypcol = int(globalCol / int(this->size / this->nproc_data));
    if (mypcol >= this->nproc_data)
        mypcol = this->nproc_data - 1;

    return mypcol > 0 ? globalCol - (this->size / this->nproc_data) * mypcol : globalCol;
}

void DistCCSMatrix::setnnz(int nnzLocal_in)
{
    if (this->comm_data != MPI_COMM_NULL)
    {
        MPI_Allreduce(&nnzLocal_in, &this->nnz, 1, MPI_INT, MPI_SUM, this->comm_data);
        this->nnzLocal = nnzLocal_in;
        this->rowindLocal = new int[nnzLocal];
        this->colptrLocal[this->numColLocal] = nnzLocal_in + 1;
    }
}

void DistCCSMatrix::group_broadcast(double *&H, double *&S)
{
    // Broadcast all data of DistCCSMatrix across MPI_COMM_WORLD, source are the first nproc_data processes
    // from i < nproc_data, the process i will send data to process i+j*nproc_data
    int rank_world, size_world;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank_world);
    MPI_Comm_size(MPI_COMM_WORLD, &size_world);
    MPI_Comm_rank(this->comm, &rank); // rank is used as color
    MPI_Comm_size(this->comm, &size);
    MPI_Comm comm_same_rank;
    MPI_Comm_split(MPI_COMM_WORLD, rank, rank_world, &comm_same_rank);
    int rank_base;
    MPI_Group world_group, same_rank_group;
    MPI_Comm_group(MPI_COMM_WORLD, &world_group);
    MPI_Comm_group(comm_same_rank, &same_rank_group);
    MPI_Group_translate_ranks(world_group, 1, &rank, same_rank_group, &rank_base);
    // broadcast literally everything
    std::cout << "size = " << size << ", rank = " << rank << ", rank_base = " << rank_base << std::endl;
    std::cout << "nnz = " << nnz << ", nnzLocal = " << nnzLocal << ", numColLocal = " << numColLocal << ", firstCol = " << firstCol << std::endl;
    MPI_Bcast(&this->size, 1, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(&this->nnz, 1, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(&this->nnzLocal, 1, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(&this->numColLocal, 1, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(&this->firstCol, 1, MPI_INT, rank_base, comm_same_rank);
    if (rank_world >= this->nproc_data)
    {
        if (colptrLocal != nullptr)
        {
            delete[] colptrLocal;
            colptrLocal = nullptr;
        }
        if (rowindLocal != nullptr)
        {
            delete[] rowindLocal;
            rowindLocal = nullptr;
        }
        if (H != nullptr)
        {
            delete[] H;
            H = nullptr;
        }
        if (S != nullptr)
        {
            delete[] S;
            S = nullptr;
        }

        colptrLocal = new int[numColLocal + 1];
        rowindLocal = new int[nnzLocal];
        H = new double[nnzLocal];
        S = new double[nnzLocal];

    }
    MPI_Bcast(this->colptrLocal, this->numColLocal + 1, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(this->rowindLocal, this->nnzLocal, MPI_INT, rank_base, comm_same_rank);
    MPI_Bcast(H, this->nnzLocal, MPI_DOUBLE, rank_base, comm_same_rank);
    MPI_Bcast(S, this->nnzLocal, MPI_DOUBLE, rank_base, comm_same_rank);

    MPI_Group_free(&world_group);
    MPI_Group_free(&same_rank_group);
}

DistCCSMatrix::~DistCCSMatrix()
{
    delete[] colptrLocal;
    delete[] rowindLocal;
}
} // namespace pexsi
#endif