#include "parallel_device.h"
#include <complex>
#ifdef __MPI
namespace Parallel_Common
{
void isend_data(const double* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_DOUBLE, dest, tag, comm, request);
}
void isend_data(const std::complex<double>* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_DOUBLE_COMPLEX, dest, tag, comm, request);
}
void isend_data(const float* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_FLOAT, dest, tag, comm, request);
}
void isend_data(const std::complex<float>* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request)
{
    MPI_Isend(buf, count, MPI_COMPLEX, dest, tag, comm, request);
}
void send_data(const double* buf, int count, int dest, int tag, MPI_Comm& comm)
{
    MPI_Send(buf, count, MPI_DOUBLE, dest, tag, comm);
}
void send_data(const std::complex<double>* buf, int count, int dest, int tag, MPI_Comm& comm)
{
    MPI_Send(buf, count, MPI_DOUBLE_COMPLEX, dest, tag, comm);
}
void send_data(const float* buf, int count, int dest, int tag, MPI_Comm& comm)
{
    MPI_Send(buf, count, MPI_FLOAT, dest, tag, comm);
}
void send_data(const std::complex<float>* buf, int count, int dest, int tag, MPI_Comm& comm)
{
    MPI_Send(buf, count, MPI_COMPLEX, dest, tag, comm);
}
void recv_data(double* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status)
{
    MPI_Recv(buf, count, MPI_DOUBLE, source, tag, comm, status);
}
void recv_data(std::complex<double>* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status)
{
    MPI_Recv(buf, count, MPI_DOUBLE_COMPLEX, source, tag, comm, status);
}
void recv_data(float* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status)
{
    MPI_Recv(buf, count, MPI_FLOAT, source, tag, comm, status);
}
void recv_data(std::complex<float>* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status)
{
    MPI_Recv(buf, count, MPI_COMPLEX, source, tag, comm, status);
}
void bcast_data(std::complex<double>* object, const int& n, const MPI_Comm& comm)
{
    MPI_Bcast(object, n * 2, MPI_DOUBLE, 0, comm);
}
void bcast_data(std::complex<float>* object, const int& n, const MPI_Comm& comm)
{
    MPI_Bcast(object, n * 2, MPI_FLOAT, 0, comm);
}
void bcast_data(double* object, const int& n, const MPI_Comm& comm)
{
    MPI_Bcast(object, n, MPI_DOUBLE, 0, comm);
}
void bcast_data(float* object, const int& n, const MPI_Comm& comm)
{
    MPI_Bcast(object, n, MPI_FLOAT, 0, comm);
}
void reduce_data(std::complex<double>* object, const int& n, const MPI_Comm& comm)
{
    MPI_Allreduce(MPI_IN_PLACE, object, n * 2, MPI_DOUBLE, MPI_SUM, comm);
}
void reduce_data(std::complex<float>* object, const int& n, const MPI_Comm& comm)
{
    MPI_Allreduce(MPI_IN_PLACE, object, n * 2, MPI_FLOAT, MPI_SUM, comm);
}
void reduce_data(double* object, const int& n, const MPI_Comm& comm)
{
    MPI_Allreduce(MPI_IN_PLACE, object, n, MPI_DOUBLE, MPI_SUM, comm);
}
void reduce_data(float* object, const int& n, const MPI_Comm& comm)
{
    MPI_Allreduce(MPI_IN_PLACE, object, n, MPI_FLOAT, MPI_SUM, comm);
}
void gatherv_data(const double* sendbuf, int sendcount, double* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE, recvbuf, recvcounts, displs, MPI_DOUBLE, comm);
}
void gatherv_data(const std::complex<double>* sendbuf, int sendcount, std::complex<double>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_DOUBLE_COMPLEX, recvbuf, recvcounts, displs, MPI_DOUBLE_COMPLEX, comm);
}
void gatherv_data(const float* sendbuf, int sendcount, float* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_FLOAT, recvbuf, recvcounts, displs, MPI_FLOAT, comm);
}
void gatherv_data(const std::complex<float>* sendbuf, int sendcount, std::complex<float>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm)
{
    MPI_Allgatherv(sendbuf, sendcount, MPI_COMPLEX, recvbuf, recvcounts, displs, MPI_COMPLEX, comm);
}

#include <nccl.h>
#include <cuda_runtime.h>
#include "parallel_comm.h"

void gatherv_nccl(const double* sendbuf, int sendcount, double* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm, ncclComm_t nccl_comm, cudaStream_t stream)
{
    int size;
    MPI_Comm_size(comm, &size);
    if (size <= 0) return;

    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > max_count) max_count = recvcounts[i];
    }
    if (max_count == 0) return;

    double *tmp_send = nullptr, *tmp_recv = nullptr;
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(tmp_send, max_count);
    base_device::memory::resize_memory_op<double, base_device::DEVICE_GPU>()(tmp_recv, size * max_count);

    if (sendcount > 0) {
        cudaMemcpyAsync(tmp_send, sendbuf, sendcount * sizeof(double), cudaMemcpyDeviceToDevice, stream);
    }

    ncclAllGather(tmp_send, tmp_recv, max_count, ncclDouble, nccl_comm, stream);

    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > 0) {
            cudaMemcpyAsync(recvbuf + displs[i], tmp_recv + i * max_count, recvcounts[i] * sizeof(double), cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(tmp_send);
    base_device::memory::delete_memory_op<double, base_device::DEVICE_GPU>()(tmp_recv);
}

void gatherv_nccl(const std::complex<double>* sendbuf, int sendcount, std::complex<double>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm, ncclComm_t nccl_comm, cudaStream_t stream)
{
    int size;
    MPI_Comm_size(comm, &size);
    if (size <= 0) return;

    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > max_count) max_count = recvcounts[i];
    }
    if (max_count == 0) return;

    std::complex<double> *tmp_send = nullptr, *tmp_recv = nullptr;
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tmp_send, max_count);
    base_device::memory::resize_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tmp_recv, size * max_count);

    if (sendcount > 0) {
        cudaMemcpyAsync(tmp_send, sendbuf, sendcount * sizeof(std::complex<double>), cudaMemcpyDeviceToDevice, stream);
    }

    ncclAllGather(tmp_send, tmp_recv, max_count * 2, ncclDouble, nccl_comm, stream);

    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > 0) {
            cudaMemcpyAsync(recvbuf + displs[i], tmp_recv + i * max_count, recvcounts[i] * sizeof(std::complex<double>), cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tmp_send);
    base_device::memory::delete_memory_op<std::complex<double>, base_device::DEVICE_GPU>()(tmp_recv);
}

void gatherv_nccl(const float* sendbuf, int sendcount, float* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm, ncclComm_t nccl_comm, cudaStream_t stream)
{
    int size;
    MPI_Comm_size(comm, &size);
    if (size <= 0) return;

    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > max_count) max_count = recvcounts[i];
    }
    if (max_count == 0) return;

    float *tmp_send = nullptr, *tmp_recv = nullptr;
    base_device::memory::resize_memory_op<float, base_device::DEVICE_GPU>()(tmp_send, max_count);
    base_device::memory::resize_memory_op<float, base_device::DEVICE_GPU>()(tmp_recv, size * max_count);

    if (sendcount > 0) {
        cudaMemcpyAsync(tmp_send, sendbuf, sendcount * sizeof(float), cudaMemcpyDeviceToDevice, stream);
    }

    ncclAllGather(tmp_send, tmp_recv, max_count, ncclFloat, nccl_comm, stream);

    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > 0) {
            cudaMemcpyAsync(recvbuf + displs[i], tmp_recv + i * max_count, recvcounts[i] * sizeof(float), cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    base_device::memory::delete_memory_op<float, base_device::DEVICE_GPU>()(tmp_send);
    base_device::memory::delete_memory_op<float, base_device::DEVICE_GPU>()(tmp_recv);
}

void gatherv_nccl(const std::complex<float>* sendbuf, int sendcount, std::complex<float>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm, ncclComm_t nccl_comm, cudaStream_t stream)
{
    int size;
    MPI_Comm_size(comm, &size);
    if (size <= 0) return;

    int max_count = 0;
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > max_count) max_count = recvcounts[i];
    }
    if (max_count == 0) return;

    std::complex<float> *tmp_send = nullptr, *tmp_recv = nullptr;
    base_device::memory::resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(tmp_send, max_count);
    base_device::memory::resize_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(tmp_recv, size * max_count);

    if (sendcount > 0) {
        cudaMemcpyAsync(tmp_send, sendbuf, sendcount * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream);
    }

    ncclAllGather(tmp_send, tmp_recv, max_count * 2, ncclFloat, nccl_comm, stream);

    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > 0) {
            cudaMemcpyAsync(recvbuf + displs[i], tmp_recv + i * max_count, recvcounts[i] * sizeof(std::complex<float>), cudaMemcpyDeviceToDevice, stream);
        }
    }

    cudaStreamSynchronize(stream);
    base_device::memory::delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(tmp_send);
    base_device::memory::delete_memory_op<std::complex<float>, base_device::DEVICE_GPU>()(tmp_recv);
}
}

#ifndef __CUDA_MPI
template <typename T>
struct object_cpu_point<T, base_device::DEVICE_GPU>
{
    bool alloc = false;
    T* get(const T* object, const int& n, T* tmp_space = nullptr)
    {
        T* object_cpu = nullptr;
        alloc = false;

        if (tmp_space == nullptr)
        {
            base_device::memory::resize_memory_op<T, base_device::DEVICE_CPU>()(object_cpu, n);
            alloc = true;
        }
        else
        {
            object_cpu = tmp_space;
        }
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(object_cpu,
                                                                                                          object,
                                                                                                          n);

        return object_cpu;
    }
    void sync_h2d(T* object, const T* object_cpu, const int& n)
    {
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_GPU, base_device::DEVICE_CPU>()(object,
                                                                                                          object_cpu,
                                                                                                          n);
    }
    void sync_d2h(T* object_cpu, const T* object, const int& n)
    {
        base_device::memory::synchronize_memory_op<T, base_device::DEVICE_CPU, base_device::DEVICE_GPU>()(object_cpu,
                                                                                                          object,
                                                                                                          n);
    }
    void del(T* object_cpu)
    {
        if (alloc)
        {
            base_device::memory::delete_memory_op<T, base_device::DEVICE_CPU>()(object_cpu);
        }
    }
};

template <typename T>
struct object_cpu_point<T, base_device::DEVICE_CPU>
{
    bool alloc = false;
    T* get(const T* object, const int& n, T* tmp_space = nullptr)
    {
        return const_cast<T*>(object);
    }
    void sync_h2d(T* object, const T* object_cpu, const int& n)
    {
    }
    void sync_d2h(T* object_cpu, const T* object, const int& n)
    {
    }
    void del(T* object_cpu)
    {
    }
};

template struct object_cpu_point<double, base_device::DEVICE_CPU>;
template struct object_cpu_point<double, base_device::DEVICE_GPU>;
template struct object_cpu_point<std::complex<double>, base_device::DEVICE_CPU>;
template struct object_cpu_point<std::complex<double>, base_device::DEVICE_GPU>;
template struct object_cpu_point<float, base_device::DEVICE_CPU>;
template struct object_cpu_point<float, base_device::DEVICE_GPU>;
template struct object_cpu_point<std::complex<float>, base_device::DEVICE_CPU>;
template struct object_cpu_point<std::complex<float>, base_device::DEVICE_GPU>;
#endif

} // namespace Parallel_Common

#endif