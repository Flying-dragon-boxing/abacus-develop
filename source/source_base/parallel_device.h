#ifndef __PARALLEL_DEVICE_H__
#define __PARALLEL_DEVICE_H__
#ifdef __MPI
#include "base/macros/cuda.h"
#include "mpi.h"
#include "source_base/module_device/device.h"
#include "source_base/module_device/memory_op.h"

#include <complex>
#include <nccl.h>
#include <cuda.h>
#include <cuda_runtime.h>
namespace Parallel_Common
{
void isend_data(const double* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request);
void isend_data(const std::complex<double>* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request);
void isend_data(const float* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request);
void isend_data(const std::complex<float>* buf, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request);
void send_data(const double* buf, int count, int dest, int tag, MPI_Comm& comm);
void send_data(const std::complex<double>* buf, int count, int dest, int tag, MPI_Comm& comm);
void send_data(const float* buf, int count, int dest, int tag, MPI_Comm& comm);
void send_data(const std::complex<float>* buf, int count, int dest, int tag, MPI_Comm& comm);
void recv_data(double* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status);
void recv_data(std::complex<double>* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status);
void recv_data(float* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status);
void recv_data(std::complex<float>* buf, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status);
void bcast_data(std::complex<double>* object, const int& n, const MPI_Comm& comm);
void bcast_data(std::complex<float>* object, const int& n, const MPI_Comm& comm);
void bcast_data(double* object, const int& n, const MPI_Comm& comm);
void bcast_data(float* object, const int& n, const MPI_Comm& comm);
void reduce_data(std::complex<double>* object, const int& n, const MPI_Comm& comm);
void reduce_data(std::complex<float>* object, const int& n, const MPI_Comm& comm);
void reduce_data(double* object, const int& n, const MPI_Comm& comm);
void reduce_data(float* object, const int& n, const MPI_Comm& comm);
void gatherv_data(const double* sendbuf, int sendcount, double* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm);
void gatherv_data(const std::complex<double>* sendbuf, int sendcount, std::complex<double>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm);
void gatherv_data(const float* sendbuf, int sendcount, float* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm);
void gatherv_data(const std::complex<float>* sendbuf, int sendcount, std::complex<float>* recvbuf, const int* recvcounts, const int* displs, MPI_Comm& comm);

#ifndef __CUDA_MPI
template<typename T, typename Device>
struct object_cpu_point
{
    bool alloc = false;
    T* get(const T* object, const int& n, T* tmp_space = nullptr);
    void del(T* object);
    void sync_d2h(T* object_cpu, const T* object, const int& n);
    void sync_h2d(T* object, const T* object_cpu, const int& n);
};
#endif

/**
 * @brief send data in Device
 * 
 */
template <typename T, typename Device>
void send_dev(const T* object, int count, int dest, int tag, MPI_Comm& comm, T* tmp_space = nullptr)
{
#ifdef __CUDA_MPI
    send_data(object, count, dest, tag, comm);
#else
    object_cpu_point<T,Device> o;
    T* object_cpu = o.get(object, count, tmp_space);
    o.sync_d2h(object_cpu, object, count);
    send_data(object_cpu, count, dest, tag, comm);
    o.del(object_cpu);
#endif
    return;
}

/**
 * @brief isend data in Device
 * @note before the date in send_space is recieved, it should not be modified
 * 
 */
template <typename T, typename Device>
void isend_dev(const T* object, int count, int dest, int tag, MPI_Comm& comm, MPI_Request* request, T* send_space)
{
#ifdef __CUDA_MPI
    isend_data(object, count, dest, tag, comm, request);
#else
    object_cpu_point<T,Device> o;
    T* object_cpu = o.get(object, count, send_space);
    o.sync_d2h(object_cpu, object, count);
    isend_data(object_cpu, count, dest, tag, comm, request);
    o.del(object_cpu);
#endif
    return;
}

/**
 * @brief recv data in Device
 * 
 */
template <typename T, typename Device>
void recv_dev(T* object, int count, int source, int tag, MPI_Comm& comm, MPI_Status* status, T* tmp_space = nullptr)
{
#ifdef __CUDA_MPI
    recv_data(object, count, source, tag, comm, status);
#else
    object_cpu_point<T,Device> o;
    T* object_cpu = o.get(object, count, tmp_space);
    recv_data(object_cpu, count, source, tag, comm, status);
    o.sync_h2d(object, object_cpu, count);
    o.del(object_cpu);
#endif
    return;
}

/**
 * @brief bcast data in Device
 * 
 * @tparam T: float, double, std::complex<float>, std::complex<double>
 * @tparam Device 
 * @param ctx Device ctx
 * @param object complex arrays in Device
 * @param n the size of complex arrays
 * @param comm MPI_Comm
 * @param tmp_space tmp space in CPU
 */
template <typename T, typename Device>
void bcast_dev(T* object, const int& n, const MPI_Comm& comm, T* tmp_space = nullptr)
{
#ifdef __CUDA_MPI
    bcast_data(object, n, comm);
#else
    object_cpu_point<T,Device> o;
    T* object_cpu = o.get(object, n, tmp_space);
    o.sync_d2h(object_cpu, object, n);
    bcast_data(object_cpu, n, comm);
    o.sync_h2d(object, object_cpu, n);
    o.del(object_cpu);
#endif
    return;
}

template <typename T, typename Device>
void reduce_dev(T* object, const int& n, const MPI_Comm& comm, T* tmp_space = nullptr)
{
#ifdef __CUDA_MPI
    reduce_data(object, n, comm);
#else
    object_cpu_point<T,Device> o;
    T* object_cpu = o.get(object, n, tmp_space);
    o.sync_d2h(object_cpu, object, n);
    reduce_data(object_cpu, n, comm);
    o.sync_h2d(object, object_cpu, n);
    o.del(object_cpu);
#endif
    return;
}

template <typename T, typename Device>
void gatherv_dev(const T* sendbuf,
                 int sendcount,
                 T* recvbuf,
                 const int* recvcounts,
                 const int* displs,
                 MPI_Comm& comm,
                 T* tmp_sspace = nullptr,
                 T* tmp_rspace = nullptr)
{
#ifdef __CUDA_MPI
    gatherv_data(sendbuf, sendcount, recvbuf, recvcounts, displs, comm);
#else
    object_cpu_point<T,Device> o1, o2;
    int size = 0;
    MPI_Comm_size(comm, &size);
    int gather_space = displs[size - 1] + recvcounts[size - 1];
    T* sendbuf_cpu = o1.get(sendbuf, sendcount, tmp_sspace);
    T* recvbuf_cpu = o2.get(recvbuf, gather_space, tmp_rspace);
    o1.sync_d2h(sendbuf_cpu, sendbuf, sendcount);
    gatherv_data(sendbuf_cpu, sendcount, recvbuf_cpu, recvcounts, displs, comm);
    o2.sync_h2d(recvbuf, recvbuf_cpu, gather_space);
    o1.del(sendbuf_cpu);
    o2.del(recvbuf_cpu);
#endif
    return;
}

template <typename T, typename Device>
void gatherv_nccl(const T* sendbuf,
                  int sendcount,
                  T* recvbuf,
                  const int* recvcounts,
                  const int* displs,
                  ncclComm_t comm,
                  cudaStream_t stream)
{
    int size;
    ncclCommCount(comm, &size);
    if (size <= 1) {
        if (size == 1 && sendbuf != recvbuf && sendcount > 0) {
            cudaMemcpyAsync(recvbuf, sendbuf, (size_t)sendcount * sizeof(T), cudaMemcpyDeviceToDevice, stream);
        }
        return;
    }
    printf("size%d\n", size);
    // 1. 本地计算最大计数以进行对齐
    int nrecv_max = 0;
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > nrecv_max) nrecv_max = recvcounts[i];
    }
    if (nrecv_max <= 0) return;

    // 2. 异步分配临时缓冲区 (对齐 Padding 区域)
    // 字节单位计算
    size_t unit_size = sizeof(T);
    size_t send_bytes_max = (size_t)nrecv_max * unit_size;
    size_t recv_bytes_total = send_bytes_max * size;

    void *d_tmp_send = nullptr;
    void *d_tmp_recv = nullptr;

    // 使用框架封装或原生的异步分配
    cudaMallocAsync(&d_tmp_send, send_bytes_max, stream);
    cudaMallocAsync(&d_tmp_recv, recv_bytes_total, stream);

    // 3. Padding: 将原始数据拷贝到对齐区域的起始位置
    if (sendcount > 0) {
        cudaMemcpyAsync(d_tmp_send, (const void*)sendbuf, (size_t)sendcount * unit_size, cudaMemcpyDeviceToDevice, stream);
    }

    // 4. 执行对齐后的 ncclAllGather (按 Uint8 字节传输)
    // 传输长度为每个 rank 对应的最大字节数
    ncclAllGather((const void*)d_tmp_send,
                  (void*)d_tmp_recv,
                  send_bytes_max,
                  ncclUint8,
                  comm,
                  stream);

    // 5. Unpacking: 从对齐的缓冲区按原位移偏移拷贝回紧凑布局
    for (int i = 0; i < size; ++i) {
        if (recvcounts[i] > 0) {
            cudaMemcpyAsync(
                (void*)(recvbuf + displs[i]),
                (const void*)((char*)d_tmp_recv + (size_t)i * send_bytes_max),
                (size_t)recvcounts[i] * unit_size,
                cudaMemcpyDeviceToDevice,
                stream
            );
        }
    }

    // 6. 异步释放，保持流水线
    cudaFreeAsync(d_tmp_send, stream);
    cudaFreeAsync(d_tmp_recv, stream);
}

}
    

#endif
#endif