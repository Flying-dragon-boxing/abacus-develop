#ifndef FFT_CUDA_H
#define FFT_CUDA_H

#include "fft_base.h"
#include "cufft.h"
#include "cuda_runtime.h"
namespace ModuleBase
{
template <typename FPTYPE>
class FFT_CUDA : public FFT_BASE<FPTYPE>
{
    public:
        FFT_CUDA(){};
        ~FFT_CUDA(){}; 
        
	    void setupFFT() override; 

        void clear() override;

        void cleanFFT() override;

        /** 
        * @brief Initialize the fft parameters
        * @param nx_in  number of grid points in x direction
        * @param ny_in  number of grid points in y direction
        * @param nz_in  number of grid points in z direction
        * 
        */
        void initfft(int nx_in, 
                     int ny_in, 
                     int nz_in) override;
        
        /**
         * @brief Get the real space data
         * @return real space data
         */
        std::complex<FPTYPE>* get_auxr_3d_data() const override;
        
        /**
         * @brief Forward FFT in 3D
         * @param in  input data, complex FPTYPE
         * @param out  output data, complex FPTYPE
         * 
         * This function performs the forward FFT in 3D.
         */
        void fft3D_forward(std::complex<FPTYPE>* in, 
                           std::complex<FPTYPE>* out) const override;
        /**
         * @brief Backward FFT in 3D
         * @param in  input data, complex FPTYPE
         * @param out  output data, complex FPTYPE
         * 
         * This function performs the backward FFT in 3D.
         */
        void fft3D_backward(std::complex<FPTYPE>* in, 
                            std::complex<FPTYPE>* out) const override;

        /**
         * @brief Setup batched FFT plan
         * @param batch_size  number of FFTs to perform in batch
         * 
         * This function creates a batched FFT plan using cufftPlanMany.
         * Call this before using batched FFT operations.
         */
        void setupFFT_batched(int batch_size);

        /**
         * @brief Clean up batched FFT plan
         * 
         * This function destroys the batched FFT plan.
         */
        void cleanFFT_batched();

        /**
         * @brief Forward batched FFT in 3D
         * @param in  input data, complex FPTYPE [batch_size, nz, ny, nx]
         * @param out  output data, complex FPTYPE [batch_size, nz, ny, nx]
         * @param batch_size  number of FFTs to perform
         * 
         * This function performs batched forward FFT in 3D.
         */
        void fft3D_forward_batched(std::complex<FPTYPE>* in, 
                                   std::complex<FPTYPE>* out,
                                   int batch_size) const;

        /**
         * @brief Backward batched FFT in 3D
         * @param in  input data, complex FPTYPE [batch_size, nz, ny, nx]
         * @param out  output data, complex FPTYPE [batch_size, nz, ny, nx]
         * @param batch_size  number of FFTs to perform
         * 
         * This function performs batched backward FFT in 3D.
         */
        void fft3D_backward_batched(std::complex<FPTYPE>* in, 
                                    std::complex<FPTYPE>* out,
                                    int batch_size) const;

    private:
        cufftHandle c_handle = {};
        cufftHandle z_handle = {};
        cufftHandle c_handle_batch = {};  // batched plan for float
        cufftHandle z_handle_batch = {};  // batched plan for double
        
        std::complex<float>* c_auxr_3d = nullptr;  // fft space
        std::complex<double>* z_auxr_3d = nullptr; // fft space

        // Workspace for optimized FFT performance
        void* c_workspace = nullptr;  // workspace for single precision
        void* z_workspace = nullptr;  // workspace for double precision
        size_t c_workspace_size = 0;
        size_t z_workspace_size = 0;

        int planned_batch_size = 0;
        bool batch_initialized = false;

};

} // namespace ModuleBase
#endif
