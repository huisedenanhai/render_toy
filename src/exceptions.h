#include <cuda.h>
#include <cuda_runtime.h>
#include <exception>
#include <optix.h>

namespace toy {
class CudaException : public std::exception {
public:
  CudaException(cudaError_t code, const char *what)
      : m_Code(code), std::exception(what) {}
  inline cudaError_t code() const noexcept {
    return m_Code;
  }

private:
  cudaError_t m_Code;
};

class CuException : public std::exception {
public:
  CuException(CUresult code, const char *what)
      : m_Code(code), std::exception(what) {}
  inline CUresult code() const noexcept {
    return m_Code;
  }

private:
  CUresult m_Code;
};

class OptixException : public std::exception {
public:
  OptixException(OptixResult code, const char *what)
      : m_Code(code), std::exception(what) {}
  inline OptixResult code() const noexcept {
    return m_Code;
  }

private:
  OptixResult m_Code;
};

} // namespace toy

#define TOY_CUDA_CHECK_OR_THROW(exp, cleanup)                                  \
  do {                                                                         \
    cudaError_t res = (exp);                                                   \
    if (res != cudaSuccess) {                                                  \
      {                                                                        \
        cleanup;                                                               \
      }                                                                        \
      throw toy::CudaException(res, cudaGetErrorString(res));                  \
    }                                                                          \
  } while (0)

#define TOY_CU_CHECK_OR_THROW(exp, cleanup)                                    \
  do {                                                                         \
    CUresult res = (exp);                                                      \
    if (res != CUDA_SUCCESS) {                                                 \
      {                                                                        \
        cleanup;                                                               \
      }                                                                        \
      const char *info;                                                        \
      cuGetErrorString(res, &info);                                            \
      throw toy::CuException(res, info);                                       \
    }                                                                          \
  } while (0)

#define TOY_OPTIX_CHECK_OR_THROW(exp, cleanup)                                 \
  do {                                                                         \
    OptixResult res = (exp);                                                   \
    if (res != OPTIX_SUCCESS) {                                                \
      {                                                                        \
        cleanup;                                                               \
      }                                                                        \
      throw toy::OptixException(res, optixGetErrorString(res));                \
    }                                                                          \
  } while (0)