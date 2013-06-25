#ifndef RAVL_CUDA_CUDAUTIL_HH_
#define RAVL_CUDA_CUDAUTIL_HH_

#include "Ravl/SysLog.hh"
#include "cuda_runtime.h"
#include "Ravl/Exception.hh"

namespace RavlN { namespace CUDAN {

  const char *CudaErrorString(int error_id);

#define RavlCUDACheckError(x) \
  { if(x != cudaSuccess) { RavlError("CUDA call failed %d : %s ",(int) x,CudaErrorString(x)); RavlAssert(0); throw RavlN::ExceptionOperationFailedC("CUDA op failed."); } }
}}

#endif
