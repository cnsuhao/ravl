#ifndef RAVL_CUDA_ARRAYOP_HH_
#define RAVL_CUDA_ARRAYOP_HH_

#include "cuda_runtime.h"

namespace RavlN { namespace CUDAN {

  //! Fill memory with a pattern
  void DoCUDAFill(void *refData,const void *value,size_t elemSize,size_t numElems,cudaStream_t cstrm = 0);

  //! Fill memory with a pattern
  void DoCUDAFill2d(void *refData,const void *value,size_t elemSize,size_t numElems,size_t numRows,size_t stride,cudaStream_t cstrm = 0);


  //! Compute number of threads and block size.
  void ComputeDim(size_t numElems,unsigned &numThreads,dim3 &numBlocks);

  //! Compute number of threads and block size. using only x block dimension and threads.
  void ComputeDimSmall(size_t numElems,unsigned &numThreads,dim3 &numBlocks);


}}

#endif
