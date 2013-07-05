
#include "cuda_runtime.h"
#include "Ravl/CUDA/ArrayOp.hh"
#include "Ravl/CUDA/Util.hh"
#include "Ravl/Assert.hh"
#include "Ravl/SysLog.hh"

namespace RavlN { namespace CUDAN {
  
  template<typename DataT>
  static __device__ inline DataT *OffsetByte(DataT *data,int offset,size_t stride)
  {
    return reinterpret_cast<DataT *>(&reinterpret_cast<char *>(data)[stride * offset]);
  }

  template<typename DataT>
  static __device__ inline const DataT *OffsetByte(const DataT *data,int offset,size_t stride)
  {
    return reinterpret_cast<const DataT *>(&reinterpret_cast<const char *>(data)[stride * offset]);
  }

  // There are probably better ways of doing this, but it works for now.

  __global__ void DevDoCUDAFill2(unsigned short *refData,unsigned short val,size_t numElems)
  {
    const size_t tid = blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y) + threadIdx.x;
    if(tid < numElems)
      refData[tid] = val;
  }

  __global__ void DevDoCUDAFill4(unsigned *refData,unsigned val,size_t numElems)
  {
    const size_t tid = blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y) + threadIdx.x;
    if(tid < numElems)
      refData[tid] = val;
  }

  static const unsigned g_maxGridSize = 65535;
  
  void ComputeDim(size_t numElems,unsigned &numThreads,dim3 &numBlocks) {
    numThreads = 32;
    if(numElems <= numThreads) {
      numThreads = numElems;
      return ;
    }
    size_t count = numElems/numThreads;
    if((count+1) <= g_maxGridSize) {
      numBlocks.x = count;
      if((count * numThreads) < numElems)
        numBlocks.x++;
    } else {
      numBlocks.x = g_maxGridSize;
      count /= (g_maxGridSize+1);
      if((count+1) <= g_maxGridSize) {
        numBlocks.y = count;
        if((numBlocks.x * numBlocks.y * numThreads) < numElems)
          numBlocks.y++;
      } else {
        RavlIssueError("Block size too big.");
      }
    }
    
  }
  
  void ComputeDimSmall(size_t numElems,unsigned &numThreads,dim3 &numBlocks)
  {
    numThreads = 32;
    if(numElems <= numThreads) {
      numThreads = numElems;
      return ;
    }
    size_t count = numElems/numThreads;
    if((count+1) <= g_maxGridSize) {
      numBlocks.x = count;
      if((count * numThreads) < numElems)
        numBlocks.x++;
    } else {
      numThreads=512;
      count = numElems/numThreads;
      if((count+1) >= g_maxGridSize) {
        RavlIssueError("Block size too big.");
      }
      numBlocks.x = count;
      if((count * numThreads) < numElems)
        numBlocks.x++;
    }
  }

  
  //! Fill memory with a pattern
  __host__ void DoCUDAFill(void *refData,const void *value,size_t elemSize,size_t numElems,cudaStream_t cstrm)
  {
    if(numElems == 0)
      return ;
    //RavlDebug("Doing fill size Elem:%zu  Size:%zu ...",elemSize,numElems);
    switch(elemSize) {
      case 1: {
        unsigned char aValue = *((const unsigned char *) value);
        RavlCUDACheckError(cudaMemsetAsync(refData,aValue,numElems,cstrm));
        return ;
      }
      case 2: {
        unsigned short aValue = *((const unsigned short *) value);
        unsigned int numThreads;
        dim3 numBlocks;
        ComputeDim(numElems,numThreads,numBlocks);
        DevDoCUDAFill2<<<numBlocks, numThreads,0,cstrm>>>((unsigned short *)refData,aValue,numElems);
        return ;
      }
      case 4: {
        unsigned aValue = *((const unsigned *) value);
        unsigned int numThreads;
        dim3 numBlocks;
        ComputeDim(numElems,numThreads,numBlocks);
        //RavlDebug("Using sizes: %u %u %u Threads:%u ",numBlocks.x,numBlocks.y,numBlocks.z,numThreads);
        DevDoCUDAFill4<<<numBlocks, numThreads,0,cstrm>>>((unsigned *)refData,aValue,numElems);
        return ;
      }
      default:
        RavlIssueError("Fill size not supported.");
    }
  }
   
   
  __global__ void DevDoCUDAFill42d(unsigned *refData,unsigned val,size_t numElems,size_t stride)
  {
    const size_t tid = blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y) + threadIdx.x;
    unsigned *at = OffsetByte(refData,blockIdx.z,stride);
    if(tid < numElems)
      at[tid] = val;
  }

  void DoCUDAFill2d(void *refData,const void *value,size_t elemSize,size_t numElems,size_t numRows,size_t stride,cudaStream_t cstrm)
  {
    if(numElems == 0)
      return ;
    //RavlDebug("Doing fill size Elem:%zu  Size:%zu ...",elemSize,numElems);
    switch(elemSize) {
      case 1: {
        unsigned char aValue = *((const unsigned char *) value);
        RavlCUDACheckError(cudaMemset2DAsync(refData, stride, aValue, numElems, numRows, cstrm));
        return ;
      }
      case 4: {
        unsigned aValue = *((const unsigned *) value);
        unsigned int numThreads;
        dim3 numBlocks;
        ComputeDim(numElems,numThreads,numBlocks);
        numBlocks.z = numRows;
        //RavlDebug("Using sizes: %u %u %u Threads:%u ",numBlocks.x,numBlocks.y,numBlocks.z,numThreads);
        DevDoCUDAFill42d<<<numBlocks, numThreads,0,cstrm>>>((unsigned *)refData,aValue,numElems,stride);
        return ;
      }
      default:
        RavlIssueError("Fill size not supported.");
    }
  }


}}