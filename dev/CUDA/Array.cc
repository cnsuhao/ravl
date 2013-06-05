/*
 * CUDAArray.cc
 *
 *  Created on: May 25, 2013
 *      Author: charlesgalambos
 */

#include "Ravl/CUDA/Array.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/CUDA/Util.hh"

namespace RavlN { namespace CUDAN {

  //! Constructor
  CStreamC::CStreamC()
  {
    RavlCUDACheckError(cudaStreamCreate(&m_cstream));
  }

  //! Destructor
  CStreamC::~CStreamC()
  {
    RavlCUDACheckError(cudaStreamDestroy(m_cstream));
  }

  void CStreamC::Wait()
  {
    RavlCUDACheckError(cudaStreamSynchronize(m_cstream));
  }

  //! Default constructor

  CRawBufferC::CRawBufferC()
   : m_buffer(0),
     m_sizeBytes(0)
  {}

  //! Allocate a number of bytes

  CRawBufferC::CRawBufferC(size_t bytes)
   : m_buffer(0),
     m_sizeBytes(bytes)
  {
    RavlCUDACheckError(cudaMalloc(&m_buffer,bytes));
  }

  //! Destructor

  CRawBufferC::~CRawBufferC()
  {
    if(m_buffer != 0) {
      RavlCUDACheckError(cudaFree(m_buffer));
    }
  }

}}
