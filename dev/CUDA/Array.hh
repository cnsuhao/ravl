/*
 * CUDAArray.hh
 *
 *  Created on: May 25, 2013
 *      Author: charlesgalambos
 */

#ifndef RAVL_CUDA_ARRAY1D_HH_
#define RAVL_CUDA_ARRAY1D_HH_

#include "Ravl/SmartPtr.hh"
#include "Ravl/SArray1d.hh"
#include "Ravl/SArray2d.hh"
#include "cuda_runtime.h"
#include "Ravl/CUDA/Util.hh"
#include "Ravl/CUDA/ArrayOp.hh"

namespace RavlN { namespace CUDAN {


  //! Setup a CUDA stream
  class CStreamC
  {
  public:
    //! Constructor
    CStreamC();

    //! Destructor
    ~CStreamC();

    cudaStream_t Stream()
    { return m_cstream; }

    operator cudaStream_t &()
    { return m_cstream; }

    //! Wait for all operations to complete
    void Wait();

  protected:
    cudaStream_t m_cstream;

  private:
    CStreamC(const CStreamC &copy)
      : m_cstream(0)
    { RavlAssert(0); }
  };

  //! Raw buffer

  class CRawBufferC
   : public RavlN::RCBodyVC
  {
  public:
    //! Default constructor
    CRawBufferC();

    //! Allocate a number of bytes
    CRawBufferC(size_t bytes);

    //! Destructor
    virtual ~CRawBufferC();

    //! Access buffer
    void *RawBuffer()
    { return m_buffer; }

    //! Access buffer
    const void *RawBuffer() const
    { return m_buffer; }

    //! Number of bytes in buffer
    size_t SizeBytes() const
    { return m_sizeBytes; }

    //! Handle to the buffer
    typedef RavlN::SmartPtrC<CRawBufferC> RefT;
  protected:
    void *m_buffer;
    size_t m_sizeBytes;
  };

  //! Buffer
  template<typename DataT>
  class CBufferC
    : public CRawBufferC
  {
  public:
    //! Construct empty buffer.
    CBufferC()
      : m_size(0)
    {}

    //! Construct buffer with given size.
    CBufferC(size_t elements)
      : CRawBufferC(elements * sizeof(DataT)),
        m_size(elements)
    {}

    //! Access buffer
    DataT *Buffer()
    { return reinterpret_cast<DataT *>(RawBuffer()); }

    //! Access buffer
    const DataT *Buffer() const
    { return reinterpret_cast<DataT *>(RawBuffer()); }

    //! Size in elements of buffer.
    size_t Size() const
    { return m_size; }

    //! Handle to the buffer
    typedef typename RavlN::SmartPtrC<CBufferC<DataT> > RefT;

  protected:
    size_t m_size;
  };


  //! Buffer
  template<typename DataT>
  class CBuffer2dC
    : public CBufferC<DataT>
  {
  public:
    CBuffer2dC(size_t size1,size_t size2)
      : m_size1(size1),
        m_size2(size2),
        m_byteStride(0)
    {
      void *buffer = 0;
      RavlCUDACheckError(cudaMallocPitch(&buffer,&m_byteStride,m_size2 * sizeof(DataT),m_size1));
      this->m_buffer = buffer;
      this->m_size = m_byteStride * m_size1 / sizeof(DataT);
      this->m_sizeBytes = m_byteStride * m_size1;
      //RavlDebug("CArray stride:%zu Size:%zu %zu ",m_byteStride,m_size1,m_size2);
    }

    //! Access buffer
    DataT *Buffer()
    { return reinterpret_cast<DataT *>(this->RawBuffer()); }

    //! Access buffer
    const DataT *Buffer() const
    { return reinterpret_cast<DataT *>(this->RawBuffer()); }

    //! Size in elements of buffer.
    size_t Size1() const
    { return m_size1; }

    //! Size in elements of buffer.
    size_t Size2() const
    { return m_size2; }

    //! Access the stride in bytes
    size_t ByteStride() const
    { return m_byteStride; }

    //! Handle to the buffer
    typedef RavlN::SmartPtrC<CBuffer2dC<DataT> > RefT;

  protected:
    size_t m_size1;
    size_t m_size2;
    size_t m_byteStride;
  };



  //! A 1d array
  template<typename DataT>
  class CArray1dC
  {
  public:
    //! Default constructor
    CArray1dC()
     : m_refData(0),
       m_size(0)
    {}

    //! Construct with size
    CArray1dC(size_t numSize)
     : m_buffer(new CBufferC<DataT>(numSize)),
       m_refData(m_buffer->Buffer()),
       m_size(numSize)
    {}

    CArray1dC(CBufferC<DataT> &buffer,float *refData,size_t size)
     : m_buffer(&buffer),
       m_refData(refData),
       m_size(size)
    {}

    //! Copy contents of 'values' into this array.
    void Set(const SizeBufferAccessC<DataT> &values,cudaStream_t cstrm)
    {
      RavlAssert(values.Size() <= m_size);
      RavlCUDACheckError(cudaMemcpyAsync(m_refData,values.DataStart(),values.Size() * sizeof(DataT),cudaMemcpyHostToDevice,cstrm));
    }

    //! Copy contents of 'values' from this array.
    void Get(SizeBufferAccessC<DataT> &values,cudaStream_t cstrm) const
    {
      RavlAssert(values.Size() <= m_size);
      RavlCUDACheckError(cudaMemcpyAsync(values.DataStart(),m_refData,values.Size() * sizeof(DataT),cudaMemcpyDeviceToHost,cstrm));
    }

    void GetPart(int offset,size_t len,SizeBufferAccessC<DataT> &values,cudaStream_t cstrm) const
    {
      RavlAssert(len + offset <= m_size);
      RavlCUDACheckError(cudaMemcpyAsync(values.DataStart(),&m_refData[offset],len * sizeof(DataT),cudaMemcpyDeviceToHost,cstrm));
    }

    void SetPart(int offset,size_t len,const SizeBufferAccessC<DataT> &values,cudaStream_t cstrm) const
    {
      RavlAssert(len + offset <= m_size);
      RavlCUDACheckError(cudaMemcpyAsync(&m_refData[offset],values.DataStart(),len * sizeof(DataT),cudaMemcpyHostToDevice,cstrm));
    }


    void CopyFrom(const CArray1dC<DataT> &src,size_t srcOffset,size_t dstOffset,size_t len,cudaStream_t cstrm)
    {
      RavlAssert(dstOffset + len <= m_size);
      RavlAssert(srcOffset + len <= src.Size());
      RavlCUDACheckError(cudaMemcpyAsync(&m_refData[dstOffset],src[srcOffset],len * sizeof(DataT),cudaMemcpyDeviceToDevice,cstrm));
    }

    //! Fill array with value
    void Fill(const DataT &value,cudaStream_t cstrm)
    {
      if(m_size == 0)
        return ;
      DoCUDAFill(m_refData,&value,sizeof(DataT),m_size,cstrm);
    }

    //! Set contents of array to zero
    //! This does a byte wise set, so care must be taken if DataT is not a built in type
    void SetZero(cudaStream_t cstrm)
    {
      if(m_size == 0)
        return ;
      cudaMemsetAsync(m_refData,0,m_size * sizeof(DataT),cstrm);
    }

    //! Access member
    DataT &operator[](int off)
    {
      RavlAssert(off >= 0 && (size_t) off <= m_size);
      return m_refData[off];
    }

    //! Access member
    const DataT &operator[](int off) const
    {
      RavlAssert((size_t) off <= m_size);
      return m_refData[off];
    }

    //! Access size.
    size_t Size() const
    { return m_size; }
  protected:
    typename CBufferC<DataT>::RefT m_buffer;
    float *m_refData;
    size_t m_size;
  };



  //! A 2d array
  template<typename DataT>
  class CArray2dC
  {
  public:
    CArray2dC()
      : m_refData(0),
        m_size1(0),
        m_size2(0)
    {}

    //! Construct 2d array
    CArray2dC(size_t size1,size_t size2)
      : m_buffer(new CBuffer2dC<DataT>(size1,size2)),
        m_refData(m_buffer->Buffer()),
        m_size1(size1),
        m_size2(size2)
    {}

    CArray1dC<DataT> SliceRow(int row)
    {
      RavlAssert(row < m_size1 && row >= 0);
      return CArray1dC<DataT>(*m_buffer,ComputeAddress(row,0),m_size2);
    }

    //! Set contents of array to zero
    //! This does a byte wise set, so care must be taken if DataT is not a built in type
    void SetZero(cudaStream_t cstrm)
    {
      if(m_size1 == 0 || m_size2 == 0)
        return ;
      RavlCUDACheckError(cudaMemset2DAsync(m_refData,m_buffer->ByteStride(),0,m_size2 * sizeof(DataT),m_size1,cstrm));
    }

    //! Set a row to zero
    void RowSetZero(int row,cudaStream_t cstrm)
    {
      if(m_size2 == 0)
        return ;
      RavlAssert((unsigned) row < m_size1);
      RavlCUDACheckError(cudaMemsetAsync(ComputeAddress(row,0),0,m_size2 * sizeof(DataT),cstrm));
    }

    //! Set a row to zero
    void RowPartSetZero(int row,int col,unsigned size,cudaStream_t cstrm)
    {
      RavlAssert(size + col <= m_size2);
      RavlAssert((unsigned) row < m_size1);
      RavlCUDACheckError(cudaMemsetAsync(ComputeAddress(row,col),0,size * sizeof(DataT),cstrm));
    }

    //! Set a block of entries to zero
    void BlockSetZero(int row,int col,int rowSize,int colSize,cudaStream_t cstrm)
    {
      RavlAssert(rowSize >= 0 && colSize >= 0);
#if 1
      for(int i = 0;i < rowSize;i++) {
        RowPartSetZero(row + i,col,colSize,cstrm);
      }
#else
      RavlCUDACheckError(cudaMemset2DAsync(ComputeAddress(row,col), m_buffer->ByteStride(), 0, rowSize, colSize, cstrm));
#endif
    }

    //! Copy contents of 'values' into this array.
    void Set(const SArray2dC<DataT> &values,cudaStream_t cstrm)
    {
      RavlAssert(values.Size1() <= m_size1);
      RavlAssert(values.Size2() <= m_size2);
      //RavlAssert(values.IsContinuous());
#if 1
      for(unsigned i = 0;i < values.Size1();i++) {
        RavlCUDACheckError(cudaMemcpyAsync(ComputeAddress(i,0),&(values[i][0]),values.Size2() * sizeof(DataT),cudaMemcpyHostToDevice,cstrm));
      }
#else
      RavlCUDACheckError(cudaMemcpy2DAsync(m_refData,m_buffer->ByteStride(),
                   values.DataStart(),values.Stride() * sizeof(DataT),
                   values.Size2() * sizeof(DataT),
                   values.Size1(),
                   cudaMemcpyHostToDevice,cstrm));
#endif
    }

    //! Copy contents of 'values' from this array.
    void Get(SArray2dC<DataT> &values,cudaStream_t cstrm) const
    {
      RavlAssert(values.Size1() <= m_size1);
      RavlAssert(values.Size2() <= m_size2);
      RavlDebug("Sizes: %u %u ",(unsigned) values.Size1(),(unsigned) values.Size2());
#if 1
      for(unsigned i = 0;i < values.Size1();i++) {
        RavlCUDACheckError(cudaMemcpyAsync(&(values[i][0]),ComputeAddress(i,0),values.Size2() * sizeof(DataT),cudaMemcpyDeviceToHost,cstrm));
      }
#else
      RavlCUDACheckError(cudaMemcpy2DAsync(values.DataStart(),values.Stride() * sizeof(DataT),
                        m_refData,m_buffer->ByteStride(),
                        values.Size2() * sizeof(DataT),
                        values.Size1(),
                        cudaMemcpyDeviceToHost,cstrm));
#endif
    }

    void GetRow(int row,int offset,size_t len,SArray1dC<DataT> &values,cudaStream_t cstrm) const
    {
      RavlAssert(len <= values.Size());
      RavlAssert(offset + len <= m_size2);
      RavlAssert((size_t) row < m_size1);
      RavlCUDACheckError(cudaMemcpyAsync(&(values[0]),ComputeAddress(row,offset),len * sizeof(DataT),cudaMemcpyDeviceToHost,cstrm));
    }

    void SetRow(int row,int offset,size_t len,const SArray1dC<DataT> &values,cudaStream_t cstrm)
    {
      RavlAssert(len <= values.Size());
      RavlAssert(offset + len <= m_size2);
      RavlAssert((size_t) row < m_size1);
      RavlCUDACheckError(cudaMemcpyAsync(ComputeAddress(row,offset),&(values[0]),len * sizeof(DataT),cudaMemcpyHostToDevice,cstrm));
    }

    void CopyRowFrom(int row,const CArray2dC<DataT> &src,size_t srcOffset,size_t dstOffset,size_t len,cudaStream_t cstrm)
    {
      RavlAssert((size_t) row < m_size1);
      RavlAssert((size_t) row < src.Size1());
      RavlAssert(dstOffset + len <= m_size2);
      RavlAssert(srcOffset + len <= src.Size2());
      RavlCUDACheckError(cudaMemcpyAsync(ComputeAddress(row,dstOffset),
                                         src.ComputeAddress(row,srcOffset),len * sizeof(DataT),cudaMemcpyDeviceToDevice,cstrm));
    }

    void CopyRowFrom(int row,const CArray1dC<DataT> &src,size_t srcOffset,size_t dstOffset,size_t len,cudaStream_t cstrm)
    {
      RavlAssert((size_t) row < m_size1);
      RavlAssert(dstOffset + len <= m_size2);
      RavlAssert(srcOffset + len <= src.Size());
      RavlCUDACheckError(cudaMemcpyAsync(ComputeAddress(row,dstOffset),
                                         &src[srcOffset],len * sizeof(DataT),cudaMemcpyDeviceToDevice,cstrm));
    }

    //! Access size of first dimension.
    size_t Size1() const
    { return m_size1; }

    //! Access size of first dimension.
    size_t Size2() const
    { return m_size2; }

    //! Access byte stride of the array
    size_t ByteStride() const
    { return m_buffer->ByteStride(); }

    //! Compute address of an element
    DataT *ComputeAddress(int ind1,int ind2)
    { return reinterpret_cast<DataT *>(reinterpret_cast<char *>(m_refData) + ind1 * m_buffer->ByteStride()) + ind2; }

    //! Compute address of an element
    const DataT *ComputeAddress(int ind1,int ind2) const
    { return reinterpret_cast<const DataT *>(reinterpret_cast<const char *>(m_refData) + ind1 * m_buffer->ByteStride()) + ind2; }

  protected:
    typename CBuffer2dC<DataT>::RefT m_buffer;
    float *m_refData;
    size_t m_size1;
    size_t m_size2;
  };

}}



#endif /* CUDAARRAY_HH_ */
