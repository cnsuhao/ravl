
#include "Ravl/RealHistogramNd.hh"

namespace RavlN {

  //: Default constructor

  RealHistogramBaseNdC::RealHistogramBaseNdC()
   : m_dataSize(0)
  {}


  RealHistogramBaseNdC::RealHistogramBaseNdC(const SArray1dC<RealRangeC> &ranges,const SArray1dC<size_t> &binSizes)
   : m_inputLimits(ranges),
     m_gridSize(binSizes)
  {
    Init();
  }

  RealHistogramBaseNdC::RealHistogramBaseNdC(BinIStreamC &strm)
  {
    ByteT version = 0;
    strm >> version;
    if(version != 1)
      throw ExceptionUnexpectedVersionInStreamC("RealHistogramNdC<>");
    strm >> m_inputLimits >> m_gridSize;
    Init();
  }
  //: Binary stream constructor


  //: Scaling.
  VectorC RealHistogramBaseNdC::Scale() const {
    VectorC ret(m_gridSize.Size());
    for(unsigned i = 0;i < ret.Size();i++)
      ret[i] = 1.0/m_inputLimits[i].Size();
    return ret;
  }

  //: Offset used in hash table.
  VectorC RealHistogramBaseNdC::Offset() const {
    VectorC ret(m_gridSize.Size());
    for(unsigned i = 0;i < ret.Size();i++)
      ret[i] = m_inputLimits[i].Min();
    return ret;
  }

  void RealHistogramBaseNdC::Init() {
    m_gridScale = SArray1dC<size_t>(m_gridSize.Size());
    size_t scale = 1;
    for(unsigned i = 0;i < m_gridSize.Size();i++) {
      m_gridScale[i] = scale;
      scale = scale * m_gridSize[i];
    }
    m_dataSize = scale;
  }

  //: Access bin vector falls in.

  IndexNdC RealHistogramBaseNdC::Bin(const VectorC &v) const {
    if(v.Size() != m_inputLimits.Size()) {
      throw RavlN::ExceptionOutOfRangeC("Unexpected number of input parameters.");
    }
    IndexNdC ret(m_inputLimits.Size());

    return ret;
  }

  //: Get the middle of given bin.

  VectorC RealHistogramBaseNdC::MidBin(const IndexNdC &bin) const {
    if(bin.Size() != m_inputLimits.Size()) {
      throw RavlN::ExceptionOutOfRangeC("Unexpected number of input parameters.");
    }
    VectorC ret(m_inputLimits.Size());

    return ret;
  }

  //! Compute index of bin
  bool RealHistogramBaseNdC::ComputeIndex(const VectorC &data,size_t &bin) const {
    if(!data.IsValid()) {
       throw RavlN::ExceptionOutOfRangeC("Invalid input vector.");
     }
     if(data.Size() != m_inputLimits.Size()) {
       throw RavlN::ExceptionOutOfRangeC("Unexpected number of input parameters.");
     }
     size_t indexAt = 0;
     bool ret = true;
     for(unsigned i = 0;i < m_gridSize.Size();i++) {
       RealT value = m_inputLimits[i].Clip(data[i]);
       RealT rIndex = ((value - m_inputLimits[i].Min())/m_inputLimits[i].Size()) * (double) m_gridSize[i];
       int index = Floor(rIndex);
       if(index < 0) { // Clip lower limit.
         index = 0;
         ret = false;
         //std::cerr << "Under flow.\n";
       } else if(index >= static_cast<int>(m_gridSize[i])) { // Clip upper limit.
         //std::cerr << "Over flow. Index=" << index << " Size:" << m_gridSize[i] << "\n";
         index = (m_gridSize[i]-1);
         ret = false;
       }
       indexAt += (size_t) index * m_gridScale[i];
     }
     bin = indexAt;
     return ret;
  }

  //! Compute index of bin
  //! Returns false if vote has been clipped to bounds
  bool RealHistogramBaseNdC::ComputeIndexAndRemainders(const VectorC &data,size_t &bin,SArray1dC<RealT> &offset) const {
    if(!data.IsValid()) {
      throw RavlN::ExceptionOutOfRangeC("Invalid input vector.");
    }
    if(data.Size() != m_inputLimits.Size()) {
      throw RavlN::ExceptionOutOfRangeC("Unexpected number of input parameters.");
    }
    bool ret = true;
    UIntT indexAt = 0;
    offset = SArray1dC<RealT>(m_gridSize.Size());
    for(unsigned i = 0;i < m_gridSize.Size();i++) {
      RealT value = m_inputLimits[i].Clip(data[i]);
      offset[i] = (((value - m_inputLimits[i].Min())/m_inputLimits[i].Size()) * (double) m_gridSize[i]) - 0.5;
      int index = Floor(offset[i]);
      if(index < 0) { // Clip lower limit.
        index = 0;
        offset[i] = 0.0;
        ret = false;
      } else if(index >= static_cast<int>(m_gridSize[i]-1)) { // Clip upper limit.
        index = (m_gridSize[i]-2);
        offset[i] = 1.0;
        if(index >= static_cast<int>(m_gridSize[i]))
          ret = false;
      } else {
        offset[i] -= (double) index;
      }
      indexAt += index * m_gridScale[i];
    }
    bin = indexAt;
    return ret;
  }


}