
#include "Ravl/MacOSX/QTImageIPort.hh"
#include "Ravl/MacOSX/QTVideoCaptureThread.h"

namespace RavlImageN {

  DPPortQTImageImplBaseC::DPPortQTImageImplBaseC(const StringC &name,const std::type_info &pixelType)
  {}

  //! Start capture.
  bool DPPortQTImageImplBaseC::StartCapture() {
    RavlAssertMsg(0,"abstract method called .");
    return false;
  }

  //! Stop capture.
  bool DPPortQTImageImplBaseC::StopCapture()
  {
    RavlAssertMsg(0,"abstract method called .");
    return false;
  }

  //--------------------------------------------------
  
  template<class PixelT>
  class DPPortQTImageImplFinalC
    : public DPPortQTImageImplC<PixelT>,
      public QTCaptureThreadBodyC
  {
  public:
    DPPortQTImageImplFinalC(const StringC &name,int buffSize)
     : DPPortQTImageImplC<PixelT>(name,buffSize),
       QTCaptureThreadBodyC(name,typeid(PixelT))
    {}
    
    //! Start capture.
    virtual bool StartCapture() {
      if(!Start()) {
        std::cerr << "Failed to start camera. \n";
        return false;
      }
      this->m_isStarted = true;
      return true;
    }

    bool StopCapture()
    {
      Stop();
      return true;
    }
    //! Handle incoming buffer
    virtual bool RecieveBuffer(CVImageBufferRef &videoFrame)
    {
      size_t rows = CVPixelBufferGetHeight(videoFrame);
      size_t cols = CVPixelBufferGetWidth(videoFrame);
      RavlImageN::BufferCVImageC<PixelT> newBuffer(videoFrame);
      this->m_queue.Put(RavlImageN::ImageC<PixelT>(rows,cols,newBuffer));
      //RavlN::Save("@X",anImage);
      return true;
    }

  protected:
    
  };

  
  
  // Find the correct implementation and open it
  bool OpenQTImageImpl(const StringC &name,
                       const std::type_info &pixelType,
                       int buffSize,
                       DPPortQTImageImplBaseC::RefT &handle) 
  {
    if(pixelType == typeid(ByteRGBValueC)) {
      handle = new DPPortQTImageImplFinalC<ByteRGBValueC>(name,buffSize);
      return true;
    }
    if(pixelType == typeid(ByteT)) {
      handle = new DPPortQTImageImplFinalC<ByteT>(name,buffSize);
      return true;
    }
    return false;
  }

}
