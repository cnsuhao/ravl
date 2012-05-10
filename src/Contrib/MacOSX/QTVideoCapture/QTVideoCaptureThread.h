/* This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, Charles Galambos.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here*/
/*! lib=RavlMacOSXVideoCapture */
#import <Foundation/Foundation.h>
#import <QTKit/QTKit.h>
#include "Ravl/String.hh"
#include "Ravl/RCHandleV.hh"
#include "Ravl/DP/Port.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Image/ByteYUVValue.hh"
#include "Ravl/Image/ByteYUV422Value.hh"
#import "Ravl/MacOSX/BufferCVImage.hh"

namespace RavlImageN {

  class QTCaptureThreadBodyC
    : virtual public RavlN::RCBodyVC
  {
  public:
    QTCaptureThreadBodyC(const RavlN::StringC &name,const std::type_info &pixelType);

    //: Destructor
    ~QTCaptureThreadBodyC();

    //: start capture
    bool Start();

    //: stop capture
    bool Stop();

    //: Handle incoming buffer
    virtual bool RecieveBuffer(CVImageBufferRef &aBuffer);
  protected:
    RavlN::StringC m_name;
    id m_captureThread;
    const std::type_info &m_pixelType;
  };


  class QTCaptureThreadC
    : RavlN::RCHandleC<QTCaptureThreadBodyC>
  {
  public:
    QTCaptureThreadC(QTCaptureThreadBodyC *body)
     : RavlN::RCHandleC<QTCaptureThreadBodyC>(body)
    {}

    //! Handle incoming buffer
    bool RecieveBuffer(CVImageBufferRef &aBuffer)
    { return Body().RecieveBuffer(aBuffer); }

  };
}


@interface RavlQTCaptureThread : NSObject {
  QTCaptureSession            *mCaptureSession;
  QTCaptureDecompressedVideoOutput    *mCaptureVideoOutput;
  QTCaptureDeviceInput        *mCaptureVideoDeviceInput;
  CVImageBufferRef        mCurrentImageBuffer;
  BOOL mFinished;
  RavlImageN::QTCaptureThreadBodyC *mCaptureThread;
}

// methods
- (BOOL)openCamera: (const std::type_info *)pixelType;

- (void)startCapture;

- (void)stopCapture;

- (void)setFrameHandler:(RavlImageN::QTCaptureThreadBodyC *)frameHandler;

@end


