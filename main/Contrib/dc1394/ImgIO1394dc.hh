// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVLIMAGE_IMGIO1394DC_HEADER
#define RAVLIMAGE_IMGIO1394DC_HEADER 1
//! rcsid="$Id$"
//! lib=RavlImgIO1394dc
//! author="Charles Galambos"
//! docentry="Ravl.Images.Video.Video IO.IIDC"
#include "Ravl/DP/SPort.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/DP/AttributeType.hh"
#include "Ravl/Hash.hh"
#include "Ravl/Tuple2.hh"
#include <libdc1394/dc1394_control.h>

#include "Ravl/DP/AttributeValueTypes.hh"
#include "Ravl/StrStream.hh"
#include "Ravl/Image/ByteYUV422Value.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/MTLocks.hh"

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {
  //! userlevel=Develop
  //: Firewire dc camera control base class
  template<typename PixelT>
  class ImgIO1394dcBaseC
  {
  public:
    ImgIO1394dcBaseC(UIntT channel=0);
    //: Constructor.
    // channel > 100 means DMA acceess to channel = channel-100

    ~ImgIO1394dcBaseC();
    //: Destructor.

    bool Open(const StringC &dev);
    //: Open camera on device.

    bool CaptureImage(ImageC<PixelT> &img);
    //: Capture an image.

    bool HandleGetAttr(const StringC &attrName, StringC &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleSetAttr(const StringC &attrName, const StringC &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleGetAttr(const StringC &attrName, IntT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleSetAttr(const StringC &attrName, const IntT &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleGetAttr(const StringC &attrName, RealT &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleSetAttr(const StringC &attrName, const RealT &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleGetAttr(const StringC &attrName, bool &attrValue);
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    bool HandleSetAttr(const StringC &attrName, const bool &attrValue);
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    RealT SetFrameRate(RealT speed);
    //: Set capture framerate.
    // Returns the actual frame rate.

  protected:
    void BuildAttrList(AttributeCtrlBodyC &attrCtrl);
    //: Build attribute list.

    enum ControlTypeT { CT_IntValue, CT_FloatValue, CT_OnOff,CT_Auto };

    HashC<StringC,Tuple2C<IntT,ControlTypeT> > name2featureid;

    raw1394handle_t raw1394handle;
    dc1394_cameracapture camera;
//    nodeid_t cameraNode;

    StringC camera_vendor;
    StringC camera_model;
    StringC camera_euid;

    int cam_channel;
    int cam_format;
    int cam_mode;
    int cam_speed;
    int cam_framerate;
    quadlet_t available_framerates;
  private:
    struct Feature1394dcC
    {
      int featureno;
      const char *name;
      const char *desc;
    };

    const Feature1394dcC *FindFeature(int featureNo)
    {
      static const Feature1394dcC featureNames[] =
        { { FEATURE_BRIGHTNESS     ,"brightness"     ,"Brightness control." },
          { FEATURE_EXPOSURE       ,"exposure"       ,"Exposure time."      },
          { FEATURE_SHARPNESS      ,"sharpness"      ,"Sharpness filter, enhances edges in the image."   },
          { FEATURE_WHITE_BALANCE  ,"white_balance"  ,"Control relative gain of the different colour channels." },
          { FEATURE_HUE            ,"hue"            ,"Colour setup. " },
          { FEATURE_SATURATION     ,"saturation"     ,"Colour saturation." },
          { FEATURE_GAMMA          ,"gamma"          ,"Enable/Disable gamma correction." },
          { FEATURE_SHUTTER        ,"shutter_speed"  ,"Shutter speed. " },
          { FEATURE_GAIN           ,"gain"           ,"Grey level gain." },
          { FEATURE_IRIS           ,"iris"           ,"Set the iris, controls the amount of light entering the camera." },
          { FEATURE_FOCUS          ,"focus"          ,"Set the focus of the camera." },
          { FEATURE_TEMPERATURE    ,"temperature"    ,"Who knows ?" },
          { FEATURE_TRIGGER        ,"trigger"        ,"Set the camera trigger mode."},
          { FEATURE_ZOOM           ,"zoom"           ,"Set the focal length of the optics for the camera." },
          { FEATURE_PAN            ,"pan"            ,"The camera's horizontally angle." },
          { FEATURE_TILT           ,"tilt"           ,"The camera's vertical angle." },
          { FEATURE_OPTICAL_FILTER ,"optical_filter" ,"Who knows ?" },
          { FEATURE_CAPTURE_SIZE   ,"capture_size"   ,"??" },
          { FEATURE_CAPTURE_QUALITY,"capture_quality","??" },
          { FEATURE_CAPTURE_QUALITY,0            ,0 }
        };

      for(int i = 0; featureNames[i].name != 0; i++)
      {
        if(featureNames[i].featureno == featureNo)
          return &featureNames[i];
      }
      return 0;
    }
  };

  struct FrameRate1394dcC
  {
    RealT speed;
    unsigned int value;
  };

  extern const FrameRate1394dcC frameRates[];

  //! userlevel=Develop
  //: Firewire dc camera grabbing body
  template<typename PixelT>
  class DPIImage1394dcBodyC
    : public DPIPortBodyC<ImageC<PixelT> >,
      public ImgIO1394dcBaseC<PixelT>
  {
  public:
    DPIImage1394dcBodyC(const StringC &dev, UIntT channel=0)
      : ImgIO1394dcBaseC<PixelT>(channel)
    {
      Open(dev);
      BuildAttrList(*this);
    }
    //: Constructor.

    virtual bool IsGetReady() const
    { return raw1394handle != 0; }
    //: Is some data ready ?
    // true = yes.
    // Defaults to !IsGetEOS().

    virtual bool IsGetEOS() const
    { return !IsGetReady(); }
    //: Has the End Of Stream been reached ?
    // true = yes.
    
    virtual bool Get(ImageC<PixelT> &buff)
    { return CaptureImage(buff); }
    //: Get next image.

    virtual ImageC<PixelT> Get() {
      ImageC<PixelT> buff;
      if(!CaptureImage(buff))
        throw DataNotReadyC("Failed to capture image. ");
      return buff;
    }
    //: Get next image.

    virtual bool GetAttr(const StringC &attrName,StringC &attrValue){
      if(HandleGetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool SetAttr(const StringC &attrName,const StringC &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.


    virtual bool GetAttr(const StringC &attrName,IntT &attrValue){
      if(HandleGetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool SetAttr(const StringC &attrName,const IntT &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool GetAttr(const StringC &attrName,RealT &attrValue){
      if(HandleGetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool SetAttr(const StringC &attrName,const RealT &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
        return true;
      return DPPortBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool GetAttr(const StringC &attrName,bool &attrValue){
      if(HandleGetAttr(attrName,attrValue))
        return true;
      return AttributeCtrlBodyC::GetAttr(attrName,attrValue);
    }
    //: Get a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

    virtual bool SetAttr(const StringC &attrName,const bool &attrValue) {
      if(HandleSetAttr(attrName,attrValue))
        return true;
      return AttributeCtrlBodyC::SetAttr(attrName,attrValue);
    }
    //: Set a stream attribute.
    // Returns false if the attribute name is unknown.
    // This is for handling stream attributes such as frame rate, and compression ratios.

  protected:
  };

  //! userlevel=Develop
  //: Firewire dc camera grabbing

  template<class PixelT>
  class DPIImage1394dcC
    : public DPIPortC<ImageC<PixelT> >
  {
  public:
    DPIImage1394dcC()
      : DPEntityC(true)
    {}
    //: Default constructor.

    DPIImage1394dcC(const StringC &str, UIntT channel=0)
      : DPEntityC(*new DPIImage1394dcBodyC<PixelT>(str, channel))
    {}
  };

  //ImgIO1394dcBaseC

  //: Default constructor.
  template<typename PixelT>
  ImgIO1394dcBaseC<PixelT>::ImgIO1394dcBaseC(UIntT channel)
    : raw1394handle(0),
      cam_channel(channel),
      cam_format(FORMAT_VGA_NONCOMPRESSED),
      cam_mode(-1),
      cam_speed(SPEED_400),
      cam_framerate(FRAMERATE_15)
  {
    if(typeid(PixelT) == typeid(ByteT))
    {
      cerr << "Open in grayscale mode\n";
      cam_mode = MODE_640x480_MONO;
    }
    else if(typeid(PixelT) == typeid(ByteRGBValueC))
    {
      cerr << "Open in RGB mode\n";
      cam_mode = MODE_640x480_RGB;
    }
    else if(typeid(PixelT) == typeid(ByteYUV422ValueC))
    {
      cerr << "Open in YUV422 mode\n";
      cam_mode = MODE_640x480_YUV422;
    }
    else
    {
      cerr << "usupported image pixel type\n";
    }
  }

  //: Destructor.
  template<typename PixelT>
  ImgIO1394dcBaseC<PixelT>::~ImgIO1394dcBaseC()
  {
    if(raw1394handle == 0)
      return ;
    if(cam_channel >= 100) // DMA
    {
      if(dc1394_dma_unlisten(raw1394handle, &camera) != DC1394_SUCCESS)
        cerr << "couldn't stop the camera?\n";
      if(dc1394_dma_release_camera(raw1394handle, &camera) != DC1394_SUCCESS)
        cerr << "couldn't release the camera?\n";
    }
    else
    {
      if(dc1394_stop_iso_transmission(raw1394handle,camera.node)!=DC1394_SUCCESS)
        cerr << "couldn't stop the camera?\n";
      dc1394_release_camera(raw1394handle,&camera);
    }
    raw1394_destroy_handle(raw1394handle);
  }

  //: Open camera on device.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::Open(const StringC &dev)
  {
    cerr << "Cam channel = " << cam_channel << '\n';
    if(cam_channel >= 100) // DMA access
      cerr << "ImgIO1394dcBaseC::Open(), Called. Dev=" << dev << " using DMA acceess\n";
    else
      cerr << "ImgIO1394dcBaseC::Open(), Called. Dev=" << dev << "\n";

    if(cam_mode == -1)
    {
      cerr << "unsupported format\n";
      return false;
    }

    MTWriteLockC hold(2);
    raw1394handle = dc1394_create_handle(0); //assume only one port=0
    if(raw1394handle == 0)
    {
      cerr << "ERROR: Unable to aqure a raw1394 handle\n"
              "Please check \n"
              "  - if the kernel modules `ieee1394',`raw1394' and `ohci1394' are loaded  \n"
              "  - if you have read/write access to /dev/raw1394";
      return false;
    }
    IntT numNodes = raw1394_get_nodecount(raw1394handle);
    IntT numCameras;
    nodeid_t * camera_nodes = dc1394_get_camera_nodes(raw1394handle,&numCameras,1);
    if (numCameras<1)
    {
      cerr << "ERROR: No cameras found. \n";
      return false;
    }
    cerr << numCameras << " camera(s) found\n";
    cerr << numNodes << " camera node(s) found\n";

    if(cam_channel >= 100) // DMA
      camera.node = camera_nodes[cam_channel - 100];
    else
      camera.node = camera_nodes[cam_channel];

    if(camera.node == numNodes-1)
    {
      cerr << "\n"
              "Sorry, your camera is the highest numbered node\n"
              "of the bus, and has therefore become the root node.\n"
              "The root node is responsible for maintaining \n"
              "the timing of isochronous transactions on the IEEE \n"
              "1394 bus.  However, if the root node is not cycle master \n"
              "capable (it doesn't have to be), then isochronous \n"
              "transactions will not work.  The host controller card is \n"
              "cycle master capable, however, most cameras are not.\n"
              "\n"
              "The quick solution is to add the parameter \n"
              "attempt_root=1 when loading the OHCI driver as a \n"
              "module.  So please do (as root):\n"
              "\n"
              "   rmmod ohci1394\n"
              "   insmod ohci1394 attempt_root=1\n"
              "\n"
              "for more information see the FAQ at \n"
              "http://linux1394.sourceforge.net/faq.html#DCbusmgmt\n"
              "\n";
      return false;
    }

    // Get some information about the camera....
    dc1394_camerainfo camerainfo;
    if(dc1394_get_camera_info(raw1394handle,camera.node,&camerainfo) < 0)
    {
      cerr << "Failed to get camera info. \n";
      return false;
    }

    camera_vendor = StringC(camerainfo.vendor);
    camera_model = StringC(camerainfo.model);
    {
      StrOStreamC strm;
      strm.os() << hex << camerainfo.euid_64;
      camera_euid = strm.String();
    }

    // Query framerates.
    if(dc1394_query_supported_framerates(raw1394handle, camera.node, cam_format,
                                         cam_mode, &available_framerates) != DC1394_SUCCESS)
    {
      cerr << "Failed to query rates. \n";
      return false;
    }

    cerr << "Rate=" << hex << available_framerates << "\n" << dec;

    // Find fastest supported framerate.
    if (available_framerates & (1U << (31-5)))
      cam_framerate = FRAMERATE_60;
    else if (available_framerates & (1U << (31-4)))
      cam_framerate = FRAMERATE_30;
    else if (available_framerates & (1U << (31-3)))
      cam_framerate = FRAMERATE_15;
    else if (available_framerates & (1U << (31-2)))
      cam_framerate = FRAMERATE_7_5;
    else if (available_framerates & (1U << (31-1)))
      cam_framerate = FRAMERATE_3_75;
    else if (available_framerates & (1U << (31-0)))
      cam_framerate = FRAMERATE_1_875;

    // Setup capture.
    if(cam_channel >= 100) // DMA
    {
      dc1394_feature_set features;
      if(dc1394_get_camera_feature_set(raw1394handle, camera.node, &features) != DC1394_SUCCESS )
      {
        cerr << "unable to get camera features\n";
      }
      else
      {
        cerr << "features\n";
        dc1394_print_feature_set(&features);
      }
      const int NUM_BUFFERS = 8;
      const int DROP_FRAMES = 1;
      unsigned int ch, sp;
      if(dc1394_get_iso_channel_and_speed(raw1394handle, camera.node, &ch, &sp) != DC1394_SUCCESS )
      {
         cerr << "unable to get channel and speed\n";
      }
      cerr << "channel=" << ch << "\nspeed" << sp << "\n";
      cerr << "trining to set\nchannel=" << cam_channel-100 << "\nspeed" << cam_speed << "\n";
      if(dc1394_dma_setup_capture(raw1394handle, camera.node, cam_channel-100+1, cam_format,
                                  cam_mode, cam_speed, cam_framerate, NUM_BUFFERS,
                                  DROP_FRAMES, dev.chars(), &camera) != DC1394_SUCCESS)
      {
        cerr << "unable to setup camera-\n"
                "check that the video mode, framerate and format are\n"
                "supported by your camera\n";
        return false;
      }
    }
    else
    {
      if(dc1394_setup_capture(raw1394handle, camera.node, cam_channel, cam_format,
                   cam_mode, cam_speed, cam_framerate, &camera)!=DC1394_SUCCESS)
      {
        cerr << "unable to setup camera-\n"
                "check that the video mode, framerate and format are\n"
                "supported by your camera\n";
        return false;
      }
    }

    if (dc1394_start_iso_transmission(raw1394handle, camera.node) != DC1394_SUCCESS)
    {
      cerr << "unable to start camera iso transmission\n";
      return false;
    }
    cerr << "Size=" << camera.frame_width << " " << camera.frame_height << "\n";
    return true;
  }

  //: Capture an image.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::CaptureImage(ImageC<PixelT> &img)
  {
    MTWriteLockC hold(2);
    if(cam_channel >= 100) // DMA
    {
      if(dc1394_dma_single_capture(&camera) != DC1394_SUCCESS)
      {
        cerr << "unable to capture a frame\n";
        return false;
      }
    }
    else
    {
      if (dc1394_single_capture(raw1394handle,&camera) != DC1394_SUCCESS)
      {
        cerr << "unable to capture a frame\n";
        return false;
      }
    }

    img = ImageC<PixelT>(camera.frame_height, camera.frame_width,
                        (PixelT *) camera.capture_buffer, false).Copy();

    if(cam_channel > 100) // DMA
      dc1394_dma_done_with_buffer(&camera);

    return true;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleGetAttr(const StringC &attrName,StringC &attrValue)
  {
    if(attrName == "vendor")
    {
      attrValue = camera_vendor;
      return true;
    }
    if(attrName == "model")
    {
      attrValue = camera_model;
      return true;
    }
    if(attrName == "euid")
    {
      attrValue = camera_euid;
      return true;
    }
    return false;
  }

  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleSetAttr(const StringC &attrName,const StringC &attrValue)
  {
    return false;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleGetAttr(const StringC &attrName,IntT &attrValue)
  {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;

    switch(featureInfo.Data2())
    {
    case CT_IntValue:
      dc1394_get_feature_value(raw1394handle,camera.node,featureInfo.Data1(),(UIntT *) &attrValue);
      break;
    case CT_FloatValue:
    case CT_OnOff:
    case CT_Auto:
      cerr << "ImgIO1394dcBaseC::HandleGetAttr(), WARNING: Used for On/Off or Auto. \n";
      return false;
    }
    return true;
  }

  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleSetAttr(const StringC &attrName,const IntT &attrValue)
  {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2())
    {
    case CT_FloatValue:
    case CT_IntValue:
      dc1394_set_feature_value(raw1394handle,camera.node,featureInfo.Data1(),attrValue);
      break;
    case CT_OnOff:
    case CT_Auto:
      cerr << "ImgIO1394dcBaseC::HandleSetAttr(), WARNING: Used for On/Off or Auto. \n";
      return false;
    }
    return true;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleGetAttr(const StringC &attrName,RealT &attrValue)
  {
    MTWriteLockC hold(2);
    if(attrName == "framerate")
    {
      unsigned int setting = 0;
      dc1394_get_video_framerate(raw1394handle, camera.node, &setting);
      for(int i = 0;frameRates[i].speed > 0;i++)
      {
        //cerr << "Checking " << frameRates[i].value << "  " << setting << "\n";
        if(frameRates[i].value == setting)
        {
          attrValue = frameRates[i].speed;
          return true;
        }
      }
      cerr << "ImgIO1394dcBaseC::HandleGetAttr(), Unrecognised speed attribute " << setting << "\n";
      attrValue = 15;
      return true;
    }
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName, featureInfo))
      return false;
    float tmp;
    switch(featureInfo.Data2())
    {
    case CT_IntValue:
      dc1394_query_absolute_feature_value(raw1394handle,camera.node,featureInfo.Data1(),&tmp);
      attrValue = (RealT) tmp;
      break;
    case CT_FloatValue:
    case CT_OnOff:
    case CT_Auto:
      cerr << "ImgIO1394dcBaseC::HandleGetAttr(), WARNING: Used for On/Off or Auto. \n";
      return false;
    }
    return true;
  }

  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleSetAttr(const StringC &attrName,const RealT &attrValue)
  {
    MTWriteLockC hold(2);
    if(attrName == "framerate")
    {
      SetFrameRate(attrValue);
      return true;
    }
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2())
    {
    case CT_FloatValue:
    case CT_IntValue:
      dc1394_set_absolute_feature_value(raw1394handle,camera.node,featureInfo.Data1(),(float) attrValue);
      break;
    case CT_OnOff:
    case CT_Auto:
      cerr << "ImgIO1394dcBaseC::HandleSetAttr(), WARNING: Used for On/Off or Auto. \n";
      return false;
    }
    return true;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleGetAttr(const StringC &attrName,bool &attrValue)
  {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    dc1394bool_t tmp;
    switch(featureInfo.Data2())
    {
    case CT_FloatValue:
    case CT_IntValue:
      return false;
    case CT_OnOff:
      dc1394_is_feature_on(raw1394handle,camera.node,featureInfo.Data1(),&tmp);
      break;
    case CT_Auto:
      dc1394_is_feature_auto(raw1394handle,camera.node,featureInfo.Data1(),&tmp);
      break;
    }
    attrValue = (tmp != DC1394_FALSE);
    return true;
  }

  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  template<typename PixelT>
  bool ImgIO1394dcBaseC<PixelT>::HandleSetAttr(const StringC &attrName,const bool &attrValue)
  {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2())
    {
    case CT_FloatValue:
    case CT_IntValue:
      return false;
    case CT_OnOff:
      dc1394_feature_on_off(raw1394handle,camera.node,featureInfo.Data1(),
                            (attrValue == DC1394_FALSE) ? false : true);
      break;
    case CT_Auto:
      dc1394_auto_on_off(raw1394handle,camera.node,featureInfo.Data1(),
                         (attrValue == DC1394_FALSE) ? false : true);
      break;
    }
    return true;
  }

  //: Build attribute list.
  template<typename PixelT>
  void ImgIO1394dcBaseC<PixelT>::BuildAttrList(AttributeCtrlBodyC &attrCtrl)
  {
    MTWriteLockC hold(2);

    attrCtrl.RegisterAttribute(AttributeTypeStringC("vendor","Supplier of device",true,false));
    attrCtrl.RegisterAttribute(AttributeTypeStringC("model","model of device",true,false));
    attrCtrl.RegisterAttribute(AttributeTypeStringC("euid","Unique id for device",true,false));

    // libdc1394 BUG FIX, clear available flags before calling get_feature_set.
    dc1394_feature_set featureSet;
    for(int i = 0;i < NUM_FEATURES;i++)
      featureSet.feature[i].available = DC1394_FALSE;

    dc1394_get_camera_feature_set(raw1394handle,camera.node,&featureSet);

    // Go through feature list.
    for(int i = 0;i < NUM_FEATURES;i++)
    {
      const dc1394_feature_info &feature = featureSet.feature[i];
      if(feature.available != DC1394_TRUE)
        continue; // Feature is not avalable on this camera.
      const Feature1394dcC *featureInfo = FindFeature(feature.feature_id);
      const char *cfeatName = featureInfo->name;
      if(cfeatName == 0)
      {
        cerr << "WARNING: Unknown featureid " << feature.feature_id << "\n";
        continue;
      }
      StringC featName(cfeatName);

      if((feature.abs_control > 0) && (feature.absolute_capable > 0))
      {
        ONDEBUG(cerr << "Setting up " << featName << " Absolute. Min=" << feature.abs_min << " Max=" << feature.abs_max << " Value=" << feature.abs_value << "\n");
        RealT diff = feature.abs_max - feature.abs_min;
        AttributeTypeNumC<RealT> attr(featName,featureInfo->desc,feature.readout_capable,feature.manual_capable,
                    feature.abs_min,feature.abs_max,diff/1000,feature.abs_value);
        attrCtrl.RegisterAttribute(attr);
        name2featureid[featName] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_IntValue);
      }
      else
      {
        ONDEBUG(cerr << "Setting up " << featName << " Int. Min=" << feature.min << " Max=" << feature.max << " Value=" << feature.value << "\n");
        AttributeTypeNumC<IntT> attr(featName,featureInfo->desc,feature.readout_capable,feature.manual_capable,
                   feature.min,feature.max,1,feature.value);
        attrCtrl.RegisterAttribute(attr);
        name2featureid[featName] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_IntValue);
      }
      if(feature.on_off_capable)
      {
        StringC name = featName + "_enable";
        AttributeTypeBoolC attr(name,featureInfo->desc,true,true,feature.is_on);
        attrCtrl.RegisterAttribute(attr);
        name2featureid[name] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_OnOff);
      }
      if(feature.auto_capable)
      {
        StringC name = featName + "_auto";
        AttributeTypeBoolC attr(name,featureInfo->desc,true,true,feature.auto_active);
        attrCtrl.RegisterAttribute(attr);
        name2featureid[name] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_Auto);

        //AttributeTypeNumC<IntT> attr(name,featureInfo->desc,true,true,feature.auto_active);
      }
    }

    // Setup framerate attribute.
    RealT minspeed = 1e8;
    RealT maxspeed = 0;
    for(int i = 0;frameRates[i].speed > 0;i++)
    {
      //cerr << "Rate=" << frameRates[i].speed << " " << !(available_framerates & (1U << (31-(frameRates[i].value - FRAMERATE_MIN)))) << "\n";
      if ((available_framerates & (1U << (31-(frameRates[i].value - FRAMERATE_MIN)))) == 0)
        continue; // framerate not supported.
      if(frameRates[i].value < minspeed)
        minspeed = frameRates[i].value;
      if(frameRates[i].value > maxspeed)
        maxspeed = frameRates[i].value;
    }

    AttributeTypeNumC<RealT> frameRateAttr("framerate","Number of images a second to capture",true,true,minspeed,maxspeed,1,(maxspeed + minspeed)/2);
    attrCtrl.RegisterAttribute(frameRateAttr);
  }

  //: Set capture framerate.
  // Returns the actual frame rate.
  template<typename PixelT>
  RealT ImgIO1394dcBaseC<PixelT>::SetFrameRate(RealT speed)
  {
    // Find closest setting.
    RealT err = 1000000;
    int setting = -1;
    for(int i = 0;frameRates[i].speed > 0;i++)
    {
      //cerr << "Rate=" << frameRates[i].speed << " " << !(available_framerates & (1U << (31-(frameRates[i].value - FRAMERATE_MIN)))) << "\n";
      if ((available_framerates & (1U << (31-(frameRates[i].value - FRAMERATE_MIN)))) == 0)
        continue; // framerate not supported.
      RealT nerr = Abs(frameRates[i].speed - speed);
      if(nerr < err)
      {
        setting = i;
        err = nerr;
      }
    }
    if(setting < 0)
    {
      cerr << "ERROR: failed to find appropriate framerate. \n";
      return -1;
    }
    unsigned int current = 0;
    unsigned int newsetting = (unsigned int) frameRates[setting].value;

    dc1394_get_video_framerate(raw1394handle,camera.node,&current);

    if(current != newsetting) { // Need to change
      if(dc1394_set_video_framerate(raw1394handle, camera.node, newsetting) != DC1394_SUCCESS)
      {
        cerr << "Failed to setup camera for new framerate. \n";
        return 0;
      }
/*      if (dc1394_stop_iso_transmission(raw1394handle,camera.node)!=DC1394_SUCCESS)
        cerr << "ERROR: couldn't stop the camera?\n";
      //dc1394_set_video_framerate(raw1394handle,camera.node,);
      dc1394_release_camera(raw1394handle,&camera);
      if (dc1394_setup_capture(raw1394handle,cameraNode,
			       cam_channel,
			       cam_format,
			       cam_mode,
			       cam_speed,
			       newsetting,
			       &camera)!=DC1394_SUCCESS)
      {
        cerr << "Failed to setup camera for new framerate. \n";
        return 0;
      }
      cam_framerate = newsetting;
      if (dc1394_start_iso_transmission(raw1394handle,camera.node) != DC1394_SUCCESS)
      {
        cerr << "ERROR: unable to restart camera iso transmission\n";
        return 0;
      }*/
    }
    return frameRates[setting].speed; // Return actual speed.
  }

}

#endif
