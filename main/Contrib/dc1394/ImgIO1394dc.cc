// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImgIO1394dc
//! author="Charles Galambos"

#include "Ravl/Image/ImgIO1394dc.hh"
#include "Ravl/DP/AttributeValueTypes.hh"
#include "Ravl/MTLocks.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {
  
  struct Feature1394dcC {
    int featureno;
    const char *name;
    const char *desc;
  };

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
  
  
  //ImgIO1394dcBaseC
  
  //: Default constructor.
  
  ImgIO1394dcBaseC::ImgIO1394dcBaseC()
    : raw1394handle(0)
  {}
  
  //: Destructor.
  
  ImgIO1394dcBaseC::~ImgIO1394dcBaseC()
  {
    if(raw1394handle == 0) 
      return ;
    if (dc1394_stop_iso_transmission(raw1394handle,camera.node)!=DC1394_SUCCESS) 
      cerr << "couldn't stop the camera?\n";
    dc1394_release_camera(raw1394handle,&camera);
    raw1394_destroy_handle(raw1394handle);
  }

  //: Open camera on device.
  
  bool ImgIO1394dcBaseC::Open(const StringC &dev,const type_info &npixType) {
    cerr << "ImgIO1394dcBaseC::Open(), Called. Dev=" << dev << "\n";
    MTWriteLockC hold(2);
    raw1394handle = dc1394_create_handle(0);
    if(raw1394handle == 0) {
      cerr << "ERROR: Unable to open raw1394 device, check modules`ieee1394',`raw1394' and `ohci1394' are loaded  \n";
      return false;
    }
    IntT numNodes = raw1394_get_nodecount(raw1394handle);
    IntT numCameras;
    nodeid_t * camera_nodes = dc1394_get_camera_nodes(raw1394handle,&numCameras,1);
    if (numCameras<1) {
      cerr << "ERROR: No camera's found. \n";
      return false;
    }
    
    if( camera_nodes[0] == numNodes-1) {
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
    
    if (dc1394_setup_capture(raw1394handle,camera_nodes[0],
			     0, /* channel */ 
			     FORMAT_VGA_NONCOMPRESSED,
			     MODE_640x480_MONO,
			     SPEED_400,
			     FRAMERATE_15,
			     &camera)!=DC1394_SUCCESS) {
      cerr << "unable to setup camera-\n"
	"check line %d of %s to make sure\n"
	"that the video mode,framerate and format are\n"
	"supported by your camera\n";
      return false;
    }
    
    if (dc1394_start_iso_transmission(raw1394handle,camera.node) != DC1394_SUCCESS)  {
      cerr << "unable to start camera iso transmission\n";
      return false;
    }
    cerr << "Size=" << camera.frame_width << " " << camera.frame_height << "\n";
    return true;
  }
  
  //: Capture an image.
  
  bool ImgIO1394dcBaseC::CaptureImage(ImageC<ByteT> &img) {
    MTWriteLockC hold(2);
    if (dc1394_single_capture(raw1394handle,&camera) != DC1394_SUCCESS)  {
      cerr << "unable to capture a frame\n";
      return false;
    }
    img = ImageC<ByteT>(camera.frame_height,camera.frame_width,(ByteT *) camera.capture_buffer,false).Copy();
    return true;
  }
  
  static const Feature1394dcC *FindFeature(int featureNo) {
    for(int i = 0;featureNames[i].name != 0;i++) {
      if(featureNames[i].featureno == featureNo)
	return &featureNames[i];
    }
    return 0;
  }

  //: Get a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool ImgIO1394dcBaseC::HandleGetAttr(const StringC &attrName,IntT &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2()) {
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
  
  bool ImgIO1394dcBaseC::HandleSetAttr(const StringC &attrName,const IntT &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2()) {
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
  
  bool ImgIO1394dcBaseC::HandleGetAttr(const StringC &attrName,RealT &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    float tmp;
    switch(featureInfo.Data2()) {
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
  
  bool ImgIO1394dcBaseC::HandleSetAttr(const StringC &attrName,const RealT &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2()) {
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
  
  bool ImgIO1394dcBaseC::HandleGetAttr(const StringC &attrName,bool &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    dc1394bool_t tmp;
    switch(featureInfo.Data2()) {
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
    attrValue = tmp;
    return true;
  }
  
  //: Set a stream attribute.
  // Returns false if the attribute name is unknown.
  // This is for handling stream attributes such as frame rate, and compression ratios.
  
  bool ImgIO1394dcBaseC::HandleSetAttr(const StringC &attrName,const bool &attrValue) {
    MTWriteLockC hold(2);
    Tuple2C<IntT,ControlTypeT> featureInfo;
    if(!name2featureid.Lookup(attrName,featureInfo))
      return false;
    switch(featureInfo.Data2()) {
    case CT_FloatValue:
    case CT_IntValue:
      return false;
    case CT_OnOff:
      dc1394_feature_on_off(raw1394handle,camera.node,featureInfo.Data1(),attrValue);
      break;
    case CT_Auto:
      dc1394_auto_on_off(raw1394handle,camera.node,featureInfo.Data1(),attrValue);
      break;
    }
    return true;
  }
  
  //: Build attribute list.
  
  void ImgIO1394dcBaseC::BuildAttrList(AttributeCtrlBodyC &attrCtrl) {
    MTWriteLockC hold(2);
    dc1394_feature_set featureSet;
    dc1394_get_camera_feature_set(raw1394handle,camera.node,&featureSet);
    
    // Go through feature list.
    
    for(int i = 0;i < NUM_FEATURES;i++) {
      const dc1394_feature_info &feature = featureSet.feature[i];
      if(!feature.available)
	continue; // Feature is not avalable on this camera.
      const Feature1394dcC *featureInfo = FindFeature(feature.feature_id);
      const char *cfeatName = featureInfo->name;
      if(cfeatName == 0) {
	cerr << "WARNING: Unknown featureid " << feature.feature_id << "\n";
	continue;
      }
      StringC featName(cfeatName);
      
      if(feature.abs_control && (feature.absolute_capable > 0)) {
	ONDEBUG(cerr << "Setting up " << featName << " Absolute. Min=" << feature.abs_min << " Max=" << feature.abs_max << " Value=" << feature.abs_value << "\n");
	RealT diff = feature.abs_max - feature.abs_min;
	AttributeTypeNumC<RealT> attr(featName,featureInfo->desc,feature.readout_capable,feature.manual_capable,
				      feature.abs_min,feature.abs_max,diff/1000,feature.abs_value);
	attrCtrl.RegisterAttribute(attr);
	name2featureid[featName] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_IntValue);
      } else {
	ONDEBUG(cerr << "Setting up " << featName << " Int. Min=" << feature.min << " Max=" << feature.max << " Value=" << feature.value << "\n");
	AttributeTypeNumC<IntT> attr(featName,featureInfo->desc,feature.readout_capable,feature.manual_capable,
				     feature.min,feature.max,1,feature.value);
	attrCtrl.RegisterAttribute(attr);
	name2featureid[featName] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_IntValue);
      }
      if(feature.on_off_capable) {
	StringC name = featName + "_enable";
	AttributeTypeBoolC attr(name,featureInfo->desc,true,true,feature.is_on);
	attrCtrl.RegisterAttribute(attr);
	name2featureid[name] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_OnOff);
      }
      if(feature.auto_capable) {
	StringC name = featName + "_auto";
	AttributeTypeBoolC attr(name,featureInfo->desc,true,true,feature.auto_active);
	attrCtrl.RegisterAttribute(attr);
	name2featureid[name] = Tuple2C<IntT,ControlTypeT>(feature.feature_id,CT_Auto);
      }
      
    }
  }

}

