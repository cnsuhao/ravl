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

namespace RavlImageN {

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
    if (dc1394_single_capture(raw1394handle,&camera) != DC1394_SUCCESS)  {
      cerr << "unable to capture a frame\n";
      return false;
    }
    img = ImageC<ByteT>(camera.frame_height,camera.frame_width,(ByteT *) camera.capture_buffer,false).Copy();
    return true;
  }

}

