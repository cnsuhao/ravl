// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=CSPDriver
//! file="Ravl/Contrib/ClipStationPro/exCSPGrab.cc"
//! userlevel=Normal
//! author="Lee Gregory"
//! docentry="Ravl.Images.Video.Video IO.ClipStationPro"

#include "Ravl/Image/ImgIOCSP.hh"
#include "Ravl/TimeCode.hh"
#include "Ravl/DList.hh"

using namespace RavlImageN;

int main (int argc , char ** argv ) 
{
  // An example program to grab a list of images and timecodes
  
  // setup some lists 
  DListC<ImageC<ByteYUV422ValueC> > imgList ;                       // The list of images 
  DListC<TimeCodeC>                 tcList  ;                       // The list of timecodes 
  
  // setup the capture cards 
  StringC device1 ("PCI,card:0") ;                                  // Hardware device name 
  ImageRectangleC rect (576,720) ;                                  // The image size 
  DPIImageClipStationProC<ByteYUV422ValueC> csp1 (device1, rect) ;   // Capture interface 1 
  

  // now lets grab some images and timecodes and store them in the list 
  const UIntT numberOfGrabs = 10 ; 
  for ( UIntT count = 1 ; count <= numberOfGrabs ; ++count ) 
    {
      // grab the image 
      ImageC<ByteYUV422ValueC> image = csp1.Get() ; 
      // grab the timecode 
      TimeCodeC timeCode = csp1.GetAttr("timecode") ;                  // The GetAttr method returns a string, but 
      cerr << "\n Grabbed image with timecode " << timeCode.ToText() ; // strings can be implicly converted to timecodes

      // now store them in their lists
      imgList.InsLast(image) ; 
      tcList.InsLast (timeCode) ; 
 }
  
  return 0 ; 
}
