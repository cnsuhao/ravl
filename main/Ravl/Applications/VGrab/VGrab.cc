// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
///////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! file="Ravl/Applications/VideoGrab/VideoGrab.cc"
//! docentry="Ravl.Applications.VideoGrab"
//! author="Lee Gregory"
//! userlevel=Normal
#include "Ravl/EntryPnt.hh"
#include "Ravl/Option.hh" 
#include "Ravl/Image/ByteYUV422Value.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Image/ImageRectangle.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/DP/Port.hh"
#include "Ravl/Assert.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/TimeCode.hh"
#include "Ravl/OS/Date.hh"

using namespace RavlN ; 
using namespace RavlImageN ; 

int VGrab(int argc, char ** argv) 
{
  typedef ByteYUV422ValueC PixelT ; 
  typedef ImageC<PixelT> ImageT ; 

  // get some options and usage 
  // ----------------------------
  OptionC opts(argc,argv) ; 
  StringC usage = "This program captures video from grabber cards. The grabber device can be specified using RAVL virtual files and defaults to the ClipStationPro driver. \n Output formats are determined by the extension specified, 4cif is the defalut." ; 
  opts.Comment(usage) ; 

  RealT delay       = opts.Real    ("delay", 0.0,               "The delay in seconds " ) ; 
  bool showTC       = opts.Boolean ("g",     false,             "Get current timecode " ) ; 
  bool verbose      = opts.Boolean ("v",     false,             "Be verbose ? " ) ; 
  StringC start     = opts.String  ("start", "now",             "The starting timecode [HH:MM:SS:FF , now] " ) ; 
  StringC end       = opts.String  ("end",   "",                "The ending timecode [HH:MM:SS:FF] ") ; 
  FilenameC outFile = opts.String  ("out",   "tmp1.4cif",       "The output filename for capture device 1 (.cif, .4cif,  .ppm, .avi, .jpg ) ") ; 
  FilenameC device  = opts.String  ("dev",   "@CSP:PCI,card:0", "The capture device (See RAVL virtual files for more details )" ) ; 
  UIntT  howMany    = opts.Int     ("n",     25,                "Duration of capture (number of frames to be grabbed), NB Cannot be used with real timecodes") ; 
  UIntT  frameStep  = opts.Int     ("step",  1,                 "Capture every nth frame " ) ; 

  opts.DependXor ("end n") ; 
  opts.Check() ;  


  Sleep (delay) ; 


  //: create a name for the timecode file 
  // -------------------------------------
  FilenameC tcFile ; 
  if ( outFile.PathComponent() == "" ) tcFile = outFile.BaseNameComponent()+".tc" ;
  else 
    tcFile = outFile.PathComponent()  + "/" + outFile.BaseNameComponent()+".tc" ; 
  cerr << "tcFile -= " << tcFile ; 
  OStreamC tcStream(tcFile) ; 




  // try to setup the capture device 
  // --------------------------------
  cerr << "\n   -  Trying to setup capture device " << device << " ....  " ; 
  DPIPortC<ImageT> inStream ; 
  if ( ! OpenISequence ( inStream, device, "", verbose) ) 
    { RavlIssueError ("\n   -  Failed to open input device, exiting .... " ) ; } 
  cerr << "\t\tdone  ! " ; 



 // setup some variables 
  // -----------------------  
  ImageT tmpImage  = inStream.Get(); 
  TimeCodeC timeNow   = inStream.GetAttr("timecode") ;
  DListC<ImageT> imgList ;
  DListC<TimeCodeC> tcList ; 


  // check timecode support for device 
  // -------------------------
  cerr << "\n   -  Checking Timecode Support .... " ;
  bool hasTimecode = false ; 
  bool useTimecode = false ;
  
  // does device support it 
  DListC<StringC> attList ; 
  inStream.GetAttrList(attList) ; 
  for ( DLIterC<StringC> iter(attList) ; iter ; iter++ ) 
    if ( iter.Data() == "timecode" ) 
      { hasTimecode = true ; break ; } 
  
  // See if a timecode can be found ; 
  if (timeNow!="00:00:00:00" && timeNow!="")  //: BUG - What if the timecode really is 00:00:00:00 ! 
    useTimecode = true ; 

  // display some status 
  if ( hasTimecode ) 
    cerr << "\n     -  Device supports timecodes " ; 
  else 
    cerr << "\n     -  Device does not support timecodes " ; 
  
  if ( useTimecode ) 
    cerr << "\n     -  Timecode read successfully from video stream " ; 
  else 
    cerr << "\n     -  Unable to read timecode from video stream " ; 

  
  // do some checks 
  if ( !useTimecode && ( opts.IsOnCommandLine("end") || start != "now" ) ) 
    RavlIssueError ("\n   -  Error unable to use timecodes, specify frame count instead and set -start to 'now' ") ; 
  

  
  //: Show Current timecode 
  // ----------------------------
  if ( showTC ) {
    if ( hasTimecode ) 
      cerr << "\n\n   -  The Current timecode is " << timeNow.ToText() << "\n\n" ; 
    else 
      cerr << "\n\n   - Option -g is invalid since device " << device 
	   << " does not have timecode support\n\n " ;
    return 0 ; 
  }
      


  // try to open the output stream 
  // ------------------------------
  DPOPortC<ImageT>  outStream ; 
  if ( ! OpenOSequence ( outStream, outFile, "", verbose) ) 
    { RavlIssueError ("\n\n   - Failed to open output file stream ") ; }



  // Wait for timecode 
  // -----------------
  cerr << "\n\n   -  Waiting for timecode ...."  ; 
  //: Wait for timecode 
  IntT statusStep = 0 ; 
  if ( start != "now" )
  while (timeNow < TimeCodeC(start)-1) {
    tmpImage = inStream.Get() ; 
    timeNow =  inStream.GetAttr("timecode") ;  
    ++ statusStep ; 
    
    // display status every 100th frame 
    TimeCodeC timeLeft (0) ;
    if (statusStep >= 100 ) {
      timeLeft = TimeCodeC(start) - timeNow ; 
      cerr << "\n    -  Time now: " << timeNow.ToText()  << "\t Waiting for: " << (TimeCodeC(start)-1).ToText() << "\t\tTime remaining: " 
	   << timeLeft.ToText() ;  
      statusStep = 0 ; } 
  }

  
  if ( ! hasTimecode ) cerr << "\n   - Device: " << device << " has no timecode support, unable to detect dropped frames " ;  



  //   Start capture 
  // -----------------
  // -----------------
  cerr << "\n\n   - Starting capture " ; 
 
  
  // using timecodes 
  // ----------------
  if ( useTimecode  ) 
    {
      TimeCodeC nextGrab = timeNow + 1 ; 
      TimeCodeC endTime = end ; 
      if ( !opts.IsOnCommandLine("end") ) endTime = nextGrab + howMany - 1; 
      cerr << "\n   -  First Grab will be: " << nextGrab.ToText() << "  \t and endtime is: " << endTime.ToText() ; 

      while ( true ) 
      {
	//Sleep ( .5) ; 
	//cerr << "\n Time now " << timeNow.ToText() << "end is " << endTime.ToText() << " cond = " << (timeNow>=nextGrab) ; 
	if (timeNow >= endTime) break ; 
	tmpImage = inStream.Get() ; 
	timeNow =  inStream.GetAttr("timecode") ;  
	// decide if we want this frame 
	if ( timeNow >= nextGrab ) 
	  {
	    if (verbose) cerr << "\n     -  Grabbed frame " << timeNow.ToText() << "\t expected: " << nextGrab.ToText() ; 

	    if ( timeNow != nextGrab ) {
	      cerr << "\n  *** Dropped frame with timecode " << nextGrab.ToText() 
		   << "\t Grabbing " << timeNow.ToText() << " instead !" ; 
	      nextGrab = timeNow ; 
	    }
	    imgList.InsLast(tmpImage) ; 
	    tcList.InsLast(timeNow) ; 
	    nextGrab += frameStep ; 
	  } 
      }
    }
  
  else { 
    // just count frames 
    // -----------------
    UIntT count = 1 ; 
    UIntT stepCount = frameStep ; 
    while (true) 
      {
	if (count > howMany) break ; 
	tmpImage = inStream.Get() ; 
	timeNow =  inStream.GetAttr("timecode") ;  
	// decide if we want this frame 
	if (stepCount >= frameStep) 
	  {
	    tcList.InsLast(timeNow) ; 
	    imgList.InsLast(tmpImage) ; 
	    stepCount = 1 ; 
	    count ++ ; 
	  }
	else 
	  ++ stepCount ; 
      }
    //nextGrab = timeNow + frameStep ; 
    tmpImage = inStream.Get() ; 
    timeNow = inStream.GetAttr("timecode") ;  
    }
  


  // Save image sequence 
  // ---------------------
  cerr << "\n   - Saving " << imgList.Size() << " frames .... " ; 
  for ( DLIterC<ImageT> iter (imgList) ; iter.IsElm() ; iter.Next() )
    outStream.Put( iter.Data() ) ; 
  cerr << "\t\t\t\tdone! " ; 

  // Save timecodes 
  // ----------------
  if ( hasTimecode ) 
    for (DLIterC<TimeCodeC> iter (tcList) ; iter.IsElm() ; iter.Next() ) 
      tcStream << iter.Data().ToText() << "\n" ; 


 return 0 ; 
}

RAVL_ENTRY_POINT(VGrab) ; 
