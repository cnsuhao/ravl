#include "Ravl/Option.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/EntryPnt.hh"
#include "Ravl/Image/DvDevice.hh"
#include "Ravl/DP/FileFormatIO.hh"

using namespace RavlImageN;

int doDvGrab(int argc, char **argv)
{  
  OptionC   opt(argc,argv);
  FilenameC     OutFile = opt.String("o", NULL, "output file");
  StringC       PpmImage = opt.String("ppm", "00:00:00:00", "Grab a single frame at given timecode");
  StringC       tcStart = opt.String("start", "00:00:00:00", "start timecode");
  StringC       tcEnd   = opt.String("end", "00:00:00:00", "end timecode");
  bool          Audio   = opt.Boolean("audio", false, "just grab the audio to a wav file");
  opt.Compulsory("o");
  opt.Check();
  
  DvDeviceC dev;
  
  //: Just grab a single image
  if(opt.IsOnCommandLine("ppm")) {
    ImageC<ByteRGBValueC>im = dev.grabFrame((TimeCodeC)PpmImage);
    Save(PpmImage, im, "", true);
  }
  //: Lets grab a whole sequence
  else {
    if(!dev.isPlaying()) {
      dev.Pause();
    }
    
    if(Audio) {
      dev.grabWav(OutFile, (TimeCodeC)tcStart, (TimeCodeC)tcEnd);
    } else {
      dev.grabSequence(OutFile, (TimeCodeC)tcStart, (TimeCodeC)tcEnd);
    }
    
    dev.Pause();
  }

  return 0;
}

//: This puts a wrapper around the main program that catches
//: exceptions and turns them into readable error messages.


RAVL_ENTRY_POINT(doDvGrab);
