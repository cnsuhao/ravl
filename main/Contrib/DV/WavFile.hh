#ifndef WavFile_HH
#define WavFile_HH
////////////////////////////////////////////////////////////////////////////
//! author="Kieron J Messer"
//! date="24/9/100"
//! lib=DvDevice
//! docentry="Drivers.Linux133194"
//! rcsid="$Id$"
  
class istream;
class ostream;

///// PalFrameC ////////////////////////////////////////////////////////
//! userlevel=Normal
//: Put a brief description of your class
// Put a more detailed description of your class here.  You use embedded html
// to make it clearer in the html documentation
#include"Ravl/Stream.hh"
#include"Ravl/Array1d.hh"
#include"Ravl/OS/Filename.hh"

#define WAVE_FORMAT_PCM  0x0001

namespace RavlImageN {

  typedef unsigned short UShortT;
  typedef unsigned long ULongT;
  
  using namespace RavlN;

  class WavFileC 
  {
  public:
    WavFileC() {}
    //: Default constructor
    
    WavFileC(FilenameC & fname, UIntT bitsPerSample, UIntT sampleRate, UIntT numChannels);
    
    void write(const Array1dC<char> & data);
    //: write a frame of audio data to file  
    
    void Close();
    //: close the wave file, must be called
    
  protected:
    FilenameC fname;
    OStreamC ofs;
    UIntT BitsPerSample;
    UIntT SampleRate;
    UIntT NumChannels;
    UIntT BytesWritten;
  };

} // end namespace 



#endif
