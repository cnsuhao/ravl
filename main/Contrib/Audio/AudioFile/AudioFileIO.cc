// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlAudioFile

#include "Ravl/Audio/AudioFileIO.hh"
#include "Ravl/TypeName.hh"

namespace RavlAudioN {
  using namespace RavlN;
  
  //: Constructor.
  
  AudioFileBaseC::AudioFileBaseC(const StringC &fileName,int channel,bool forInput,const type_info &ndtype)
    : handle(0),
      setup(0),
      frameSize(0),
      channel(0),
      dtype(0)
  {
    if(forInput)
      IOpen(fileName,channel,ndtype);
    else
      OOpen(fileName,channel,ndtype);      
  }

  //: Destructor.
  
  AudioFileBaseC::~AudioFileBaseC() {
    if(handle != 0)
      afCloseFile(handle);
    if(setup != 0)
      afFreeFileSetup(setup);
  }
  
  //: Setup IO.
  
  bool AudioFileBaseC::SetupChannel(int channel,const type_info &ndtype) {
    if(ndtype == typeid(Int16T)) {
      afSetVirtualChannels(handle,channel,1);
      afSetVirtualSampleFormat(handle,channel,AF_SAMPFMT_TWOSCOMP, 16);
    } else {
      cerr << "AudioFileBaseC::IOpen(), Don't know to handle type " << TypeName(ndtype) << "\n";
      return false;
    }
    return true;
  }
  
  //: Open audio device.
  
  bool AudioFileBaseC::IOpen(const StringC &fn,int nchannel,const type_info &ndtype) {
    if(ndtype != typeid(Int16T))
      return false;
    dtype = &ndtype;
    channel = nchannel;
    fileName = fn;
    handle = afOpenFile(fn.chars(),"r",0);
    if(handle != 0)
      return false;
    return SetupChannel(channel,ndtype);
  }
  
  //: Open audio device.
  
  bool AudioFileBaseC::OOpen(const StringC &fn,int nchannel,const type_info &ndtype) {
    if(ndtype != typeid(Int16T))
      return false;
    dtype = &ndtype;
    channel = nchannel;
    setup = afNewFileSetup();
    fileName = fn;
    if(setup == 0)
      return false;
    afInitChannels(setup, AF_DEFAULT_TRACK, 1);
    afInitSampleFormat(setup, AF_DEFAULT_TRACK, AF_SAMPFMT_TWOSCOMP, 16);
    StringC tfn(fn);
    StringC ext = tfn.after('.');
    
    if(ext == "wav") {
      afInitFileFormat(setup, AF_FILE_WAVE);
    } else if(ext == "aiff") {
      afInitFileFormat(setup, AF_FILE_AIFF);
    } else if(ext == "aiffc") {
      afInitFileFormat(setup, AF_FILE_AIFFC);
    } else if(ext == "bicsf") {
      afInitFileFormat(setup, AF_FILE_BICSF);
    } else if(ext == "nextsnd") {
      afInitFileFormat(setup, AF_FILE_NEXTSND);
    } else if(ext == "au") {
      afInitFileFormat(setup, AF_FILE_RAWDATA);
    }
    return true;
  }
  
  //: Set number of bits to use in samples.
  // returns actual number of bits.
  
  bool AudioFileBaseC::SetSampleBits(IntT bits) {
    return false;
  }
  
  //: Set frequency of samples
  // Returns actual frequency.
  
  bool AudioFileBaseC::SetSampleRate(IntT rate) {
    return false;
  }
  
  //: Get number of bits to use in samples.
  // returns actual number of bits.
  
  bool AudioFileBaseC::GetSampleBits(IntT &bits) {
    return false;
  }
  
  //: Get frequency of samples
  // Returns actual frequency.
  
  bool AudioFileBaseC::GetSampleRate(IntT &rate) {
    return false;
  }
  
  //: Read bytes from audio stream.
  // Returns false if error occured.
  
  bool AudioFileBaseC::Read(void *buf,IntT &len) {
    RavlAssert((len % frameSize) == 0);
    IntT ret = afReadFrames(handle,AF_DEFAULT_TRACK,buf,len / frameSize);
    if(ret < 0) {
      //...
      cerr << "AudioFileBaseC::Read(), Error reading data." << ret << "\n";
      return false;
    }
    return true;
  }
  
  //: Write bytes to audio stream.
  // Returns false if error occured.
  
  bool AudioFileBaseC::Write(const void *buf,IntT len) {
    if(handle == 0) {
      RavlAssert(setup != 0);
      handle = afOpenFile(fileName.chars(),"w",setup);
      if(handle == 0)
	throw DataNotReadyC("AudioFileBaseC::Write(), Failed to open file for writting. ");
      SetupChannel(channel,*dtype);
    }
    RavlAssert((len % frameSize) == 0);
    IntT ret = afWriteFrames(handle,AF_DEFAULT_TRACK,buf,len / frameSize);
    if(ret < 0) {
      //...
      cerr << "AudioFileBaseC::Write(), Error writting data." << ret << "\n";
      return false;
    }
    return true;
  }
  
}
