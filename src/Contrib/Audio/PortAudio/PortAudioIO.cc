// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, Charles Galambos
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlPortAudio
//! file="Ravl/Contrib/Audio/PortAudio/PortAudioIO.cc"

#include "Ravl/Audio/PortAudioIO.hh"
#include "Ravl/Audio/PortAudioFormat.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/DP/AttributeType.hh"
#include "Ravl/SysLog.hh"

#include <string.h>

#define DODEBUG 1
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlAudioN {
  using namespace RavlN;
  
  //: Constructor.
  
  PortAudioBaseC::PortAudioBaseC(const StringC &fileName,int nchannel,bool nforInput,const type_info &ndtype)
    : m_stream(0),
      m_doneSetup(false),
      m_latency(0),
      frameSize(FileFormatPortAudioBodyC::FrameSize(ndtype)),
      channel(nchannel),
      dtype(&ndtype),
      sampleRate(16000),
      forInput(nforInput)
  {
    RavlN::MutexLockC lock(PortAudioMutex());
    const PaDeviceInfo *devInfo = Pa_GetDeviceInfo(channel);
    if(devInfo == 0) {
      RavlError("No information found about device!");
      // FIXME:- Throw exception?
      return ;
    }
    sampleRate = devInfo->defaultSampleRate;
    if(nforInput) {
      m_latency = devInfo->defaultHighInputLatency;
    } else {
      m_latency = devInfo->defaultHighOutputLatency;
    }
    ONDEBUG(RavlDebug("Default latency %d frame size %u ",m_latency,(unsigned) frameSize));
  }

  //: Destructor.
  
  PortAudioBaseC::~PortAudioBaseC() {
    if(m_stream != 0) {
      RavlN::MutexLockC lock(PortAudioMutex());
      Pa_CloseStream(m_stream);
      m_stream = 0;
    }
  }
  
  //: Build Attributes 
  bool PortAudioBaseC::BuildAttributes ( AttributeCtrlBodyC & attributes )
  {
    // build parent attributes
    AudioIOBaseC::BuildAttributes( attributes ) ;

    // cant set attributes when reading from a file
    if ( forInput) {
      AttributeTypeC
        sampleRate = attributes.GetAttrType("samplerate") ,
        sampleBits  = attributes.GetAttrType("samplebits") ;
      if ( sampleRate.IsValid() ) sampleRate.CanWrite(false) ;
      if ( sampleBits.IsValid() ) sampleBits.CanWrite(false) ;
    }
    return true ;
  }
  
  //: Open audio device.
  
  // Setup parameters
  bool PortAudioBaseC::SetupParameters(PaStreamParameters &inputParameters) {
    memset(&inputParameters,0,sizeof(inputParameters));

    inputParameters.device = channel;
    inputParameters.channelCount = FileFormatPortAudioBodyC::Channels(*dtype);
    inputParameters.sampleFormat = FileFormatPortAudioBodyC::IsFloat(*dtype) ? paFloat32 : paInt16;
    inputParameters.suggestedLatency = m_latency ;

    return true;
  }

  bool PortAudioBaseC::IOpen() {
    ONDEBUG(RavlDebug("PortAudioBaseC::IOpen(), Called. "));

    PaStreamParameters inputParameters;
    SetupParameters(inputParameters);

    RavlN::MutexLockC lock(PortAudioMutex());

    PaError err = Pa_OpenStream(&m_stream,
            &inputParameters,
            0,
            sampleRate,
            0,
            paNoFlag,
            0, // User callback
            0  // User data
             );

    if(err != paNoError) {
      RavlError("Failed to open input.");
      return false;
    }

    return true;
  }
  
  //: Open audio device.
  
  bool PortAudioBaseC::OOpen() {
    PaStreamParameters outputParameters;
    SetupParameters(outputParameters);

    RavlN::MutexLockC lock(PortAudioMutex());

    PaError err = Pa_OpenStream(&m_stream,
            0,
            &outputParameters,
            sampleRate,
            0,
            paNoFlag,
            0, // User callback
            0  // User data
             );

    if(err != paNoError) {
      RavlError("Failed to open input.");
      return false;
    }
    return true;
  }
  
  //: Set number of bits to use in samples.
  // returns actual number of bits.
  
  bool PortAudioBaseC::SetSampleBits(IntT bits) {
    ONDEBUG(RavlDebug("SetSampleBits called."));
    return false;
  }
  
  //: Set frequency of samples
  // Returns actual frequency.
  
  bool PortAudioBaseC::SetSampleRate(RealT rate) {
    ONDEBUG(RavlDebug("SetSampleRate called."));
    PaStreamParameters ioParameters;
    SetupParameters(ioParameters);
    PaError err;
    RavlN::MutexLockC lock(PortAudioMutex());
    if(forInput) {
      err = Pa_IsFormatSupported( &ioParameters, 0, rate);
    } else {
      err = Pa_IsFormatSupported( 0, &ioParameters, rate);
    }
    lock.Unlock();
    if(err != paNoError)
       return false;
    sampleRate = rate;
    return true;
  }
  
  //: Get number of bits to use in samples.
  // returns actual number of bits.
  
  bool PortAudioBaseC::GetSampleBits(IntT &bits) {
    ONDEBUG(RavlDebug("GetSampleBits(IntT &bits)"));

    return false;
  }

  //: Get frequency of samples
  // Returns actual frequency.
  
  bool PortAudioBaseC::GetSampleRate(RealT &rate) {
    ONDEBUG(RavlDebug("GetSampleRate(RealT &rate)")) ;
    if(m_stream == 0) {
      rate = sampleRate;
      return true;
    }
    RavlN::MutexLockC lock(PortAudioMutex());
    const PaStreamInfo *streamInfo =Pa_GetStreamInfo(m_stream);
    lock.Unlock();
    if(streamInfo == 0) {
      RavlError("Failed to find information about stream.");
      return false;
    }
    rate = streamInfo->sampleRate;
    return true;
  }
  
  //: Read bytes from audio stream.
  // Returns false if error occurred.
  
  bool PortAudioBaseC::Read(void *buf,IntT &len) {
    if(!m_doneSetup) {
      if(!IOpen())
        return false;
    }
    if(m_stream == 0)
      return false;
    if((len % frameSize) != 0) {
      RavlError("Miss aligned read!");
      return false;
    }
    int blocks = len / frameSize;
    PaError err = Pa_ReadStream(m_stream, buf, blocks);
    if(err) {
      RavlError("Error writing to stream. ");
      return false;
    }
    return true;
  }
  
  //: Write bytes to audio stream.
  // Returns false if error occurred.
  
  bool PortAudioBaseC::Write(const void *buf,IntT len) {
    if(!m_doneSetup) {
      if(!OOpen())
        return false;
    }
    if(m_stream == 0)
      return false;
    if((len % frameSize) != 0) {
      RavlError("Miss aligned write!");
      return false;
    }
    int blocks = len / frameSize;

    PaError err = Pa_WriteStream(m_stream, buf, blocks);
    if(err) {
      RavlError("Error writing to stream. ");
      return false;
    }
    return true;
  }


  //: Seek to location in stream.
  
  bool PortAudioBaseC::Seek(UIntT off) {
    return false;
  }
  
  //: Find current location in stream.
  // May return ((UIntT) (-1)) if not implemented.
  
  UIntT PortAudioBaseC::Tell() const {
    return 0;
  }
  
  //: Find the total size of the stream.  (assuming it starts from 0)
  // May return ((UIntT) (-1)) if not implemented.
  
  UIntT PortAudioBaseC::Size() const {
    return 0;
  }
  
}
