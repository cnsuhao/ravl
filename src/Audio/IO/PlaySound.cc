
#include "Ravl/Audio/PlaySound.hh"
#include "Ravl/IO.hh"
#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/Audio/Types.hh"
#include "Ravl/DP/AttributeType.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlAudioN {

  static StringC g_defaultSoundOutput = "@DEVAUDIO:/dev/dsp";

  //! Change default sound device.
  
  bool SetDefaultSoundDevice(const StringC &deviceName) {
    g_defaultSoundOutput = deviceName;
    return true;
  }
  
  //! Play a sound to the default audio device.
  bool PlaySound(const StringC &filename) {

    // open the input port
    DPIPortC<SampleElemC < 2, Int16T> > in;
    if(!OpenISequence(in,filename)) {
      cerr << "Failed to open input sound file : " << filename << "\n";
      return false;
    }

    // now lets setup an output port
    DPOPortC<SampleElemC < 2, Int16T> > out;
    if(!OpenOSequence(out,g_defaultSoundOutput)) {
      cerr << "Failed to open sound output : " << g_defaultSoundOutput << "\n";
      return false;
    }

#if DODEBUG
    // lets look at the attributes available
    DListC<AttributeTypeC> attrList;
    in.GetAttrTypes(attrList);
    std::cout << "Available Attributes are :\n" << attrList;
#endif

    // lets get some attributes
    RealT sampleRate;
    IntT sampleBits;
    in.GetAttr("samplerate", sampleRate);
    in.GetAttr("samplebits", sampleBits);
    ONDEBUG(cout << "Sample rate is " << sampleRate << " and sample bits is " << sampleBits << "\n");

    // now lets read data from file and play to device
    out.SetAttr("samplerate", sampleRate);
    SampleElemC < 2, Int16T> sample;
    while(in.Get(sample)) {
      out.Put(sample); // play sample
    }
    
    return true;
  }
}

