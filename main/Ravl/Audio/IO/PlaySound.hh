#ifndef RAVLAUDIO_PLAYSOUND_HH
#define	RAVLAUDIO_PLAYSOUND_HH

#include "Ravl/String.hh"

namespace RavlAudioN {
  using namespace RavlN;
  
  //! Play a sound to the default audio device.
  bool PlaySound(const StringC &filename);

  //! Change default sound device.
  bool SetDefaultSoundDevice(const StringC &deviceName);
}

#endif

