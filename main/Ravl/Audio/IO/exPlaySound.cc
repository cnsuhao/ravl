#include "Ravl/Audio/PlaySound.hh"
#include "Ravl/Option.hh"


using namespace RavlN;

int main(int nargs,char **argv) {
  OptionC opts(nargs,argv);
  StringC filename = opts.String("","/usr/share/sounds/info.wav","File to play. ");
  opts.Check();

  if(!RavlAudioN::PlaySound(filename)) {
    std::cerr << "Failed to play sound " << filename << "\n";
    return 1;
  }
  return 0;
}

