// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2003, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
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

