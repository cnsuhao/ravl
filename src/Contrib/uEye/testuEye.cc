
#include "Ravl/Image/ImgIOuEye.hh"
#include <iostream>
#include "Ravl/IO.hh"
#include "Ravl/Option.hh"
using RavlN::ByteT;

int main(int nargs,char **argv) {
  RavlN::OptionC opts(nargs,argv);
  int camId = opts.Int("c",0,"Camera id.");
  opts.Check();
  RavlImageN::ImgIOuEyeC<ByteT> imgStrm(camId);
  int i = 0;
  while(1) {
    RavlImageN::ImageC<ByteT> img;
    if(!imgStrm.Get(img)) {
      std::cerr << "Failed to get image. \n";
      break;
    }
    //RavlN::Save(RavlN::StringC("test") + RavlN::StringC(i++) + ".ppm",img);
    RavlN::Save("@X",img);
    //break;
  }
  return 0;
}

