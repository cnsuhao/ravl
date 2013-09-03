#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
using namespace RavlN;
using namespace RavlImageN;
int main (int argc, char* argv[]) {
  DPIPortC<ImageC<ByteRGBValueC> > in;
  if(!OpenISequence(in, argv[1])) {
    // Failed to open input file. Report an error...
    exit(-1);
  }
  DPOPortC<ImageC<ByteRGBValueC> > out;
  if(!OpenOSequence(out, argv[2])) exit (-2);
  ImageC<ByteRGBValueC> im;
  while(in.Get(im)) out.Put(im);
}
