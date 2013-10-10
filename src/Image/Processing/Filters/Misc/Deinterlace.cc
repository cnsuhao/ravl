#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/RealRGBValue.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Image/DeinterlaceStream.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/Image/Font.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/Option.hh"
#include <stdio.h>

using namespace RavlN;
using namespace RavlImageN;

int main (int argc, char* argv[])
{
  RavlN::OptionC opts(argc,argv);
  bool waitForKey = opts.Boolean("wfk",false,"Wait for keypress.");
  StringC inFile = opts.String("","","Input file");
  StringC outFile = opts.String("","","Output file");
  opts.Check();
  if (argc < 3) exit(-1);
  DPIPortC<ImageC<ByteRGBValueC> > in;
  if(!OpenISequence(in, inFile)) {
    RavlError("Failed to open input '%s' ",inFile.c_str());
    exit(-2);
  }
  DPOPortC<ImageC<ByteRGBValueC> > out;
  if(!OpenOSequence(out, outFile)) {
    RavlError("Failed to open output '%s' ",outFile.c_str());
    exit (-3);
  }

  DeinterlaceStreamC<ByteRGBValueC> di(in);
  ImageC<ByteRGBValueC> im;
  int i(0);
  while(di.Get(im)) {
    //Sleep(1);
    DrawText<ByteRGBValueC>(im, ByteRGBValueC(255,255,255), Index2dC(15,15), StringC(i++));
    out.Put(im);
    if(waitForKey) {
      getchar();
    }
  }
}

