#include "Ravl/DP/SequenceIO.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/RealRGBValue.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Image/DeinterlaceStream.hh"
#include "Ravl/OS/Date.hh"
#include "Ravl/OS/CharIO.hh"
#include "Ravl/Image/Font.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/Option.hh"
#include <stdio.h>

using namespace RavlN;
using namespace RavlImageN;

int main (int argc, char* argv[])
{
  RavlN::OptionC opts(argc,argv);
  bool waitForKey = !opts.Boolean("r","Run file continuously.");
  StringC inFile = opts.String("","","Input file");
  StringC outFile = opts.String("","","Output file");
  bool verbose = opts.Boolean("v", "Adds diagnostic information to display");
  opts.Check();

  DPISPortC<ImageC<ByteRGBValueC> > in;
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
  if (waitForKey) {
    UInt64T c(0x484f1b);
    SizeT oldi = 0;
    while (c != 'q') {
      switch (c) {
      case 0x435b1b: ++i; break;
      case 0x445b1b: --i; break;
      case 0x484f1b: i=0; break;
      }
      if (di.GetAt((UIntT)i, im)) {
        cout << "good :)" << endl;
        if (verbose)
          DrawText<ByteRGBValueC>(im, ByteRGBValueC(255,255,255), Index2dC(15,15), StringC(i));
        out.Put(im);
        oldi = i;
      }
      else { i = oldi;
        cout << "bad :(" << endl;}
      c = GetKeypress();
    }
  }
  else {
    while (di.Get(im)) {
      if (verbose)
        DrawText<ByteRGBValueC>(im, ByteRGBValueC(255,255,255), Index2dC(15,15), StringC(i++));
      out.Put(im);
      Sleep(0.02);
      ++i;
    }
  }
}

