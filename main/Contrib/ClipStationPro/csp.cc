////////////////////////////////////////////////////
//! rcsid="$Id$"

#include "Ravl/Image/ImgIOCSP.hh"
#include "Ravl/Option.hh"
#include "Ravl/DP/FileFormatIO.hh"
#include "Ravl/Array2dIter2.hh"
#include "Ravl/DP/Converter.hh"
#include "Ravl/DP/SequenceIO.hh"

#include "Ravl/DP/ThreadPipe.hh"
#include "Ravl/DP/MTIOConnect.hh"
#include "Ravl/DP/Compose.hh"

using namespace RavlImageN;

extern ImageC<ByteT> ByteYUV422ImageCT2ByteImageCT(const ImageC<ByteYUV422ValueC> &dat);

int main(int nargs,char **argv) {
  OptionC opts(nargs,argv);
  StringC dev = opts.String("d","PCI,card:1","Device to use.");
  bool seq = opts.Boolean("s",false,"Sequence. ");
  StringC out = opts.String("","@X","Output");
  opts.Check();
  cerr << "Opening device.\n";
  
  DPIImageClipStationProBodyC<ByteYUV422ValueC> cspio(dev,ImageRectangleC(576,720));
  
  DPOPortC<ImageC<ByteYUV422ValueC> > outp;

  if(!OpenOSequence(outp,out)) {
    cerr << "Failed to open output. \n";
    return 1;
  }
  
  do {
    ImageC<ByteYUV422ValueC> img = cspio.Get();
    cerr << "Save image 1\n";
#if 1
    if(!outp.Put(img)) {
      cerr << "Failed to save image. \n";
      return 1;
    }
#endif
  } while(seq) ;
  
#if 0
  ImageRectangleC rect(720,576);
  ClipStatioProDeviceC csp(dev,rect);
  
  char data[720][576][3];
  int i = 5;
  do {
    csp.GetFrame(data,720,576);
    csp.PutFrame(data,720,576);
  } while(i-- >0);
#else
  
#endif
  
  cerr << "Done.\n";
  return 0;
}
