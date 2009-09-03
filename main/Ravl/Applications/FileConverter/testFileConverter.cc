#include "Ravl/Stream.hh"
#include "Ravl/String.hh"
#include "Ravl/OS/Directory.hh"
#include "Ravl/OS/Filename.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/ByteRGBValue.hh"
#include "Ravl/Array2dIter.hh"
#include "Ravl/IO.hh"
#include "Ravl/Tuple3.hh"
#include "Ravl/DList.hh"
#include "Ravl/DLIter.hh"
#include "Ravl/Math.hh"

using namespace RavlN;
using namespace RavlImageN;

int testImageConvert(const Tuple3C<StringC,StringC,UIntT>& imFiles) {
  DirectoryC dir = StringC(getenv("PROJECT_OUT"))+"/share/RAVL/testData/";
  FilenameC in =  dir + imFiles.Data1();
  FilenameC out = dir + imFiles.Data2();
  if (!in.Exists())  return __LINE__;
  FilenameC prog = StringC(getenv("PROJECT_OUT"))+"/bin/conv";
  if (!prog.Exists()) {
    cout << "conv not compiled in this build\n";
    return 0;
  }
  StringC cmd = prog + " -s " + in + " " + out;
  if (system(cmd.chars()) != 0) return __LINE__;
  ImageC<ByteRGBValueC> i, o;
  Load(in, i); Load(out, o);
  if (i.Frame().Size() != o.Frame().Size()) return __LINE__;
  for (Array2dIter2C<ByteRGBValueC,ByteRGBValueC>p(i,o); p; ++p)
    for (UIntT c=0; c<3; ++c) 
      if (Abs((IntT)p.Data1()[c] - (IntT)p.Data2()[c]) > (IntT)imFiles.Data3()) {
        cout << p.Index() << " "<< Abs((IntT)p.Data1()[c] - (IntT)p.Data2()[c]);
        return __LINE__;
      }
  return 0;
}


int main() {
  int ln;
  DListC<Tuple3C<StringC,StringC,UIntT> > imPair;
  // Just add whatever conversion you like, with optional conversion accuracy
  imPair.Append(Tuple3C<StringC,StringC,UIntT>("in0.ppm", "out0.ppm",  0));
  imPair.Append(Tuple3C<StringC,StringC,UIntT>("in0.ppm", "out0.png",  0));
  imPair.Append(Tuple3C<StringC,StringC,UIntT>("in0.ppm", "out0.jpg", 15));
  for (DLIterC<Tuple3C<StringC,StringC,UIntT> > i(imPair); i; ++i) {
    if ((ln = testImageConvert(*i)) != 0) {
      cerr << "Test for \"conv\" failed for " << i->Data1() << " -> " << i->Data2() << " on line " << ln << "\n";
      return 1;
    }
  }
  cerr << "Test passed ok. \n";
  return 0;
}

