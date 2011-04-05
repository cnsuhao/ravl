#include "Ravl/Image/ImagePyramid.hh"

using namespace RavlN;
using namespace RavlImageN;

int main() 
{
  ImageC<ByteT> img(100,100);
  for (Array2dIterC<ByteT> i(img); i; ++i)
    *i = i.Index()[0] + i.Index()[1];

  cout << "\nPyramid without image subsampling:\n";
  ImagePyramidC<ByteT> pyramid(img, Sqrt(2), 4, false);
  for (CollectionIterC<Tuple3C<RealT,RealT,ImageC<ByteT> > > l(pyramid.Images()); l; ++l)
    cout << "Frame: " << l->Data3().Frame() << endl
         << "Pixel scale: " << l->Data2() << endl
         << "Filter scale: " << l->Data1() << endl;

  cout << "\nPyramid including image subsampling:\n";
  // Also different indexing style
  pyramid = ImagePyramidC<ByteT>(img, Sqrt(2), 4, true);
  for (IntT l=0; l<pyramid.Images().Size(); ++l) {
    Tuple3C<RealT,RealT,ImageC<ByteT> > img(pyramid.Images()[l]);
    cout << "Frame: " << img.Data3().Frame() << endl
         << "Pixel scale: " << img.Data2() << endl
         << "Filter scale: " << img.Data1() << endl;
  }
  cout << "\nMulti-octave pyramid (SIFT-style):\n";
  // create the octaves using subsampling:
  pyramid = ImagePyramidC<ByteT>(img, 3, true);
  for (IntT level=0; level<pyramid.Images().Size(); ++level) {
    // create the images in between (without subsampling)
    ImageC<ByteT> octaveImg = pyramid.Images()[level].Data3();
    ImagePyramidC<ByteT> subPyramid(octaveImg, Pow(2.0,1.0/8.0), 8, false);
    for (CollectionIterC<Tuple3C<RealT,RealT,ImageC<ByteT> > > l(subPyramid.Images()); l; ++l)
      cout << "Frame: " << l->Data3().Frame() << endl
           << "Pixel scale: " << l->Data2() * Pow(2.0, level) << endl
           << "Filter scale: " << l->Data1() * Pow(2.0, level) << endl;
  }
  
  return 0;
}
