#include "Ravl/Image/OpenCVConvert.hh"
#include "Ravl/Image/Image.hh"
#include "Ravl/Image/RealRGBValue.hh"
#include "Ravl/IO.hh"
#include "opencv/cv.h"

using namespace RavlN;
using namespace RavlImageN;
using namespace cv;
using namespace std;

int main()
{
  // set up a RAVL  image
  ImageC<RealRGBValueC> src(100,200);
  src.Fill(RealRGBValueC(200,100,150));
  for (IndexC i=20; i<=80; ++i) src[i][i] = RealRGBValueC(0,200,250);

  // convert it to an OpenCV Mat
  IplImage* im1 = 0;
  cout << "RAVL 2 CV: " << RavlImage2IplImage(src, im1) << endl;
  Mat mat1(im1);
  Mat mat2;

  // filter it
  GaussianBlur(mat1, mat2, Size(9,9), 3);

  // convert it back to a RAVL image and display it
  IplImage im2 = mat2;
  ImageC<ByteRGBValueC> out;
  cout << "CV 2 RAVL: " << IplImage2RavlImage(&im2, out) << endl;
  cvReleaseImage(&im1);
  RavlN::Save("@X", out);
}
