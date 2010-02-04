#include "Ravl/String.hh"
#include "Ravl/Image/WarpAffine.hh"
#include "Ravl/Image/WarpScale2d.hh"
#include "Ravl/Array2dIter2.hh"

using namespace RavlN; 
using namespace RavlImageN;

int TestTransform1() {
  //create image
  ImageC<float> img(5, 5);
  img.Fill(0.);

  img[2][2] = 1;


  Affine2dC tr(RavlN::Vector2dC(1, 1), 0, RavlN::Vector2dC(0, 0));
  //cout << "transform:" << tr << endl;

  WarpAffineC<float, float, float, PixelMixerAssignC<float, float>, SampleNearestC<float, float> >
                       warp(ImageRectangleC(5, 5), tr);

  ImageC<float> res;
  warp.Apply(img, res);

  for(SizeT r = 0; r < 5; r++) {
    for(SizeT c = 0; c < 5; c++) {
      //printf("%3g ", res[r][c]);
      if(res[r][c] != img[r][c]) {
        cerr << "Error at " << r << "  " << c << endl;
        return 1;
      }
    }
    //printf("\n");
  }
  return 0;
}

int TestTransform1b() {
  //create image
  ImageC<float> img(5, 5);
  img.Fill(0.);

  img[2][2] = 1;


  Affine2dC tr(RavlN::Vector2dC(1, 1), 0, RavlN::Vector2dC(0, 0));
  cout << "transform:" << tr << endl;

  WarpAffineC<float, float, float, PixelMixerAssignC<float, float>, SampleBilinearC<float, float> >
                       warp(ImageRectangleC(5, 5), tr);

  ImageC<float> res;
  warp.Apply(img, res);

  for(SizeT r = 0; r < 5; r++) {
    for(SizeT c = 0; c < 5; c++) {
      printf("%3g ", res[r][c]);
      if(res[r][c] != img[r][c]) {
        cerr << "Error at " << r << "  " << c << endl;
        return 1;
      }
    }
    printf("\n");
  }
  return 0;
}

int TestTransform2() {
  //create image
  ImageC<float> img(16, 16);
  img.Fill(0.);

  for(SizeT r = 4; r < 8; r++) {
    for(SizeT c = 4; c < 8; c++) {
      img[r][c] = 1;
    }
  }
  for(SizeT r = 0; r < 16; r++) {
    for(SizeT c = 0; c < 16; c++) {
      //printf("%3g ", img[r][c]);
    }
    //printf("\n");
  }

  Affine2dC tr(RavlN::Vector2dC(4, 4), 0, RavlN::Vector2dC(0, 0));
  //cout << "transform:" << tr << endl;

  WarpAffineC<float, float, float, PixelMixerAssignC<float, float>, SampleNearestC<float, float> >
                       warp(ImageRectangleC(4, 4), tr);

  ImageC<float> res;
  warp.Apply(img, res);

  for(SizeT r = 0; r < 4; r++) {
    for(SizeT c = 0; c < 4; c++) {
      //printf("%3g ", res[r][c]);
      if(res[r][c] != 0 && r != 1 && c != 1) {
        cerr << "Error at " << r << "  " << c << endl;
        return 1;
      }
    }
    //printf("\n");
  }
  if(res[1][1] != 1) {
    cerr << "Error at 1   1\n";
    return 1;
  }
  return 0;
}

int TestTransform3() {
  //create image
  ImageC<float> img(16, 16);
  img.Fill(0.);

  for(SizeT r = 4; r < 8; r++) {
    for(SizeT c = 4; c < 8; c++) {
      img[r][c] = 1;
    }
  }
  for(SizeT r = 0; r < 16; r++) {
    for(SizeT c = 0; c < 16; c++) {
      //printf("%3g ", img[r][c]);
    }
    //printf("\n");
  }

  ImageC<float> res(4, 4);
  WarpScaleBilinear(img, Vector2dC(4,4), res);

  for(SizeT r = 0; r < 4; r++) {
    for(SizeT c = 0; c < 4; c++) {
      //printf("%3g ", res[r][c]);
      if(res[r][c] != 0 && r != 1 && c != 1) {
        cerr << "Error at " << r << "  " << c << endl;
        return 1;
      }
    }
    //printf("\n");
  }
  if(res[1][1] != 1) {
    cerr << "Error at 1   1\n";
    return 1;
  }

  return 0;
}

int TestTransform4() {
  printf("TestTransform4\n");
  //create image
  ImageC<float> img(16, 16);
  img.Fill(0.);

  for(SizeT r = 4; r < 8; r++) {
    for(SizeT c = 4; c < 8; c++) {
      img[r][c] = 1;
    }
  }
  for(SizeT r = 0; r < 16; r++) {
    for(SizeT c = 0; c < 16; c++) {
      printf("%3g ", img[r][c]);
    }
    printf("\n");
  }

  Vector2dC scale(1.1,1.1);

  ImageC<float> res1(10, 10);
  WarpScaleBilinear(img, scale, res1);

  Affine2dC tr(scale, 0, RavlN::Vector2dC(0, 0));
  cout << "transform:" << tr << endl;

  WarpAffineC<float, float, float, PixelMixerAssignC<float, float>, SampleBilinearC<float, float> >
                       warp(ImageRectangleC(10, 10), tr);
  warp.SetMidPixelCorrection(false);

  ImageC<float> res2;
  warp.Apply(img, res2);

  for(SizeT r = 0; r < 10; r++) {
    for(SizeT c = 0; c < 10; c++) {
      printf("%6g ", res1[r][c]);
    }
    printf("\n");
  }

  printf("\n");
  for(SizeT r = 0; r < 10; r++) {
    for(SizeT c = 0; c < 10; c++) {
      printf("%6g ", res2[r][c]);
    }
    printf("\n");
  }

  printf("\n");
  for(SizeT r = 0; r < 10; r++) {
    for(SizeT c = 0; c < 10; c++) {
      printf("%6g ", res1[r][c] - res2[r][c]);
      /*if(res[r][c] != 0 && r != 1 && c != 1) {
        cerr << "Error at " << r << "  " << c << endl;
        return 1;
      }*/
    }
    printf("\n");
  }
  /*if(res[1][1] != 1) {
    cerr << "Error at 1   1\n";
    return 1;
  }*/

  return 0;
}


int main(int argc, char **argv)
{
  if(TestTransform1() > 0) {
    cerr << "Error in WarpAffineC test 1\n";
    return 1;
  }

  if(TestTransform1b() > 0) {
    cerr << "Error in WarpAffineC test 1b\n";
    return 1;
  }

  if(TestTransform2() > 0) {
    cerr << "Error in WarpAffineC test 2\n";
    return 1;
  }

  if(TestTransform3() > 0) {
    cerr << "Error in WarpAffineC test 3\n";
    return 1;
  }

  if(TestTransform4() > 0) {
    cerr << "Error in WarpAffineC test 3\n";
    return 1;
  }

  return 0;
}


