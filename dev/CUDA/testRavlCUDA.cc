
#include "Ravl/UnitTest.hh"
#include "Ravl/CUDA/Array.hh"
#include "cuda_runtime.h"

int testCArray1d();
int testCArray2d();

int main(int nargs,char **argv) {

  RavlN::SysLogFileAndLine(true);

  RAVL_RUN_TEST(testCArray1d());
  RAVL_RUN_TEST(testCArray2d());

  return 0;
}

int testCArray1d()
{

  RavlN::CUDAN::CArray1dC<float> anArray(128);
  RavlN::CUDAN::CStreamC cstrm;

  anArray.SetZero(cstrm);

  cudaError_t state = ::cudaGetLastError();
  RAVL_TEST_TRUE(state == 0);

  RavlN::SArray1dC<float> num(8);
  for(unsigned i = 0;i < num.Size();i++)
    num[i] = i;

  anArray.Set(num,cstrm);
  state = ::cudaGetLastError();
  RAVL_TEST_TRUE(state == 0);

  RavlN::SArray1dC<float> num2(8);
  num2.Fill(0);
  anArray.Get(num2,cstrm);
  cstrm.Wait();
  state = ::cudaGetLastError();
  RAVL_TEST_TRUE(state == 0);


  for(unsigned i = 0;i < num.Size();i++) {
    //RavlDebug("%f %f ",num[i],num2[i]);
    RAVL_TEST_ALMOST_EQUALS(num[i],num2[i],1e-12);
  }

  //anArray.SetZero();
  anArray.Fill(0.5,cstrm);

  num2.Fill(0);
  state = ::cudaGetLastError();
  RavlDebug("State:%u ",state);
  RAVL_TEST_TRUE(state == 0);

  anArray.Get(num2,cstrm);

  state = ::cudaGetLastError();
  RAVL_TEST_TRUE(state == 0);

  cstrm.Wait();
  for(unsigned i = 0;i < num.Size();i++) {
    //RavlDebug("%f %f ",0.5,num2[i]);
    RAVL_TEST_ALMOST_EQUALS(0.5,num2[i],1e-12);
  }

  // Test larger arrays.

  RavlInfo("Testing fill of a larger array");
  for(unsigned j = 1;j < 10000000;j++) {
    RavlN::CUDAN::CArray1dC<float> aLargeArray(j);
    aLargeArray.SetZero(cstrm);
    aLargeArray.Fill(0.1,cstrm);
    RavlN::SArray1dC<float> tmp(aLargeArray.Size());
    aLargeArray.Get(tmp);
    for(unsigned i = 0;i < tmp.Size();i++) {
      if(RavlN::Abs(tmp[i] - 0.1) > 1e-8) {
        RavlDebug("%u -> %u : %f ",j,i,tmp[i]);
      }
      RAVL_TEST_ALMOST_EQUALS(0.1,tmp[i],1e-8);
    }
    j += j / 100;
  }

  return 0;
}

int testCArray2d()
{
  RavlN::CUDAN::CArray2dC<float> anArray(20,20);
  RavlN::CUDAN::CStreamC cstrm;
  anArray.SetZero(cstrm);

  RavlN::SArray2dC<float> num(16,16);
  for(unsigned i = 0;i < num.Size1();i++) {
    for(unsigned j = 0;j < num.Size2();j++) {
      num[i][j] = i * 17 + j;
    }
  }

  anArray.Set(num,cstrm);

  RavlN::SArray2dC<float> num2(16,16);
  num2.Fill(0);
  anArray.Get(num2,cstrm);

  cstrm.Wait();

  for(unsigned i = 0;i < num.Size1();i++) {
    for(unsigned j = 0;j < num.Size2();j++) {
      //RavlDebug("%u %u : %f %f ",i,j,num[i][j],num2[i][j]);
      RAVL_TEST_ALMOST_EQUALS(num[i][j],num2[i][j],1e-12);
    }
  }

  return 0;
}
