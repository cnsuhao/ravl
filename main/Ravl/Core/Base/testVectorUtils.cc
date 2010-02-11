// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlCore
//! file="Ravl/Core/Base/testTFVector.cc"
//! docentry="Ravl.API.Core.Tuples"
//! userlevel=Develop
//! author="Charles Galambos"
//! date="24/01/2001"

#include "Ravl/VectorUtils.hh"
#include "Ravl/Stream.hh"

using namespace RavlN;

template<class DataT>
int testDotProduct() {
  const size_t bufferLen = 10240;
  DataT buf1[bufferLen];
  DataT buf2[bufferLen];
  for(size_t i = 0; i < bufferLen; i++) {
    buf1[i] = i;
    buf2[i] = i*i / 2.;
  }

  //calculate sums old slow way
  DataT sum00_00 = 0.;
  for(size_t i = 0; i < bufferLen; i++) {
    sum00_00 += buf1[i] * buf2[i];
  }

  DataT sum01_10 = 0.;
  for(size_t i = 0; i < bufferLen-1; i++) {
    sum01_10 += buf1[i+1] * buf2[i];
  }

  DataT sum10_01 = 0.;
  for(size_t i = 0; i < bufferLen-1; i++) {
    sum10_01 += buf1[i] * buf2[i+1];
  }

  DataT sum00_11 = 0.;
  for(size_t i = 0; i < bufferLen-1; i++) {
    sum00_11 += buf1[i] * buf2[i];
  }

  DataT sum11_00 = 0.;
  for(size_t i = 0; i < bufferLen-1; i++) {
    sum11_00 += buf1[i+1] * buf2[i+1];
  }
  //test fast way
  DataT fast00_00 = RavlBaseVectorN::DotProduct(buf1, buf2, bufferLen);
  if(RavlN::Abs(fast00_00 - sum00_00) > 1e-9) return __LINE__;

  DataT fast01_10 = RavlBaseVectorN::DotProduct(buf1+1, buf2, bufferLen-1);
  if(RavlN::Abs(fast01_10 - sum01_10) > 1e-9) return __LINE__;

  double fast10_01 = RavlBaseVectorN::DotProduct(buf1, buf2+1, bufferLen-1);
  if(RavlN::Abs(fast10_01 - sum10_01) > 1e-9) return __LINE__;

  double fast00_11 = RavlBaseVectorN::DotProduct(buf1, buf2, bufferLen-1);
  if(RavlN::Abs(fast00_11 - sum00_11) > 1e-9) return __LINE__;

  double fast11_00 = RavlBaseVectorN::DotProduct(buf1+1, buf2+1, bufferLen-1);
  if(RavlN::Abs(fast11_00 - sum11_00) > 1e-9) return __LINE__;

  return 0;
}

int testConvolveKernel() {
  for(size_t matrRows = 128-2*13; matrRows <= 200; matrRows += 13) {
    //cerr << "matrRows:" << matrRows << endl;
    for(size_t matrCols = 128-2*13; matrCols <= 200; matrCols += 13) {
      //cerr << "matrCols:" << matrCols << endl;
      for(size_t kernRows = 32-2*5; kernRows <= 64; kernRows += 5) {
        //cerr << "kernRows:" << kernRows << endl;
        for(size_t kernCols = 32-2*5; kernCols <= 64; kernCols += 5) {
          //cerr << "kernCols:" << kernCols << endl;
          float matrix[matrRows][matrCols];
          float kernel[kernRows][kernCols];

          //create kernel
          for(size_t r = 0; r < kernRows; r++)
            for(size_t c = 0; c < kernCols; c++) {
              kernel[r][c] = (r-c) * (r-c);
            }

          //create matrix
          for(size_t r = 0; r < matrRows; r++)
            for(size_t c = 0; c < matrCols; c++) {
              matrix[r][c] = (r + c) / 2.;
            }


          for(size_t posRow = 20-2; posRow <= 25; posRow += 1) {
            //cerr << "posRow:" << posRow << endl;
            for(size_t posCol = 10-2; posCol <= 15; posCol += 1) {
              //cerr << "posCol:" << posCol << endl;

              //compute old way
              float resOld = 0.f;
              for(size_t r = 0; r < kernRows; r++)
                for(size_t c = 0; c < kernCols; c++) {
                  resOld += matrix[r+posRow][c+posCol] * kernel[r][c];
                }

              float resNew = 111;
              RavlBaseVectorN::ConvolveKernel(&(matrix[posRow][posCol]), &(kernel[0][0])  , kernRows, kernCols, matrCols*sizeof(float), &resNew);

              if(RavlN::Abs(resNew - resOld) > 1e-9) return __LINE__;
            }
          }
        }
      }
    }
  }
  return 0;
}

int main(int nargs,char **argv) {
  int ln;
  if((ln = testDotProduct<float>()) != 0) {
    cerr << "Error 'float' line :" << ln << "\n";
    return 1;
  }
  if((ln = testDotProduct<double>()) != 0) {
    cerr << "Error 'double' line :" << ln << "\n";
    return 1;
  }
  if((ln = testConvolveKernel()) != 0) {
    cerr << "Error line :" << ln << "\n";
    return 1;
  }
  cerr <<"Test passed ok. \n";
  return 0;
}


