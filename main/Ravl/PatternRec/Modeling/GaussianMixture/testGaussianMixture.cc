// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2002, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlPatternRec
//! file="Ravl/PatternRec/Modeling/GaussianMixture/testGaussianMixture.cc"
//! author="Kieron Messer"
//! docentry="Ravl.API.Pattern Recognition.Expectation Maximization"
//! userlevel=Develop

#include "Ravl/PatternRec/GaussianMixture.hh"
#include "Ravl/PatternRec/DesignGaussianMixture.hh"
#include "Ravl/PatternRec/SampleVector.hh"
#include "Ravl/OS/Filename.hh"

using namespace RavlN;

int testGaussianMixture();
int testDesignGaussianMixture();


int main() {
  int ln;
  if((ln = testGaussianMixture()) != 0) {
    cerr << "Test failed line " << ln << "\n";
    return 1;
  }
  cerr << "Test passed ok. \n";


  if((ln = testDesignGaussianMixture()) != 0) {
    cerr << "Test failed line " << ln << "\n";
    return 1;
  }
  cerr << "Test passed ok. \n";


  return 0;
}

int testGaussianMixture() {

  VectorC mean(2);
  mean[0] = 1;
  mean[1] = 2;
  MatrixRSC cov(2);
  cov[0][0] = 1;
  cov[0][1]=0;
  cov[1][0]=0;
  cov[1][1]=1;

  SArray1dC<MeanCovarianceC>arr(1);
  arr[0] = MeanCovarianceC(1, mean, cov);

  SArray1dC<RealT>weights(1);
  weights[0] = 1.0;

  GaussianMixtureC gm1(arr, weights, true);
  GaussianMixtureC gm2(arr, weights, false);
  GaussianMixtureC gm3;

  FilenameC tmp("/tmp/gm.strm");

  //: Lets output a model
  {
    OStreamC ofs(tmp);
    ofs << gm1;
  }
  
  //: Lets load in same model
  {
  IStreamC ifs(tmp);
  ifs >> gm3;
  }
  tmp.Remove();

  //: Lets test on data
  VectorC test(2);
  test[0] = 1.9;
  test[1] = 2.3;

#if 0
  cout << "gm1 - " << gm1.Apply(test) << endl;
  cout << "gm2 - " << gm2.Apply(test) << endl;
  cout << "gm3 - " << gm3.Apply(test) << endl;
  cout  << "MeanCov - " << arr[0].Gauss(test) << endl << flush;
#endif
  
  RealT dif = 0.001;
  if((gm1.Apply(test)[0] - gm2.Apply(test)[0])>dif) return 1;
  if((gm1.Apply(test)[0] - gm3.Apply(test)[0])>dif) return 1;
  if((gm2.Apply(test)[0] - gm3.Apply(test)[0])>dif) return 1;

  return 0;
}

int testDesignGaussianMixture() 
{
  //: Set a random seed
  //: Different seeds we do a different result
  RandomSeedDefault(123421211);

  //: Load a dataset
  StringC datafile = StringC(PROJECT_OUT) + "/share/RAVL/data/class5.strm";
  IStreamC ifs(datafile);
  SampleVectorC dset;
  ifs >> dset;

  //: Lets do EM
  DesignGaussianMixtureC em(5, true);
  GaussianMixtureC gm = em.Apply(dset);
  //: Lets do a quick test
  VectorC X(2);
  X[0] = 0.0;
  X[1] = 0.0;
  RealT value = gm.DensityValue(X);

#if 0
  cout << gm << endl;
#endif

  //: This should be about 0.03
  if((value<0.025) || (value>0.035)) return 1;

  //: Everything looks OKish
  return 0;
}
