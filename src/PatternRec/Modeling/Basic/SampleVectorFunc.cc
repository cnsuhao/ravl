/*
 * SampleVectorFunc.cc
 *
 *  Created on: 22 May 2012
 *      Author: charles
 */

#include "Ravl/PatternRec/SampleVector.hh"
#include "Ravl/DArray1dIter.hh"
#include "Ravl/DArray1dIter2.hh"
#include "Ravl/DArray1dIter3.hh"
#include "Ravl/MeanCovariance.hh"
#include "Ravl/MatrixRUT.hh"
#include "Ravl/PatternRec/DataSet2Iter.hh"
#include "Ravl/PatternRec/SampleIter.hh"
#include "Ravl/SArray1dIter2.hh"
#include "Ravl/SumsNd2.hh"
#include "Ravl/VectorMatrix.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/RandomGauss.hh"
#include "Ravl/TMatrix.hh"

namespace RavlN {
  /*
   * Compute the normalisation function using the current data, normalise the data IP and return function
   */
  FunctionC SampleVectorC::Normalise(DataSetNormaliseT normType) {

    FunctionC func;
    if(normType == DATASET_NORMALISE_NONE) {
      // do nothing
      RavlDebug("No normalisation chosen!");
    } else if(normType == DATASET_NORMALISE_MEAN) {
      MeanCovarianceC meanCov = MeanCovariance();
      func = NormalisationFunction(meanCov);
      RavlDebug("Normalise by mean and covariance of entire data");
      Normalise(func);
    } else if(normType == DATASET_NORMALISE_SCALE) {
      RavlDebug("Scaling dataset");
      FuncLinearC linFunc;
      Scale(linFunc);
      func = linFunc;
    } else {
      RavlError("Unknown normalisation function!");
    }
    return func;
  }

  FuncMeanProjectionC SampleVectorC::NormalisationFunction(const MeanCovarianceC & stats) const {

    UIntT d = VectorSize();
    VectorC stdDev(d);
    for (UIntT i = 0; i < d; i++) {
      if (stats.Covariance()[i][i] > 0)
        stdDev[i] = stats.Covariance()[i][i];
      else
        stdDev[i] = stats.Mean()[i];
    }

    for (UIntT i = 0; i < d; i++)
      stdDev[i] = Sqrt(stdDev[i]);
    stdDev.Reciprocal();

    FuncMeanProjectionC func;
    MatrixC proj(d, d);
    proj.Fill(0.0);
    proj.SetDiagonal(stdDev);
    func = FuncMeanProjectionC(stats.Mean(), proj);

    return func;

  }

  //: Undo the normalisation done by 'Normalise()'.

  FuncLinearC SampleVectorC::UndoNormalisationFunction(const MeanCovarianceC & stats) const {

    UIntT d = VectorSize();
    MatrixC stdDev(d, d);
    stdDev.Fill(0.0);
    for (UIntT i = 0; i < d; i++) {
      if (stats.Covariance()[i][i] > 0)
        stdDev[i][i] = stats.Covariance()[i][i];
      else
        stdDev[i][i] = stats.Mean()[i];
    }
    for (UIntT i = 0; i < d; i++)
      stdDev[i][i] = Sqrt(stdDev[i][i]);
    return FuncLinearC(stdDev, stats.Mean());

  }


  //: Scale each dimension between 0 and 1 and return function created to do this
  void SampleVectorC::Scale(FuncLinearC & func) {

    DArray1dIterC<VectorC> it(*this);
    if (!it)
      return;

    VectorC min = it.Data().Copy();
    VectorC max = it.Data().Copy();

    for (it++; it; it++) {
      for (SArray1dIter2C<RealT, RealT> minIt(min, *it); minIt; minIt++) {
        if (minIt.Data2() < minIt.Data1()) {
          minIt.Data1() = minIt.Data2();
        }
      }
      for (SArray1dIter2C<RealT, RealT> maxIt(max, *it); maxIt; maxIt++) {
        if (maxIt.Data2() > maxIt.Data1()) {
          maxIt.Data1() = maxIt.Data2();
        }
      }
    } // end data it


    // work out transform
    MatrixC proj(min.Size(), min.Size());
    proj.Fill(0.0);
    VectorC offset(min.Size());
    for(SizeT i=0;i<min.Size();i++) {
      proj[i][i] = 1.0 / Abs(max[i] - min[i]);
      offset[i] = (min[i]/Abs(max[i] - min[i])) * -1.0;
    }
    func = FuncLinearC(proj, offset);

    // Apply transform
    for(it.First();it;it++) {
      *it = func.Apply(*it);
    }
    return;

  }
}
