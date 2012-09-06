// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html


%include "Ravl/Swig2/Vector.i"
%include "Ravl/Swig2/Matrix.i"
    
%{
#ifdef SWIGPERL
#undef Copy
#endif

#include "Ravl/MeanCovariance.hh"

#ifdef SWIGPERL
#define Copy(s,d,n,t)   (MEM_WRAP_CHECK_(n,t) (void)memcpy((char*)(d),(const char*)(s), (n) * sizeof(t)))
#endif
%}

namespace RavlN {

  
  class MeanCovarianceC  {
  public:
   MeanCovarianceC();
    // Empty constructor, creates invalid object

    MeanCovarianceC(const MeanCovarianceC & meanCov);
    // The class MeanCovarianceC is implemented as a big object using
    // a reference counter.
    
    MeanCovarianceC(const SizeT n);
    // Creates zero mean and zero covariance matrix representing
    // the 'n'-dimensional set containing no data points.

    MeanCovarianceC(const unsigned n);
    // Creates zero mean and zero covariance matrix representing
    // the 'n'-dimensional set containing no data points.

    MeanCovarianceC(const VectorC & point);
    // Creates the mean vector and zero covariance matrix representing
    // the data set containing just one data point. The vector 'point'
    // is shared.
    
    //MeanCovarianceC(const MeanNdC & mean);
    // Creates the mean vector and zero covariance matrix representing
    // the data set represented by the 'mean'. The structure 'mean'
    // is shared.
    
    MeanCovarianceC(RealT n, 
		    const VectorC & mean, 
		    const MatrixC & ncov);
    // Creates the mean vector and zero covariance matrix representing
    // the data set containing 'n' points and represented by the 'mean'
    // and the covariance matrix 'cov'. Both 'mean' and 'cov' are
    // shared.
    
    MeanCovarianceC Copy() const;
    // Returns a new physical copy of this object.
    
    MeanCovarianceC(const SArray1dC<VectorC> & data, bool sampleStatistics = true);
    //: Compute the mean and covariance of an array of vectors.
    //!param: data - Array containing data to compute statistics on
    //!param: sampleStatistics - When true compute statistics as a sample of a random variable. (Normalise covariance by n-1 )
    
    //MeanCovarianceC(const DListC<VectorC> & data,bool sampleStatistics = true);
    //: Compute the mean and covariance of a list of vectors.
    //!param: data - List containing data to compute statistics on
    //!param: sampleStatistics - When true compute statistics as a sample of a random variable. (Normalise covariance by n-1 )
    
    // Information about an object
    // ---------------------------

    RealT Number() const;
    // Returns the number of data points which are represented by this object.
    
    const VectorC & Mean() const;
    //: Access the mean.
    // Returns the mean vector of data points which are represented
    // by this object.

    VectorC & Mean();
    //: Access the mean.
    // Returns the mean vector of data points which are represented
    // by this object.
    
    //const MeanNdC & MeanNd() const;
    //: Access the mean nd object.
    // Returns the mean vector of data points which are represented
    // by this object.
    
    const MatrixC & Covariance() const;
    //: Access the covariance.
    // Returns the covariance matrix of data points which are represented
    // by this object.

    MatrixC & Covariance();
    //: Access the covariance.
    // Returns the covariance matrix of data points which are represented
    // by this object.

    // Object modification
    // -------------------      
    
    const MeanCovarianceC & SetZero();
    // Total initialization of this object resulting in the representation
    // the empty set of data points.

    const MeanCovarianceC & operator+=(const VectorC & point);
    //: Adds one point to the set of data points.
    // Note, this is NOT a good way to compute the mean and covariance 
    // of a large dataset. Use one of the constructors from a list
    // or array of vectors.
    
    const MeanCovarianceC & operator-=(const VectorC & point);
    // Removes one point from the set of data points. Be carefull to remove
    // a point which was already added to the set, otherwise the representation
    // will not describe a real set.

    //const MeanCovarianceC & operator+=(const MeanNdC & mean);
    // Adds a number of data poits represented by the 'mean' and zero
    // covariance matrix to this set.

    //const MeanCovarianceC & operator-=(const MeanNdC & mean);
    // Removes a number of data poits represented by the 'mean' and zero
    // covariance matrix from this set. Be carefull to remove
    // points which were already added to the set, otherwise the representation
    // will not describe a real set.

    const MeanCovarianceC & operator+=(const MeanCovarianceC & meanCov);
    // Adds a number of data points represented by the 'meanCov' structure
    // to this set.

    const MeanCovarianceC & operator-=(const MeanCovarianceC & meanCov);
    // Removes a number of data points represented by the 'meanCov' structure
    // from this set. Be carefull to remove
    // points which were already added to the set, otherwise the representation
    // will not describe a real set.

    const MeanCovarianceC & Add(const VectorC & point, const VectorC & var);
    // Updates the mean and the covariance matrix by adding one N-d point
    // whose coordinates are known with the error described by the diagonal
    // convariance matrix represented byt the vector 'var'.

    const MeanCovarianceC &Remove(const VectorC & point, const VectorC & var);
    // Updates the mean and the covariance matrix by removing one N-d point
    // whose coordinates are known with the error described by the diagonal
    // convariance matrix represented byt the vector 'var'.
    // Be carefull to remove the point which was already added
    // to the set, otherwise the representation
    // will not describe a real set.
    
    const MeanCovarianceC & SetSum(const MeanCovarianceC & meanCov1,
				   const MeanCovarianceC & meanCov2);
    //: This object is set to be the union of two set of data points 'meanCov1'
    //: and 'meanCov2'.
    
    MeanCovarianceC operator*(const MeanCovarianceC &oth) const;
    //: Calculate the product of the two probability density functions.
    // This assumes the estimates of the distributions are accurate. (The number
    // of samples is ignored) 
    
    RealT Gauss(const VectorC &vec) const;
    //: Evaluate the value of guassian distribution at 'vec'.
    
    RealT MahalanobisDistance(const VectorC &vec) const;
    //: Compute the Mahalanobis to the point.
    
    void ClearCache();
    //: Clear inverse cache.
    // This must be used if you modify the mean or covariance directly.

    UIntT Hash() const;
    //: Provided for compatibility with templates.
    
    %extend {
     // Allow python to print it
      const char *__str__()
      {
        RavlN::StrOStreamC os;
        os << *self;
        return PyString_AsString(PyString_FromStringAndSize(os.String().chars(), os.String().Size()));
      }	
    }
    
   
  };
  
 
  
}
