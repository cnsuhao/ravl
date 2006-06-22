//! author="Rachel Gartshore"

// MotionModel.cc

#include "MeasurementModel.hh"

  //: Apply Motion Model function to 'data' == predict the next state
  VectorC MeasurementModelBodyC::Apply( const VectorC &data ) const
  {
    // z(k+1) = h[k+1,x(k+1)]+w(k+1)

    // the state vector x(k+1) is the same as the measurement vector z(k+1)
    return data;
  }
  
  //: Calculate Jacobian matrix of model at X
  //
  // F(k) = df(k)|
  //        -----|
  //         d(x)|
  //             |x=X
  MatrixC MeasurementModelBodyC::Jacobian(const VectorC &X) const
  {
    // Our model is:
    // z(k+1) = x(k+1) + w(k+1)
    /*
    RealT d = X[0];
    RealT cosine = cos(X[1]);
    RealT sine = sin(X[1]);
    MatrixC ret(3,2);
    ret[0][0] = cosine; ret[0][1] = d*sine;
    ret[1][0] = sine; ret[1][1] = d*cosine;
    ret[2][0] = 0.0; ret[2][1] = 1.0;
    */
    return MatrixC( 1, 0, 0, 0, 1, 0, 0, 0, 1 );

    //for(UIntT i = 0;i < X.Size();i++)
      //ret.SetColumn(i,a*MakeJacobianInput(X,i));
    //return ret;
  }
  
