//! author="Rachel Gartshore"

// KalmanFilter.cc

#include "KalmanFilter.hh"

  //: Set the System Model
  bool KalmanFilterBodyC::SystemModel( MatrixC &a, MatrixC &b )
  {
    A = a;
    B = b;
    return true;
  }

  //: Set the Measurement Model
  bool KalmanFilterBodyC::MeasurementModel( MatrixC &h )
  {
    H = h;
    return true;
  }

  // Evaluate the EKF for the Time Update (predict) stage
  VectorMatrixC  KalmanFilterBodyC::TimeUpdate( const VectorMatrixC &Xk, const MatrixC &Qk ) const
  {
    // State Prediction - FIXME - Need to add optional B*uk
    VectorC Xkplus1 = A * Xk.Vector();
    // xbar(k+1|k) = f[k,xbar(k|k),u(k)]

    // State Prediction Covariance
    // P(k+1|k)=F(k)P(k|k)F(k)' + Q(k)
    MatrixC Pkplus1 = A * Xk.Matrix() * A.T() + Qk;

    return VectorMatrixC( Xkplus1, Pkplus1 );
  }

  // Evaluate the EKF for the Measurement Update (correct) stage
  VectorMatrixC KalmanFilterBodyC::MeasurementUpdate( const VectorC &Zk,
		  const VectorMatrixC &Xbar, const MatrixC &R ) const 
  {
    // Measurement Prediction Covariance
    // S(k+1) = H(k+1) * P(k+1|k) * H(k+1)' + R(k+1)
    MatrixC Sk = H * Xbar.Matrix() * H.T() + R;

    // Filter Gain
    // W(k+1) = P(k+1|k) * H(k+1)' * S(k+1)'
    MatrixC Wk = Xbar.Matrix() * H.T() * Sk.Inverse();

    // Updated State Covariance
    // P(k+1|k+1) = P(k+1|k) - W(k+1)*S(k+1)*W(k+1)'
    MatrixC Pk = Xbar.Matrix() - Wk * Sk * Xbar.Matrix().Inverse();

    // Measurement Prediction
    // zbar(k+1|k) = h[k+1,xbar(k+1|k)]
    VectorC zbar = H * Xbar.Vector();

    // Measurement Residual
    // v(k+1) = z(k+1) - zbar(k+1|k)
    VectorC Vk = Zk - zbar;

    // Updated State Estimate
    // xbar(k+1|k+1) = xbar(k+1|k) + W(k+1)*v(k+1)
    VectorC Xk = Xbar.Vector() + Wk * Vk;
    
    return VectorMatrixC( Xk, Pk );
  }

