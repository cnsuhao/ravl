//! author="Rachel Gartshore"

// KalmanFilter.hh
//
// Base Kalman Filter Class
//

#ifndef KALMAN_FILTER_HH
#define KALMAN_FILTER_HH

#include "Ravl/RefCounter.hh"
#include "Ravl/PatternRec/Function.hh"

#include <iostream>
#include <fstream>

using namespace RavlN;

  class KalmanFilterBodyC
	  : public RCBodyC
  {
	  public:
		  KalmanFilterBodyC()
		  {}
		  //: Default Constructor
		  
		  KalmanFilterBodyC( UIntT size )
		  {}
		  //: Create a function with the given number of inputs and outputs.

		  virtual ~KalmanFilterBodyC()
		  {}
		  //: Virtual Destructor

		  virtual VectorMatrixC TimeUpdate( const VectorMatrixC &Xk, const MatrixC &Qk  ) const;
		  //: Evaluate the EKF for the Time-Update stage
		  // Given the system model f with the current state and covariance Xk and error Qk,
		  // calculate the Predicted State and Covariance
		  // Also known as the 'Predict Stage'

		  // R is the measurement noise covariance
		  virtual VectorMatrixC MeasurementUpdate( const VectorC &Zk,
				  const VectorMatrixC &Xbar, const MatrixC &R ) const;
		  //: Evaluate EKF for the Measurement-Update stage
		  // Given the measurement model h, the new measurement Zk,
		  // the error Qk and the Predicted State & Covariance Xbar (calculated by TimeUpdate)
		  // calculate the Updated State and Covariance
		  // Also known as the 'Update Stage'
		  
		  // Qk is process noise covariance
		  virtual VectorMatrixC Update( const VectorMatrixC &Xk, const MatrixC &Qk,
				  const VectorC &Zk, const MatrixC &R ) const;
		  //: Evaluate the EKF for the Predict and Update Stage
		  // Given a system model, with measurement model h. For the previous state Xk with error Qk,
		  // along with the measurement Zk with error R, calculate the updated state

		  virtual bool SystemModel( MatrixC &a, MatrixC &b );
		  virtual bool MeasurementModel( MatrixC &h );

	  protected:
		  MatrixC A;	// State Transition Matrix
		  MatrixC B;	// Optional Control to State Relation
		  MatrixC H;	// State to Measurement Relation
  };


  class KalmanFilterC
	  :public RCHandleC<KalmanFilterBodyC>
  {
	  public:
		  KalmanFilterC()
		  {}
		  //: Default Constructor
		  // Creates an invalid handle

		  KalmanFilterC( UIntT size )
			  :RCHandleC<KalmanFilterBodyC>( *new KalmanFilterBodyC(size) )
		  {}
		  //: Create a function with the given number of inputs and outputs.

	  protected:
		  KalmanFilterC( KalmanFilterBodyC &bod )
			  : RCHandleC<KalmanFilterBodyC>(bod)
		  {}
		  //: Body ptr constructor

		  inline KalmanFilterBodyC & Body()
		  { return static_cast<KalmanFilterBodyC &>(RCHandleC<KalmanFilterBodyC>::Body()); }
		  //: Access body

		  inline const KalmanFilterBodyC & Body() const
		  { return static_cast<const KalmanFilterBodyC &>(RCHandleC<KalmanFilterBodyC>::Body()); }
		  //: Access body (for constant handle

	  public:

		  bool SystemModel(  MatrixC &a, MatrixC &b )
		  { return Body().SystemModel( a, b ); }
		  //: Attempt to set the System Model

		  bool MeasurementModel( MatrixC &h )
		  { return Body().MeasurementModel( h ); }
		  //: Attempt to set the Measurement Model


		  VectorMatrixC TimeUpdate( const VectorMatrixC &Xk, const MatrixC &Qk  ) const
		  { return Body().TimeUpdate( Xk, Qk ); }
		  //: Evaluate the EKF for the Time-Update stage
		  // Given the system model f with the current state and covariance Xk and error Qk,
		  // calculate the Predicted State and Covariance
		  // Also known as the 'Predict Stage'

		  VectorMatrixC MeasurementUpdate( const VectorC &Zk,
				  const VectorMatrixC &Xbar, const MatrixC &R ) const
		  { return Body().MeasurementUpdate( Zk, Xbar, R ); }
		  //: Evaluate EKF for the Measurement-Update stage
		  // Given the measurement model h, the new measurement Zk,
		  // the error Qk and the Predicted State & Covariance Xbar (calculated by TimeUpdate)
		  // calculate the Updated State and Covariance
		  // Also known as the 'Update Stage'
		  
		  VectorMatrixC Update( const VectorMatrixC &Xk, const MatrixC &Qk,
			const VectorC &Zk, const MatrixC &R ) const
		  { return Body().Update( Xk, Qk, Zk, R ); }
		  //: Evaluate the EKF for the Predict and Update Stage
		  // Given a system model, with measurement model h. For the previous state Xk with error Qk,
		  // along with the measurement Zk with error R, calculate the updated state
  };



#endif
