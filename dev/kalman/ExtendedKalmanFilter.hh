//! author="Rachel Gartshore"

// ExtendedKalmanFilter.hh
//
// Extended Kalman Filter Class
//

#ifndef EXTENDED_KALMAN_FILTER_HH
#define EXTENDED_KALMAN_FILTER_HH

#define debugEKF 1

#include "Ravl/RefCounter.hh"
#include "Ravl/PatternRec/Function.hh"

#include <iostream>
#include <fstream>

using namespace RavlN;


  class ExtendedKalmanFilterBodyC
	  : public RCBodyC
  {
	  public:
		  ExtendedKalmanFilterBodyC()
		  {}
		  //: Default Constructor
		  
		  ExtendedKalmanFilterBodyC( UIntT size )
		  {}
		  //: Constructor for new object

		  virtual ~ExtendedKalmanFilterBodyC()
		  {}
		  //: Virtual Destructor

		  virtual bool SystemModel( FunctionC &s_model );
		  //: Attempt to set the system model

		  virtual bool MeasurementModel( FunctionC &m_model );
		  //: Attempt to set the measurement model

		  virtual VectorMatrixC TimeUpdate( const VectorMatrixC &Xk,
				  const VectorC &Uk, const MatrixC &Qk  ) const;
		  //: Evaluate the EKF for the Time-Update stage
		  // Given the system model f with the current state and covariance Xk and error Qk,
		  // calculate the Predicted State and Covariance
		  // Also known as the 'Predict Stage'

		  virtual VectorMatrixC MeasurementUpdate( const VectorMatrixC &Xbar,
				  const VectorC &Zk, const MatrixC &R ) const;
		  //: Evaluate EKF for the Measurement-Update stage
		  // Given the measurement model h, the new measurement Zk,
		  // the error Qk and the Predicted State & Covariance Xbar (calculated by TimeUpdate)
		  // calculate the Updated State and Covariance
		  // Also known as the 'Update Stage'
		  
		  virtual VectorMatrixC Update( const VectorMatrixC &Xk, const VectorC &Uk, const MatrixC &Qk,
				  const VectorC &Zk, const MatrixC &R ) const;
		  //: Evaluate the EKF for the Predict and Update Stage
		  // Given a system model, with measurement model h. For the previous state Xk with error Qk,
		  // along with the measurement Zk with error R, calculate the updated state


	  protected:
		  FunctionC	f;	// Non-linear State Space Model
		  FunctionC	h;	// Non-linear Measurement Model
  };


  class ExtendedKalmanFilterC
	  :public RCHandleC<ExtendedKalmanFilterBodyC>
  {
	  public:
		  ExtendedKalmanFilterC()
		  {}
		  //: Default Constructor
		  // Creates an invalid handle

		  ExtendedKalmanFilterC( UIntT size )
			  :RCHandleC<ExtendedKalmanFilterBodyC>( *new ExtendedKalmanFilterBodyC(size) )
		  {}
		  //: Create a function with the given number of inputs and outputs.

	  protected:
		  ExtendedKalmanFilterC( ExtendedKalmanFilterBodyC &bod )
			  : RCHandleC<ExtendedKalmanFilterBodyC>(bod)
		  {}
		  //: Body ptr constructor

		  inline ExtendedKalmanFilterBodyC & Body()
		  { return static_cast<ExtendedKalmanFilterBodyC &>(RCHandleC<ExtendedKalmanFilterBodyC>::Body()); }
		  //: Access body

		  inline const ExtendedKalmanFilterBodyC & Body() const
		  { return static_cast<const ExtendedKalmanFilterBodyC &>(RCHandleC<ExtendedKalmanFilterBodyC>::Body()); }
		  //: Access body (for constant handle

	  public:

		  bool SystemModel( FunctionC &s_model )
		  { return Body().SystemModel( s_model ); }
		  //: Attempt to set the System Model

		  bool MeasurementModel( FunctionC &m_model )
		  { return Body().MeasurementModel( m_model ); }
		  //: Attempt to set the Measurement Model


		  VectorMatrixC TimeUpdate( const VectorMatrixC &Xk,
				  const VectorC &Uk, const MatrixC &Qk  ) const
		  { return Body().TimeUpdate( Xk, Uk, Qk ); }
		  //: Evaluate the EKF for the Time-Update stage
		  // Given the system model f with the current state and covariance Xk and error Qk,
		  // calculate the Predicted State and Covariance
		  // Also known as the 'Predict Stage'

		  VectorMatrixC MeasurementUpdate( const VectorMatrixC &Xbar,
				  const VectorC &Zk, const MatrixC &R ) const
		  { return Body().MeasurementUpdate( Xbar, Zk, R ); }
		  //: Evaluate EKF for the Measurement-Update stage
		  // Given the measurement model h, the new measurement Zk,
		  // the error Qk and the Predicted State & Covariance Xbar (calculated by TimeUpdate)
		  // calculate the Updated State and Covariance
		  // Also known as the 'Update Stage'
		  
		  VectorMatrixC Update( const VectorMatrixC &Xk, const VectorC &Uk, const MatrixC &Qk,
				  const VectorC &Zk, const MatrixC &R ) const
		  { return Body().Update( Xk, Uk, Qk, Zk, R ); }
		  //: Evaluate the EKF for the Predict and Update Stage
		  // Given a system model, with measurement model h. For the previous state Xk with error Qk,
		  // along with the measurement Zk with error R, calculate the updated state
  };


#endif
