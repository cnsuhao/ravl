//! author="Rachel Gartshore"

// MotionModel.hh
/*
 * This is the Motion Model for the mobile robot Holly
 * 
 * Motion Model
 * Using odometry measurements to tell us how the robot has moved
 * Will be given the new odometry reading (which includes the camera pan)
 */


#ifndef MOTION_MODEL__HH
#define MOTION_MODEL__HH

#include "Ravl/PatternRec/Function.hh"

using namespace RavlN;

  class MotionModelBodyC
	  :public FunctionBodyC
  {
	  public:
		  MotionModelBodyC()
		  { outputSize=3; inputSize=3; cerr << "Default MotionModelBodyC constructor\n"; }
		  //: Default Constructor

		  MotionModelBodyC( UIntT insize, UIntT outsize )
			  :FunctionBodyC( *new MotionModelBodyC(insize,outsize) )
		  {cerr << "insize,outsize constructor\n";}
		  //: Constructor to set input and output size vectors
		  // input vector is uK
		  // output vector is xk and xkplus1

		  virtual VectorC Apply( const VectorC &state, const VectorC &control_input ) const;
		  //: Apply Motion Model function to state with control_input 
		  // Predict the Next State
    
		  virtual MatrixC Jacobian( const VectorC &X ) const;
		  //: Calculate Jacobian matrix at X

	  private:
		  // For our motion model:
		  // x(k+1) = x(k) + u(k) {+ v(k)}
		  // We need these to calculate the jacobian
  };


  class MotionModelC
	  :public FunctionC
  {
	  public:
		  // Creates an invalid handle
		  MotionModelC()
			  :FunctionC( *new MotionModelBodyC() )
		  {cerr <<"default constructor in motionmodelc\n";}
		  //: Default Constructor

	  protected:
		  inline MotionModelC( MotionModelBodyC &bod ) : FunctionC(bod)
		  {}
		  //: Body constructor

		  inline MotionModelC( MotionModelBodyC *bod ) : FunctionC(bod)
		  {}
		  //: Body ptr constructor
		  inline MotionModelBodyC &Body()
		  { return static_cast<MotionModelBodyC &>(FunctionC::Body()); }
		  //: Access body

		  inline const MotionModelBodyC& Body() const
		  { return static_cast<const MotionModelBodyC &>(FunctionC::Body()); }
		  //: Access body (for constant handle)
		  
	  public:

		  inline VectorC Apply( const VectorC &state, const VectorC &control_input ) const
		  { return Body().Apply(state,control_input); }
		  //: Apply Motion Model function to state with control_input 
		  // Predict the Next State
    
		  inline MatrixC Jacobian( const VectorC &X ) const
		  { return Body().Jacobian(X); }
		  //: Calculate Jacobian matrix at X

  };

#if 0
  inline istream &operator>>(istream &strm,MotionModelC &obj)
  {
    obj = MotionModelC(strm);
    return strm;
  }
  //: Load from a stream.
  // Uses virtual constructor.
  
  inline ostream &operator<<(ostream &out,const MotionModelC &obj)
  {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.
#endif
  
#endif
