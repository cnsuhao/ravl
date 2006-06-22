//! author="Rachel Gartshore"

// MeasurementModel.hh
/*
 * This is the Measurement Model for the mobile robot Holly
 * 
 */


#ifndef MEASUREMENT_MODEL__HH
#define MEASUREMENT_MODEL__HH

#include "Ravl/PatternRec/Function.hh"

using namespace RavlN;

  class MeasurementModelBodyC
	  :public FunctionBodyC
  {
	  public:
		  MeasurementModelBodyC()
		  { outputSize=3; inputSize=3; cerr << "Default in MeasurementModelBodyC constructor\n"; }
		  //: Default Constructor

		  MeasurementModelBodyC( UIntT insize, UIntT outsize )
			  :FunctionBodyC( *new MeasurementModelBodyC(insize,outsize) )
		  {cerr << "insize,outsize constructor\n";}
		  //: Constructor to set input and output size vectors
		  // input vector is uK
		  // output vector is xk and xkplus1

		  virtual VectorC Apply( const VectorC &state ) const;
		  //: Apply Motion Model function to state with control_input 
		  // Predict the Next State
    
		  virtual MatrixC Jacobian( const VectorC &X ) const;
		  //: Calculate Jacobian matrix at X

	  private:
		  // For our motion model:
		  // x(k+1) = x(k) + u(k) {+ v(k)}
		  // We need these to calculate the jacobian
  };


  class MeasurementModelC
	  :public FunctionC
  {
	  public:
		  // Creates an invalid handle
		  MeasurementModelC()
			  :FunctionC( *new MeasurementModelBodyC() )
		  {cerr <<"default constructor in Measurementmodelc\n";}
		  //: Default Constructor

	  protected:
		  inline MeasurementModelC( MeasurementModelBodyC &bod ) : FunctionC(bod)
		  {}
		  //: Body constructor

		  inline MeasurementModelC( MeasurementModelBodyC *bod ) : FunctionC(bod)
		  {}
		  //: Body ptr constructor
		  inline MeasurementModelBodyC &Body()
		  { return static_cast<MeasurementModelBodyC &>(FunctionC::Body()); }
		  //: Access body

		  inline const MeasurementModelBodyC& Body() const
		  { return static_cast<const MeasurementModelBodyC &>(FunctionC::Body()); }
		  //: Access body (for constant handle)
		  
	  public:

		  inline VectorC Apply( const VectorC &state ) const
		  { return Body().Apply(state); }
		  //: Apply Motion Model function to state with control_input 
		  // Predict the Next State
    
		  inline MatrixC Jacobian( const VectorC &X ) const
		  { return Body().Jacobian(X); }
		  //: Calculate Jacobian matrix at X

  };

#if 0
  inline istream &operator>>(istream &strm,MeasurementModelC &obj)
  {
    obj = MeasurementModelC(strm);
    return strm;
  }
  //: Load from a stream.
  // Uses virtual constructor.
  
  inline ostream &operator<<(ostream &out,const MeasurementModelC &obj)
  {
    obj.Save(out);
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.
#endif
  
#endif
