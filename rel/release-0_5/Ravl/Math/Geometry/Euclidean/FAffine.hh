// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
#ifndef RAVL_FAFFINE_HEADER
#define RAVL_FAFFINE_HEADER 1
//////////////////////////////////////////////////////
//! author="Charles Galambos"
//! date="17/3/1997"
//! docentry="Ravl.Math.Geometry"
//! rcsid="$Id$"
//! example=exFAffine.cc
//! lib=RavlMath

#include "Ravl/FMatrix.hh"
#include "Ravl/FVector.hh"

namespace RavlN {
  
  //! userlevel=Normal
  //: General affine transformation.
  
  template<unsigned int N>
  class FAffineC {
  public:
    inline FAffineC();
    //: Construct no-change transform.
    
    inline FAffineC(const FAffineC &Oth);
    //: Copy constructor.
    
    inline FAffineC(const FMatrixC<N,N> &SR, const FVectorC<N> &T);
    //: Construct from Scale/Rotate matrix and a translation vector.
    
    inline FVectorC<N> &Translation() { return T; }
    //: Access the translation component of the transformation.
    
    inline FVectorC<N> Translation() const { return T; }
    //: Constant access to the translation component of the transformation.
    
    inline void Scale(FVectorC<N> xy);
    //: In place Scaling along the X & Y axis by value given in the vector.
    // If all values 1, then no effect.
    
    inline void Translate(const FVectorC<N> &T);
    //: Add a translation in direction T.
    
    inline FVectorC<N> operator*(const FVectorC<N> &In) const;
    //: Transform Vector,  Scale, Rotate, Translate.
    // Take a vector and put it though the transformation.
    
    inline FAffineC<N> operator*(const FAffineC &In) const;
    //: Compose this transform with 'In'
    
    inline FAffineC<N> operator/(const FAffineC &In) const;
    //: 'In' / 'Out' = this; 
    
    FAffineC<N> I() const;
    //: Generate an inverse transformation.
    
    FMatrixC<N,N> &SRMatrix() { return SR; }
    //: Get Scale/Rotate matrix.
    
    const FMatrixC<N,N> &SRMatrix() const { return SR; }
    //: Get Scale/Rotate matrix.
    
    inline const FAffineC<N> &operator=(const FAffineC &Oth);
    //: Assigmment.
    
  protected:
    FMatrixC<N,N> SR; // Scale/rotate.
    FVectorC<N> T;   // Translate.
    
  private:
#ifdef __sgi
    friend istream & operator>> (istream & inS, FAffineC<N> & vector);
    friend ostream & operator<< (ostream & outS, const FAffineC<N> & vector);
#else   
    friend istream & operator>> <N> (istream & inS, FAffineC<N> & vector);
    friend ostream & operator<< <N> (ostream & outS, const FAffineC<N> & vector);
#endif
    
  };
  
  template<unsigned int N>
  ostream & operator<< (ostream & outS, const FAffineC<N> & vector);
  template<unsigned int N>
  istream & operator>> (istream & inS, FAffineC<N> & vector);
  
  /////////////////////////////////////////////////
  
  template<unsigned int N>
  inline FAffineC<N>::FAffineC()
    : SR()
  {
    T.Fill(0);
    SR.Fill(0.0);
    for(unsigned int i = 0;i < N;i++)
      SR[i][i] = 1.0;
  }
  
  template<unsigned int N>
  inline FAffineC<N>::FAffineC(const FAffineC &Oth) 
    : SR(Oth.SR),
      T(Oth.T)
  {}
  
  template<unsigned int N>
  inline FAffineC<N>::FAffineC(const FMatrixC<N,N> &nSR, const FVectorC<N> &nT) 
    : SR(nSR),
      T(nT)
  {}
  
  template<unsigned int N>
  inline void FAffineC<N>::Scale(FVectorC<N> xy) {
    for(UIntT i = 0;i < N;i++)
      for(UIntT j = 0;j < N;i++)
	SR[i][j] *= xy[j];
  }
  
  template<unsigned int N>
  inline void FAffineC<N>::Translate(const FVectorC<N> &DT) {
    T += DT;
  }
  
  template<unsigned int N>
  inline FAffineC<N> FAffineC<N>::I(void) const {
    FMatrixC<N,N> iSR = SR.I();
    return FAffineC(iSR,iSR * T * -1);
  }
  
  template<unsigned int N>
  inline FVectorC<N> FAffineC<N>::operator*(const FVectorC<N> &In) const {
    return (SR * In) + T;
  }
  
  template<unsigned int N>
  inline FAffineC<N> FAffineC<N>::operator*(const FAffineC &In) const{
    return FAffineC(In.SRMatrix()*SR, In.SRMatrix()*T + In.Translation());
  }
  
  template<unsigned int N>
  inline FAffineC<N> FAffineC<N>::operator/(const FAffineC &In) const{
    return FAffineC(SR*In.SRMatrix().I(), In.SRMatrix().I()*(T-In.Translation()));
  }
  
  template<unsigned int N>
  inline const FAffineC<N> &FAffineC<N>::operator=(const FAffineC &Oth) {
    SR = Oth.SR;
    T = Oth.T;
    return *this;
  }
  
  
  template<unsigned int N>
  ostream &
  operator<<(ostream & outS, const FAffineC<N> & vector) {
    outS << vector.SRMatrix() << "\t" << vector.Translation();
    return outS;
  }
  
  template<unsigned int N>
  istream & 
  operator>>(istream & inS, FAffineC<N> & vector) {
    inS >> vector.SRMatrix() >> "\t" >> vector.Translation();
    return inS;
  }
   
}
#endif
  