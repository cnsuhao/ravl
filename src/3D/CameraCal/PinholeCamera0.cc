#include "Ravl/3D/PinholeCamera0.hh"

namespace Ravl3DN
{
  using namespace RavlN;
  using namespace RavlImageN;

  //: Load from stream.
  bool
  PinholeCameraBody0C::Load(istream &s)
  {
    s >> m_frame;
    s >> m_fx >> m_fy >> m_cx >> m_cy;
    s >> m_R;
    s >> m_t;
    return true;
  }
    
  //: Load from binary stream.
  bool
  PinholeCameraBody0C::Load(BinIStreamC &s)
  {
    s >> m_frame;
    s >> m_fx >> m_fy >> m_cx >> m_cy;
    s >> m_R;
    s >> m_t;
    return true;
  }
    
  //: Writes object to stream, can be loaded using constructor
  bool 
  PinholeCameraBody0C::Save(ostream &s) const
  {
    s << m_frame
      << endl << m_fx 
      << " " << m_fy
      << " " << m_cx
      << " " << m_cy
      << endl;
    s << m_R;
    s << m_t << endl;
    return true;
  }
    
  //: Writes object to stream, can be loaded using constructor
  bool 
  PinholeCameraBody0C::Save(BinOStreamC &s) const
  {
    s << m_frame
      << m_fx
      << m_fy
      << m_cx
      << m_cy
      << m_R
      << m_t;
    return true;
  }

};
