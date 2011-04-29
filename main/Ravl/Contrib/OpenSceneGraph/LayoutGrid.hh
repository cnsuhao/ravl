/* 
 * File:   LayoutGrid.hh
 * Author: charlesgalambos
 *
 * Created on April 25, 2011, 10:13 AM
 */

#ifndef RAVLOSGN_LayoutGrid_HH
#define	RAVLOSGN_LayoutGrid_HH

#include "Ravl/OpenSceneGraph/Layout.hh"

namespace RavlOSGN {

  //: Base class for automatic layouts
  
  class LayoutGridC
   : public LayoutC
  {
  public:
    LayoutGridC(bool create);
    //: Constructor
    
    LayoutGridC(const XMLFactoryContextC &factory);
    //: XML factory constructor
    
  protected:
    virtual bool UpdateLayout();
    //: Update the layout
    
    virtual void DoCallback();
    //: Process a callback.
    
    bool Setup(const XMLFactoryContextC &factory);
    //: Do setup from factory
    
    virtual void ZeroOwners();
    //: Called when owner handles drop to zero.
    
    int m_stackAxis1;
    bool m_invertAxis1;
    int m_stackAxis2;
    bool m_invertAxis2;
    float m_gap;
    float m_minSize;
  };

}

#endif	/* LayoutGrid_HH */

