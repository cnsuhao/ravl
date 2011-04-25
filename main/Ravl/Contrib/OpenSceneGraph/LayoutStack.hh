/* 
 * File:   LayoutStack.hh
 * Author: charlesgalambos
 *
 * Created on April 25, 2011, 10:13 AM
 */

#ifndef RAVLOSGN_LAYOUTSTACK_HH
#define	RAVLOSGN_LAYOUTSTACK_HH

#include "Ravl/OpenSceneGraph/Layout.hh"

namespace RavlOSGN {

  //: Base class for automatic layouts
  
  class LayoutStackC
   : public LayoutC
  {
  public:
    LayoutStackC(bool create);
    //: Constructor
    
    LayoutStackC(const XMLFactoryContextC &factory);
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
    
    int m_stackAxis;
    float m_gap;
  };

}

#endif	/* LAYOUTSTACK_HH */

