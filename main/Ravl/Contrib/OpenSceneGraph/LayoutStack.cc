
#include "Ravl/OpenSceneGraph/LayoutStack.hh"
#include "Ravl/OpenSceneGraph/TypeConvert.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/RLog.hh"

namespace RavlOSGN {

  //: Constructor
  
  LayoutStackC::LayoutStackC(bool create)
   : LayoutC(create),
     m_stackAxis(0)
  {
  }
  
  //: XML factory constructor
  
  LayoutStackC::LayoutStackC(const XMLFactoryContextC &factory)
   : LayoutC(true)
  {
    Setup(factory);
  }
  
  //: Update the layout
  
  bool LayoutStackC::UpdateLayout()
  {
    rDebug("Updating layout. ");
    Vector3dC start(0,0,0);
    for(unsigned i = 0;i < m_nodes.size();i++) {
      float size= m_nodes[i]->Bounds()._max[m_stackAxis] - m_nodes[i]->Bounds()._min[m_stackAxis];
      if(m_invertAxis) {
        start[m_stackAxis] -= (size + m_gap);
      }
      Vector3dC correctedStart = start - MakeVector3d(m_nodes[i]->Bounds()._min);
      rDebug("Start %s ",RavlN::StringOf(correctedStart).data());
      m_nodes[i]->SetPosition(correctedStart);
      if(!m_invertAxis) {
        start[m_stackAxis] += (size + m_gap);
      }
    }
    
    return true;
  }
  
  //: Process a callback.
  
  void LayoutStackC::DoCallback()
  {
    LayoutC::DoCallback();
  }
  
  //: Do setup from factory
  
  bool LayoutStackC::Setup(const XMLFactoryContextC &factory)
  {
    m_stackAxis = factory.AttributeInt("stackAxis",1);
    m_invertAxis = m_stackAxis < 0;
    m_stackAxis = RavlN::Abs(m_stackAxis) - 1;
    if(m_stackAxis < 0 || m_stackAxis > 2) {
      rError("Invalid stack axis %d ",m_stackAxis);
      throw RavlN::ExceptionBadConfigC("Invalid stack axis. ");
    }
    m_gap = factory.AttributeReal("gap",0.0);
    LayoutC::Setup(factory);
    return true;
  }
  
  //: Called when owner handles drop to zero.
  
  void LayoutStackC::ZeroOwners()
  {
    LayoutC::ZeroOwners();
  }
  
  void LinkLayoutStack()
  {}
  
  static RavlN::XMLFactoryRegisterConvertC<LayoutStackC,LayoutC> g_registerXMLFactoryLayout("RavlOSGN::LayoutStackC");

}
