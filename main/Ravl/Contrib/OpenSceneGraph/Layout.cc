
#include "Ravl/OpenSceneGraph/Layout.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/RLog.hh"
#include <osg/ComputeBoundsVisitor>


namespace RavlOSGN {
  
  //: Constructor
  
  LayoutEntryC::LayoutEntryC(const NodeC &element)
   : m_element(&element)
  {
    AddChildNode(element);
    
    //SetPosition(Vector3dC(0,0,0));
    //SetScale(Vector3dC(1.0,1.0,1.0));
    
    Node()->setDataVariance(osg::Object::DYNAMIC);
    
    ComputeBounds();
    
  }
  
  bool LayoutEntryC::ComputeBounds()
  {
    osg::ComputeBoundsVisitor boundsVisitor; 
    m_element->Node()->accept(boundsVisitor);
    m_bounds = boundsVisitor.getBoundingBox(); 
    rDebug("Bounds %f %f %f : %f %f %f",
           m_bounds._min[0],m_bounds._min[1],m_bounds._min[2],
           m_bounds._max[0],m_bounds._max[1],m_bounds._max[2]
           );
    return true;
  }
  
  void LayoutEntryC::SetTargetPosition(const Vector3dC &position)
  {
    rDebug("Target position %s ",RavlN::StringOf(position).data());
    m_targetPosition = position;
    SetPosition(position);
  }

  //---------------------------------------------------------------

  //: Default constructor
  LayoutC::LayoutC(bool create)
   : GroupC(create)
  {}
  
  //: XML factory constructor
  LayoutC::LayoutC(const XMLFactoryContextC &factory)
   : GroupC(true)
  {
    Setup(factory);
  }
  
  //: Add a node object to the group.
  bool LayoutC::AddChildNode(const NodeC &node)
  {
    LayoutEntryC::RefT le = new LayoutEntryC(node);
    m_nodes.push_back(le);
    GroupC::AddChildNode(*le);
    UpdateLayout();
    return true;
  }

  //: Remove a node object from the group.
  bool LayoutC::RemoveChildNode(const NodeC &node)
  {
    for(std::vector<LayoutEntryC::RefT>::iterator i= m_nodes.begin();i != m_nodes.end();i++) {
      if((*i)->ElementRef().BodyPtr() == &node) {
        GroupC::RemoveChildNode(**i);
        m_nodes.erase(i);
        UpdateLayout();
        return true;
      }
    }
    rError("Failed to find node %p ",(void *)&node);
    return false;
  }

  //: Do setup from factory
  bool LayoutC::Setup(const XMLFactoryContextC &factory)
  {
    GroupC::Setup(factory);
    return true;
  }
   
  //: Update the layout
  bool LayoutC::UpdateLayout()
  {
    for(unsigned i =0;i < m_nodes.size();i++) {
      m_nodes[i]->SetPosition(Vector3dC(i*1.0,0.0,0.0));
    }
    return true;
  }

  //: Called when owner handles drop to zero.
  void LayoutC::ZeroOwners()
  {
    GroupC::ZeroOwners();
  }

  //: Process a callback.
  void LayoutC::DoCallback() {    
    GroupC::DoCallback();
  }
  
  void LinkLayout()
  {}
  
  static RavlN::XMLFactoryRegisterConvertC<LayoutC,GroupC> g_registerXMLFactoryLayout("RavlOSGN::LayoutC");

}