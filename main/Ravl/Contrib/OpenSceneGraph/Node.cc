// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Node.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Node.hh"
#include "Ravl/XMLFactoryRegister.hh"

#include <osg/Node>
#include <osg/StateSet>
#include <osg/Referenced>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlOSGN
{

  using namespace osg;


  // Call back.
  
  class NodeCallbackC
  : public NodeCallback
  {
  public:
    NodeCallbackC(const NodeC &node)
     : m_node(&node)
    {}
    
    virtual void operator()(Node *node, NodeVisitor *nv)
    {
      if (m_node.IsValid())
      { m_node->SigCallback()(); }
      
      traverse(node, nv);
    }
    
    NodeC::RefT m_node;
  };

  // Constructor.

  NodeC::NodeC()
   : m_sigCallback(true)
  {
    
  }

  //: XML Factory constructor
  NodeC::NodeC(const XMLFactoryContextC &factory)
   : m_sigCallback(true)
  {
    
  }

  //! Destructor.
  
  NodeC::~NodeC()
  {
    if (m_node && m_callback)
    {
      m_node->removeUpdateCallback(m_callback.get());
      m_callback = NULL;
    }
  }

  //: Do setup from factory
  
  bool NodeC::Setup(const XMLFactoryContextC &factory)
  {
    if(factory.AttributeBool("bringToFront",false)) {
      BringToFront();
    }
    return true;
  }
  
  //: Called when owner handles drop to zero.
  
  void NodeC::ZeroOwners()
  {
    RavlN::RCLayerBodyC::ZeroOwners();
  }
  

  bool NodeC::EnableCallback(const bool enable)
  {
    if (!m_node)
      return false;
    
    m_node->setUserData(new NodeRefC(this));
    
    if (enable)
    {
      if (m_callback)
        return true;

      m_callback = new NodeCallbackC(*this);
      m_node->setUpdateCallback(m_callback.get());
    }
    else
    {
      if (!m_callback)
        return true;

      m_node->removeUpdateCallback(m_callback.get());
      m_callback = NULL;
    }

    return true;
  }

  bool NodeC::BringToFront()
  {
    if (!m_node)
      return false;

    ref_ptr<StateSet> stateSetRef = m_node->getOrCreateStateSet();

    stateSetRef->setMode(GL_DEPTH_TEST, StateAttribute::OFF);
    stateSetRef->setRenderBinDetails(10, "RenderBin");

    m_node->setStateSet(stateSetRef);

    return true;
  }

  //: Set the name of the node.
  void NodeC::SetName(const std::string &name) {
    m_node->setName(name);
  }

  //: Get node handle if it exists.
  bool NodeC::GetNode(osg::Node *theOsgNode,NodeC::RefT &node) {
    NodeRefC *nd = dynamic_cast<NodeRefC *>(theOsgNode->getUserData());
    if(nd == 0)
      return false;
    node = nd->Node();
    return true;
  }

}
