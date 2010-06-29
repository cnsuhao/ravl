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

  class NodeDataC
  : public Referenced
  {
  public:
    NodeDataC(const NodeC &node)
    : m_node(&node)
    {}

    NodeC::RefT m_node;
  };

  class NodeCallbackC
  : public NodeCallback
  {
  public:
    virtual void operator()(Node *node, NodeVisitor *nv)
    {
      ref_ptr<NodeDataC> nodeDataRef = dynamic_cast<NodeDataC*>(node->getUserData());
      if (nodeDataRef && nodeDataRef->m_node.IsValid())
      {
        nodeDataRef->m_node->SigCallback()();
      }

      traverse(node, nv);
    }
  };

  NodeC::NodeC()
  : m_sigCallback(true)
  {
  }

  NodeC::~NodeC()
  {
    if (m_node && m_callback)
    {
      m_node->removeUpdateCallback(m_callback.get());
      m_callback = NULL;
    }
  }

  bool NodeC::EnableCallback(const bool enable)
  {
    if (!m_node)
      return false;

    ref_ptr<NodeDataC> nodeDataRef = new NodeDataC(*this);
    m_node->setUserData(nodeDataRef.get());

    if (enable)
    {
      if (m_callback)
        return true;

      m_callback = new NodeCallbackC;
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

}
