// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Group.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Group.hh"
#include <osg/Group>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlOSGN
{

  using namespace osg;

  GroupC::GroupC(bool create)
  {
    if (create)
      m_node = new Group();
  }

  GroupC::~GroupC()
  {
  }

  bool GroupC::AddChild(const NodeC::RefT &node)
  {
    if (!m_node || !node.IsValid())
      return false;

    NodeC::RefT nodeRef = node;
    ref_ptr<Group> groupRef = m_node->asGroup();
    if (!groupRef)
      return false;

    groupRef->addChild(nodeRef->Node());

    return true;
  }

  bool GroupC::RemoveChild(const NodeC::RefT &node)
  {
    if (!m_node || !node.IsValid())
      return false;

    NodeC::RefT nodeRef = node;
    ref_ptr<Group> groupRef = m_node->asGroup();
    if (!groupRef)
      return false;

    groupRef->removeChild(nodeRef->Node());

    return true;
  }

  bool GroupC::AddChildNode(const NodeC &node) {
    ref_ptr<Group> groupRef = m_node->asGroup();
    if (!groupRef)
      return false;
    groupRef->addChild(const_cast<NodeC &>(node).Node());
    return true;
  }
  //: Add a node object to the group.

  bool GroupC::RemoveChildNode(const NodeC &node) {
    ref_ptr<Group> groupRef = m_node->asGroup();
    if (!groupRef)
      return false;
    groupRef->removeChild(const_cast<NodeC &>(node).Node());
    return true;
  }
  //: Remove a node object from the group.

}
