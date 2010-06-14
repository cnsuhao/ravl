// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2010, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
////////////////////////////////////////////////////////////////
//! file = "Ravl/Contrib/OpenSceneGraph/Text.cc"
//! lib = RavlGUIOpenSceneGraph
//! author = "Warren Moore"

#include "Ravl/OpenSceneGraph/Text.hh"
#include <osgText/Text>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace
{
  
  osgText::Text::AlignmentType g_lookupTextAlignment[] =
  {
    osgText::TextBase::LEFT_TOP,
    osgText::TextBase::LEFT_CENTER,
    osgText::TextBase::LEFT_BOTTOM,
    osgText::TextBase::CENTER_TOP,
    osgText::TextBase::CENTER_CENTER,
    osgText::TextBase::CENTER_BOTTOM,
    osgText::TextBase::RIGHT_TOP,
    osgText::TextBase::RIGHT_CENTER,
    osgText::TextBase::RIGHT_BOTTOM,
    osgText::TextBase::LEFT_BASE_LINE,
    osgText::TextBase::CENTER_BASE_LINE,
    osgText::TextBase::RIGHT_BASE_LINE,
    osgText::TextBase::LEFT_BOTTOM_BASE_LINE,
    osgText::TextBase::CENTER_BOTTOM_BASE_LINE,
    osgText::TextBase::RIGHT_BOTTOM_BASE_LINE
  };

  osgText::Text::AxisAlignment g_lookupTextAxisAlignment[] =
  {
    osgText::TextBase::XY_PLANE,
    osgText::TextBase::REVERSED_XY_PLANE,
    osgText::TextBase::XZ_PLANE,
    osgText::TextBase::REVERSED_XZ_PLANE,
    osgText::TextBase::YZ_PLANE,
    osgText::TextBase::REVERSED_YZ_PLANE,
    osgText::TextBase::SCREEN
  };

}

namespace RavlOSGN
{

  using namespace osg;
  using namespace osgText;

  TextC::TextC(const std::string &text)
  {
    ref_ptr<Text> textRef = new Text();
    if (textRef)
      textRef->setText(text);

    m_drawable = textRef;
  }

  TextC::~TextC()
  {
  }

  bool TextC::SetText(const std::string &text)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    textRef->setText(text);

    return true;
  }

  bool TextC::SetFont(const std::string &filename)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    textRef->setFont(filename);

    return true;
  }

  bool TextC::SetAlignment(TextAlignmentT alignment)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    TextBase::AlignmentType alignmentType = g_lookupTextAlignment[alignment];
    textRef->setAlignment(alignmentType);
    
    return true;
  }

  bool TextC::SetAxisAlignment(TextAxisAlignmentT axis)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    TextBase::AxisAlignment axisAlignment = g_lookupTextAxisAlignment[axis];
    textRef->setAxisAlignment(axisAlignment);

    return true;
  }

  bool TextC::SetColour(const RavlImageN::RealRGBAValueC &colour)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    textRef->setColor(Vec4(colour.Red(), colour.Green(), colour.Blue(), colour.Alpha()));

    return true;
  }

  bool TextC::SetPosition(const RavlN::Vector3dC &position)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    textRef->setPosition(Vec3(position.X(), position.Y(), position.Z()));

    return true;
  }

  bool TextC::SetSize(RavlN::RealT size)
  {
    if (!m_drawable)
      return false;

    ref_ptr<Text> textRef = dynamic_cast<Text*>(m_drawable.get());
    if (!textRef)
      return false;

    textRef->setCharacterSize(size);

    return true;
  }

}
