// This file is part of RAVL, Recognition And Vision Library
// Copyright (C) 2011, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

namespace RavlOSGN
{

  extern bool RegisterGtkGlExtInit();
  extern bool LinkOpenSceneGraphWidget();
  extern void LinkImageByteRGBA();
  extern void LinkImageByteRGB();
  extern void LinkSphere();
  extern void LinkGroup();
  extern void LinkText();
  extern void LinkTransform();
  extern void LinkHUD();
  extern void LinkBox();
  extern void LinkModelFile();
  extern void LinkLayout();
  extern void LinkLayoutStack();
  extern void LinkLayoutGrid();
  
  void LinkGtkGlExtInit()
  {
    RegisterGtkGlExtInit();
    LinkOpenSceneGraphWidget();
    
    LinkImageByteRGBA();
    LinkImageByteRGB();
    LinkSphere();
    LinkGroup();
    LinkText();
    LinkTransform();
    LinkHUD();
    LinkBox();
    LinkModelFile();
    LinkLayout();
    LinkLayoutStack();
    LinkLayoutGrid();
  }
}
