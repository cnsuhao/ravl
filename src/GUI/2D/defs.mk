# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/GUI/2D/defs.mk"

PACKAGE=Ravl/GUI

HEADERS=Canvas.hh RawZoomCanvas.hh GUIMarkupLayerEditor.hh MarkupImageRGBA.hh  MarkupInfo.hh \
  MarkupLine2d.hh  MarkupPoint2d.hh FrameMarkup.hh  GUIMarkupCanvas.hh  MarkupImageByte.hh \
  MarkupImageRGB.hh   MarkupLayerInfo.hh  MarkupNote.hh    MarkupPolygon2d.hh

SOURCES=Canvas.cc RawCanvas2d.cc RawZoomCanvas.cc FrameMarkup.cc GUIMarkupLayerEditor.cc \
  MarkupImageRGB.cc   MarkupLine2d.cc   MarkupPolygon2d.cc MarkupImageByte.cc  MarkupInfo.cc \
  MarkupNote.cc GUIMarkupCanvas.cc  MarkupImageRGBA.cc MarkupLayerInfo.cc  MarkupPoint2d.cc  


MAINS=
#exRawZoomCanvas.cc

PLIB=RavlGUI2D

SUMMARY_LIB=Ravl

REQUIRES=libGTK2 

USESLIBS=RavlGUI RavlImage RavlIO RavlMath GTK

EXAMPLES= exCanvas.cc exRawZoomCanvas.cc exGUIMarkupCanvas.cc 

HTML=canvas_colours.html
