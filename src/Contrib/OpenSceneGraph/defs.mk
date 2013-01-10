# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2010-13, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

PACKAGE=Ravl/OpenSceneGraph

LICENSE=LGPL

REQUIRES=OpenSceneGraphGTK OpenSceneGraph libGL GTKGLExt
# NOTE: Do NOT comment out the GTKGLExt, if its not enabled though it may 
# compile this library will not work.

HEADERS=Drawable.hh Sphere.hh Image.hh ImageByteRGBA.hh ImageByteRGB.hh Text.hh \
 TriMesh.hh TexTriMesh.hh Node.hh Geode.hh Group.hh Transform.hh ModelFile.hh \
 HUD.hh OpenSceneGraphWidget.hh Box.hh CanvasManipulator.hh TypeConvert.hh \
 Layout.hh LayoutStack.hh LayoutGrid.hh NodeVisitor.hh Cylinder.hh Cone.hh \
 Capsule.hh InfinitePlane.hh

SOURCES=Drawable.cc Sphere.cc Image.cc ImageByteRGBA.cc ImageByteRGB.cc Text.cc \
 TriMesh.cc TexTriMesh.cc Node.cc Geode.cc Group.cc Transform.cc ModelFile.cc \
 HUD.cc OpenSceneGraphWidget.cc Box.cc CanvasManipulator.cc TypeConvert.cc \
 Layout.cc LayoutStack.cc LayoutGrid.cc NodeVisitor.cc Cylinder.cc Cone.cc \
 Capsule.cc InfinitePlane.cc

PLIB=RavlGUIOpenSceneGraph

MAINS=exOpenSceneGraphWidgetXML.cc

MUSTLINK=MustLinkGtkGlExtInit.cc

USESLIBS=RavlCore RavlGeom RavlImage RavlGUI Ravl3D OpenGL GTK \
 OpenSceneGraphGtk RavlXMLFactory RavlRLog

PROGLIBS=RavlOS RavlImageIO RavlExtImgIO RavlRLog

#EHT=Ravl.API.GUI.OpenSceneGraph.html

EXTERNALLIBS=OpenSceneGraph.def OpenSceneGraphGtk.def

AUXFILES=exOpenSceneGraph.xml

AUXDIR=share/RAVL/OpenSceneGraph
