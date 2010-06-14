# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2010, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE=Ravl/OpenSceneGraph

LICENSE=LGPL

REQUIRES=OpenSceneGraph libGL

HEADERS=Drawable.hh Sphere.hh ImageByteRGBA.hh Text.hh \
 Node.hh Geode.hh Group.hh Transform.hh ModelFile.hh HUD.hh \
 OpenSceneGraphWidget.hh

SOURCES=Drawable.cc Sphere.cc ImageByteRGBA.cc Text.cc \
 Node.cc Geode.cc Group.cc Transform.cc ModelFile.cc HUD.cc \
 OpenSceneGraphWidget.cc

PLIB=RavlGUIOpenSceneGraph

MAINS=exOpenSceneGraphWidget.cc

MUSTLINK=MustLinkGtkGlExtInit.cc

USESLIBS=RavlCore RavlGeom RavlImage RavlGUI OpenGL GTK OpenSceneGraphGtk

#EHT=Ravl.API.GUI.OpenSceneGraph.html

PROGLIBS=RavlOS RavlImageIO RavlExtImgIO

AUXFILES=OpenSceneGraphGtk.def

AUXDIR=lib/RAVL/libdep
