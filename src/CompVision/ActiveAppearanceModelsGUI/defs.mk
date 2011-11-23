# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2005, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/CompVision/ActiveAppearanceModels/defs.mk"

DESCRIPTION = Active appearance model gui

PACKAGE = Ravl/Image

DONOT_SUPPORT = arm

REQUIRES=libGTK2

MAINS =aamViewShapeModel.cc 

EXAMPLES =  aamViewShapeModel.cc 

HEADERS = AAMViewLib.hh

SOURCES = AAMViewLib.cc

MUSTLINK =

PLIB = RavlAAMGUI

USESLIBS = RavlAAM RavlGUI2D 

PROGLIBS = RavlImageIO RavlExtImgIO RavlMathIO RavlDPDisplay.opt RavlImgIOV4L2.opt

