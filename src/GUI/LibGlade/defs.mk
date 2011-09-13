# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2004-11, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

REQUIRES=libglade2 libGTK2

PACKAGE=Ravl/GUI

HEADERS= GladeWidget.hh GladeXML.hh GladeWindow.hh

SOURCES= GladeWidget.cc GladeXML.cc GladeWindow.cc

PLIB= RavlLibGlade

USESLIBS= RavlGUI RavlXMLFactory libglade

AUXFILES= exlibgladecustommain.glade  exlibgladecustomwidget.glade  exlibglade.glade

AUXDIR=share$(PROJECT_DIR)/Glade

EXAMPLES= exLibGlade.cc exLibGladeCustom.cc

MUSTLINK=linkRavlLibGlade.cc
