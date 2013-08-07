# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2013, React AI Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL) Version 2.1  See the lgpl.licence file
# for details or see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE=Ravl/IO

HEADERS= DataStreamIDX.hh IPortIDX.hh OPortIDX.hh IPortIDXFormat.hh FileUtilities.hh

SOURCES= DataStreamIDX.cc IPortIDX.cc OPortIDX.cc IPortIDXFormat.cc FileUtilities.cc  

MUSTLINK=LinkRavlFileIDX.cc

PLIB=RavlFileIDX

USESLIBS=RavlCore RavlIO RavlOSIO 

MAINS=doFileIDX.cc doPrepData.cc

PROGLIBS=RavlPatternRec RavlPatternRecIO