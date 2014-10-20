# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2003-14, University of Surrey
# This code may be redistributed under the terms of the GNU
# General Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here

REQUIRES = libmpeg2

DONOT_SUPPORT=VCPP

PACKAGE = Ravl/Image

HEADERS = ImgIOMPEG2.hh LibMPEG2Format.hh MPEG2Demux.hh

SOURCES = ImgIOMPEG2.cc LibMPEG2Format.cc MPEG2Demux.cc

PLIB = RavlLibMPEG2

MUSTLINK = MustLinkLibMPEG2.cc

USESLIBS = RavlImage RavlIO LibMPEG2 

PROGLIBS = RavlGUI.opt RavlDPDisplay.opt

EXAMPLES = exMPEG2.cc exMPEG2Seek.cc

EXTERNALLIBS = LibMPEG2.def
