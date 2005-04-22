# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2003, University of Surrey
# This code may be redistributed under the terms of the GNU
# General Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
#! rcsid="$Id$"
#! file="Ravl/Contrib/LibFFmpeg/defs.mk"

#REQUIRES = libffmpeg

PACKAGE = Ravl/Image

HEADERS = ImgIOFFmpeg.hh LibFFmpegFormat.hh

SOURCES = ImgIOFFmpeg.cc LibFFmpegFormat.cc

PLIB = RavlLibFFmpeg

MUSTLINK = MustLinkLibFFmpeg.cc

USESLIBS = RavlImage RavlIO LibFFmpeg RavlDPDisplay

PROGLIBS = RavlGUI

EXAMPLES = exFFmpeg.cc exFFmpegSeek.cc

AUXDIR = lib/RAVL/libdep

AUXFILES = LibFFmpeg.def
