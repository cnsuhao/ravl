# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"

REQUIRES = devVideo4Linux2

SUPPORT_ONLY = linux

PACKAGE = Ravl/Image

HEADERS = V4L2Format.hh

SOURCES = V4L2Format.cc

PLIB = RavlIOV4L2

MUSTLINK = MustLinkV4L2.cc

USESLIBS = RavlImage RavlIO RavlDPDisplay

PROGLIBS = RavlGUI

ANSIFLAG =

