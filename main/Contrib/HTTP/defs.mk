# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"
#! file="Contrib/HTTP/defs.mk"

PACKAGE=Ravl/IO

EXAMPLES = exHTTP.cc

HEADERS= HTTPStream.hh

SOURCES= HTTPStream.cc

MUSTLINK= HTTPIO.cc

PLIB= RavlHTTPIO

USESLIBS=RavlCore RavlIO RavlOS RavlThreads libcurl

AUXFILES= libcurl.def

AUXDIR=lib/RAVL/libdep

