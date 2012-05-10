# This file is part of RAVL, Recognition And Vision Library
# This code may be redistributed under the terms of the GNU Lesser
# Copyright (C) 2005, Charles Galambos.
# see http://www.gnu.org/copyleft/lesser.html
# General Public License (LGPL). See the lgpl.licence file for details or
# file-header-ends-here

PACKAGE=Ravl/MacOSX

REQUIRES=MacOSX

HEADERS=InitAutoReleasepool.hh

SOURCES=MainRunLoop.mm

PLIB=RavlMacOSXRunLoop

MUSTLINK=linkRavlMacOSXRunLoop.cc

USESLIBS=RavlCore RavlOS OSXFoundation