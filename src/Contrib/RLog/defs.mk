# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2008, OmniPerception Ltd.
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

DONOT_SUPPORT=

REQUIRES=RLog

PACKAGE=Ravl

HEADERS=RLog.hh StdioDateNode.hh

SOURCES=RLog.cc StdioDateNode.cc

PLIB=RavlRLog

SUMMARY_LIB=Ravl

USESLIBS=RavlCore RavlOS RLog

CCPPFLAGS += -DRLOG_COMPONENT=Ravl

EXAMPLES= exRLog.cc
