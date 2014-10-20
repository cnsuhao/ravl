# This file is part of CxxDoc, The RAVL C++ Documentation tool 
# Copyright (C) 2001-14, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here

DONOT_SUPPORT=VCPP

DESCRIPTION = The RAVL C++ Documentation tool

LICENSE = GPL

ifeq ($(ARC),sgi)
CCFLAGS += -CG:longbranch_limit=100000
endif

# Uncomment the following 2 lines to enable generation of a consistent date on
# CxxDoc produced documentation. This is for testing purposes and forces the
# date to Wed May 25 22:07:00 2005.
#
#CPPFLAGS+=-DCXXDOC_FALSE_TIME
#CCPPFLAGS+=-DCXXDOC_FALSE_TIME

PACKAGE= Ravl/CxxDoc

NESTED = Templates.r

LOCALHEADERS= tokenizer.h FlexLexer.h cxx.tab.h


HEADERS= Object.hh Document.hh Parser.hh CxxElements.hh \
 CxxScope.hh DocNode.hh DocTree.hh Strings.hh Executables.hh

SOURCES = Object.cc Document.cc Parser.cc \
 CxxElements.cc CxxScope.cc DocExe.cc DocNode.cc DocTree.cc \
 Strings.cc Executables.cc \
 cxx.tab.cc tokenizer.yy.cc

# We don't need to generate the lexer/parser unless we're actually extending CxxDoc
# so just build the checked-in versions (plus flex output is dependent on the 
# version of FlexLexer.h checked in)
FLEX_DO_NOT_GENERATE = 1
BISON_DO_NOT_GENERATE = 1

PLIB = RavlCxxDoc

SUMMARY_LIB=Ravl

MAINS = CxxDoc.cc 

EHT= Ravl.API.Source_Tools.CxxDoc.html Documentation.html \
 Documentation.Code.html Documentation.Sections.html \
 Documentation.docentry.html \
 Ravl.API.Source_Tools.CxxDoc.Templates.html Ravl.API.Source_Tools.CxxDoc.Internal.html \
 Ravl.undocumented.html Ravl.undocumented.anonymous.html
# Documentation.Formatting.html

USESLIBS=  RavlCore RavlOS RavlSourceTools

PROGLIBS=  
