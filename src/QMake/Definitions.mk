# This file is part of QMake, Quick Make System 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
#! file="Ravl/QMake/Definitions.mk"

# Development Bodges....

PAR_MAKE=--jobs=$(PROCS)

# End of bodges

CPPFLAGS = -DPROJECT_OUT=\"$(PROJECT_OUT)\" -DCPUG_ARCH=\"$(ARC)\"
CCPPFLAGS = -DPROJECT_OUT=\"$(PROJECT_OUT)\" -DCPUG_ARCH=\"$(ARC)\"
ifdef SHAREDBUILD
  CPPFLAGS += -DCPUG_VAR_SHARED=1
  CCPPFLAGS += -DCPUG_VAR_SHARED=1
endif

LOCALBIN=$(INSTALLHOME)/bin/utils/opt/bin
# Utilities used during the build process
# Standardise on opt version for consistency

NOLIBRARYPRELINK=1


ifdef CONFIGFILE
  include $(CONFIGFILE)
else
  include $(INSTALLHOME)/Config.Local
endif


ifndef QCWD
 QCWD :=$(shell $(GET_CWD))#
endif

export DPATH:=$(notdir $(QCWD))

# Include local definitions from the current build directory
-include $(QCWD)/defs.mk

# Sanitise relevant imported definitions
# Anything that will possibly be used to form a file path or be used as a make
# target must not have leading or trailing spaces. Lists (SOURCES, MAINS, etc.)
# are usually safe as we normally extract those using makes string substitution
# functions which work on space delimited words anyway.
ifdef PLIB
 PLIB:=$(strip $(PLIB))#
endif
ifdef USESLIBS
 USESLIBS:=$(strip $(USESLIBS))#
endif
ifdef AUXDIR
AUXDIR:=$(strip $(AUXDIR))#
endif
ifdef LIBDEPS
LIBDEPS:=$(strip $(LIBDEPS))#
endif
ifdef EXTERNAL_PROJECTS
EXTERNAL_PROJECTS:=$(strip $(EXTERNAL_PROJECTS))#
endif
ifdef AUXFILES
AUXFILES:=$(strip $(AUXFILES))#
endif

# Directories used in making.

ifndef MAKEHOME
  MAKEHOME=/path/set/by/QMake
endif

ifndef INSTALLHOME
  INSTALLHOME = $(MAKEHOME)/../../..#
endif


# Select required ANSI setting (individual over-ride is by ANSIFLAG for
# effecting both compilers, CANSIFLAG for the C compiler and  CCANSIFLAG for
# the C++ compiler).
ifdef CANSIFLAG
  RAVL_GLOBAL_CFLAGS+=$(CANSIFLAG)
else
  ifdef ANSIFLAG
    RAVL_GLOBAL_CFLAGS+=$(ANSIFLAG)
  else
    RAVL_GLOBAL_CFLAGS+=$(RAVL_CANSIFLAG)
  endif
endif

ifdef CCANSIFLAG
  RAVL_GLOBAL_CCFLAGS+=$(CCANSIFLAG)
else
  ifdef ANSIFLAG
    RAVL_GLOBAL_CCFLAGS+=$(ANSIFLAG)
  else
    RAVL_GLOBAL_CCFLAGS+=$(RAVL_CCANSIFLAG)
  endif
endif

ifndef PACKAGE
 PACKAGE=local
endif

ifdef QMAKE_INFO
  SHOWIT = 
else
  SHOWIT = @
endif

# FIXME: PACKAGEDIR is intented to replace all the RAVL's hardwired into
# the make system.  Before it can be used we have to disentangle
# what is QMAKE and the base installation.

ifndef PACKAGENAME
  PACKAGEDIR=RAVL
else
  PACKAGEDIR=$(PACKAGENAME)
endif

PACKAGE:=$(strip $(PACKAGE))#

ifndef PLIB
  BASENAME:=$(PACKAGE)/None#
else
  BASENAME:=$(PLIB)#
endif

ifndef SHAREDBUILD
 SHARED_LIB_POSTFIX=#
else
 SHARED_LIB_POSTFIX=/shared#
endif

ifndef PAGER
  PAGER=$(RAVL_PAGER)
endif

##########################
# Varient information

ifndef NOVAR
 ifndef VAR
   VAR=check#
 endif
endif

ifndef BASE_VAR
 ifeq ($(VAR),prof)
  BASE_VAR=opt#
 else
  ifeq ($(VAR),gprof)
   BASE_VAR=opt#
  else
   ifeq ($(VAR),opt)
    BASE_VAR=opt#
   else
    BASE_VAR=check#
   endif 
  endif
 endif
endif

##########################
# Roots of working directories.

#################
# Temporary directories...

ifndef DEFAULTTMP 
 TMP=/tmp
else
 TMP=$(DEFAULTTMP)
endif

WORKTMP=$(LOCALTMP)/$(ARC)/$(BASENAME)/$(VAR)$(SHARED_LIB_POSTFIX)

# A file that definitly doesn't exist.
#NOFILE = /notme/hiya/fruitcake/whippy

##########
# Working directories...

PROJ_ID := $(subst ~,,$(PROJECT_OUT))

ifndef LOCALTMP
 LOCALTMP:=$(TMP)/$(LOGNAME)/qm/$(PROJ_ID)/
endif

# ROOTDIR is where the software will be installed.

ifndef ROOTDIR
  ROOTDIR:=$(PROJECT_OUT)
endif

#########################
# Target directories.

# Admin files used in build but not needed after.

INST_ADMIN=$(ROOTDIR)/share/RAVL/Admin

# Documentation

INST_AUTODOC=$(ROOTDIR)/share/doc/Auto
INST_DOCNODE=$(INST_AUTODOC)/DocNode
INST_DOC= $(ROOTDIR)/share/doc/RAVL
INST_PDOC=$(INST_DOC)/$(PACKAGE)
INST_HTML=$(INST_DOC)/html
INST_DOCEXAMPLES=$(INST_DOC)/examples
INST_EHT= $(INST_ADMIN)/AutoDoc/EHT
INST_MAN1=$(ROOTDIR)/share/man/man1
INST_MAN3=$(ROOTDIR)/share/man/man3
INST_MAN5=$(ROOTDIR)/share/man/man5

# Auxilary files.
INST_AUX=$(ROOTDIR)/$(AUXDIR)

# Binaries

INST_LIB=$(ROOTDIR)/lib/RAVL/$(ARC)/$(VAR)$(SHARED_LIB_POSTFIX)
INST_OBJS=$(WORKTMP)/objs
INST_FORCEOBJS = $(ROOTDIR)/lib/RAVL/$(ARC)/obj

# Test stuff.
INST_TEST=$(INST_ADMIN)/$(ARC)/test
INST_TESTBIN=$(INST_TEST)/bin
INST_TESTLOG=$(INST_TEST)/log
INST_TESTDB =$(INST_TEST)/TestExes

INST_LIBDEF=$(ROOTDIR)/lib/RAVL/libdep
INST_BIN=$(ROOTDIR)/bin/utils/$(VAR)$(SHARED_LIB_POSTFIX)/bin
INST_GENBIN=$(ROOTDIR)/bin# Machine independent scripts.
INST_INCLUDE:=$(ROOTDIR)/include
INST_DEPEND=$(INST_ADMIN)/$(ARC)/depend/$(PACKAGE)/$(BASENAME)

ifeq ($(PACKAGE),local)
INST_HEADER:=$(INST_INCLUDE)
INST_HEADERSYM:=$(INST_ADMIN)/syminc
else
INST_HEADER:=$(INST_INCLUDE)/$(PACKAGE)
INST_HEADERSYM:=$(INST_ADMIN)/syminc/$(PACKAGE)
endif

INST_HEADERCERT:=$(INST_ADMIN)/Cert/$(PACKAGE)

# Java classes

INST_JAVA    = $(ROOTDIR)/java
INST_JAVAEXE = $(INST_BIN)

############################
# include info on RAVL system.

#INCLUDES=


# Were to look for .def files, First in the current directory,
# then the   current PROJECT_OUT def's and finally those that 
# were installed with the make system. 

DEF_INC = -I. -I$(INST_LIBDEF) -I$(INSTALLHOME)/lib/RAVL/libdep  $(patsubst %,-I%/lib/RAVL/libdep,$(EXTERNAL_PROJECTS))

############################
# Some targets.

# Published dependancy flag.
TARG_DEPFLAG=$(INST_ADMIN)/$(ARC)/depend/$(PACKAGE)/$(BASENAME)/.depend
TARG_HDRFLAG=$(INST_ADMIN)/$(ARC)/depend/$(PACKAGE)/$(BASENAME)/.header

##############################
# Make setup

# Parallel Jobs
ifndef PROCS
  SYSCONF:=$(LOCALBIN)/SysConf

  PROCS=$(shell $(SYSCONF) -1)
endif
export PROCS

# Basic make.
MAKESM=$(MAKE) $(RAVL_MAKEFLAGS) $(CONFIG_MAKEFLAGS) $(DEF_INC) 
#$(PAR_MAKE)

# Make with dependancies
MAKEMD=$(MAKESM) -f $(MAKEHOME)/MainDep.mk

# Make with dependancies
MAKEMO=$(MAKESM) -f $(MAKEHOME)/Main.mk

# Clean up makefile.
MAKECL=$(MAKESM) -f $(MAKEHOME)/Clean.mk

# Clean up makefile.
MAKEUT=$(MAKESM) -f $(MAKEHOME)/Util.mk

# Clean up makefile.
MAKEDC=$(MAKESM) -f $(MAKEHOME)/Doc.mk

# With Show it prefix.

SMAKESM=+ $(SHOWIT)$(MAKESM)
SMAKEMD=+ $(SHOWIT)$(MAKEMD)
SMAKEMO=+ $(SHOWIT)$(MAKEMO)
SMAKECL=+ $(SHOWIT)$(MAKECL)
SMAKEUT=+ $(SHOWIT)$(MAKEUT)
SMAKEDC=+ $(SHOWIT)$(MAKEDC)


# Define default file extensions used (and ensure any user-specified extensions
# do not have any erroneous spaces)

ifndef SHAREDEXT 
 SHAREDEXT:=so
else
 SHAREDEXT:=$(strip $(SHAREDEXT))#
endif

ifndef SHAREDBUILD
 LIBEXT:=.a
else
 LIBEXT:=.$(SHAREDEXT)
endif

# Default Object file extension
ifndef OBJEXT
  OBJEXT:=.o#
else
 OBJEXT:=$(strip $(OBJEXT))#
endif

# Default C++ source file extension.
ifndef CXXEXT
  CXXEXT:=.cc#
else
 CXXEXT:=$(strip $(CXXEXT))#
endif

# Default C++ auxilary source file extension. (used to force template instansiation.)
ifndef CXXAUXEXT
  CXXAUXEXT:=.cxx#
else
 CXXAUXEXT:=$(strip $(CXXAUXEXT))#
endif

# Default C++ header file extension.
ifndef CHXXEXT
  CHXXEXT:=.hh#
else
 CHXXEXT:=$(strip $(CHXXEXT))#
endif

# Default C source file extension.
ifndef CEXT
  CEXT:=.c#
else
 CEXT:=$(strip $(CEXT))#
endif

# Default C header file extension.
ifndef CHEXT
  CHEXT:=.h#
else
 CHEXT:=$(strip $(CHEXT))#
endif

# Extension expected on executables.
ifndef EXEEXT
  EXEEXT:=#
else
 EXEEXT:=$(strip $(EXEEXT))#
endif


# Set flags for dependancy production

MKDEPFLAGS=-MM
AMKDEPFLAGS = -Wp,-MMD,$(WORKTMP)/$*.d
MKDEPUP = echo '$$(INST_OBJS)/' | $(TR) '\n' ' '  > $(INST_DEPEND)/$*.d ; cat $(WORKTMP)/$*.d  >> $(INST_DEPEND)/$*.d ; \
 $(RM) $(WORKTMP)/$*.d


# Set appropriate RAVL_CFLAGS/CCFLAGS/LDFLAGS/LDLIBFLAGS for current build

ifeq ($(VAR),check)
  RAVL_CFLAGS=$(CONFIGURE_CFLAGS) $(RAVL_GLOBAL_CFLAGS) $(RAVL_DEFAULT_CFLAGS) $(RAVL_CHECK_CFLAGS)
  RAVL_CCFLAGS=$(CONFIGURE_CCFLAGS) $(RAVL_GLOBAL_CCFLAGS) $(RAVL_DEFAULT_CCFLAGS) $(RAVL_CHECK_CCFLAGS)
  RAVL_LDFLAGS=$(CONFGIURE_LDFLAGS) $(RAVL_GLOBAL_LDFLAGS) $(RAVL_DEFAULT_LDFLAGS) $(RAVL_CHECK_LDFLAGS)
  RAVL_LDLIBFLAGS=$(CONFGIURE_LDLIBFLAGS) $(RAVL_GLOBAL_LDLIBFLAGS) $(RAVL_DEFAULT_LDLIBFLAGS) $(RAVL_CHECK_LDLIBFLAGS)
endif

ifeq ($(VAR),debug)
  RAVL_CFLAGS=$(CONFIGURE_CFLAGS) $(RAVL_GLOBAL_CFLAGS) $(RAVL_DEFAULT_CFLAGS) $(RAVL_DEBUG_CFLAGS)
  RAVL_CCFLAGS=$(CONFIGURE_CCFLAGS) $(RAVL_GLOBAL_CCFLAGS) $(RAVL_DEFAULT_CCFLAGS) $(RAVL_DEBUG_CCFLAGS)
  RAVL_LDFLAGS=$(CONFGIURE_LDFLAGS) $(RAVL_GLOBAL_LDFLAGS) $(RAVL_DEFAULT_LDFLAGS) $(RAVL_DEBUG_LDFLAGS)
  RAVL_LDLIBFLAGS=$(CONFGIURE_LDLIBFLAGS) $(RAVL_GLOBAL_LDLIBFLAGS) $(RAVL_DEFAULT_LDLIBFLAGS) $(RAVL_DEBUG_LDLIBFLAGS)
endif

ifeq ($(VAR),opt)
  RAVL_CFLAGS=$(CONFIGURE_CFLAGS) $(RAVL_GLOBAL_CFLAGS) $(RAVL_OPT_CFLAGS)
  RAVL_CCFLAGS=$(CONFIGURE_CCFLAGS) $(RAVL_GLOBAL_CCFLAGS) $(RAVL_OPT_CCFLAGS)
  RAVL_LDFLAGS=$(CONFGIURE_LDFLAGS) $(RAVL_GLOBAL_LDFLAGS) $(RAVL_OPT_LDFLAGS)
  RAVL_LDLIBFLAGS=$(CONFGIURE_LDLIBFLAGS) $(RAVL_GLOBAL_LDLIBFLAGS) $(RAVL_OPT_LDLIBFLAGS)
endif

ifeq ($(VAR),prof)
  RAVL_CFLAGS=$(CONFIGURE_CFLAGS) $(RAVL_GLOBAL_CFLAGS) $(RAVL_OPT_CFLAGS) $(RAVL_PROF_CFLAGS)
  RAVL_CCFLAGS=$(CONFIGURE_CCFLAGS) $(RAVL_GLOBAL_CCFLAGS) $(RAVL_OPT_CCFLAGS) $(RAVL_PROF_CCFLAGS)
  RAVL_LDFLAGS=$(CONFGIURE_LDFLAGS) $(RAVL_GLOBAL_LDFLAGS) $(RAVL_OPT_LDFLAGS) $(RAVL_PROF_LDFLAGS)
  RAVL_LDLIBFLAGS=$(CONFGIURE_LDLIBFLAGS) $(RAVL_GLOBAL_LDLIBFLAGS) $(RAVL_OPT_LDLIBFLAGS) $(RAVL_PROF_LDLIBFLAGS)
endif

ifeq ($(VAR),gprof)
  RAVL_CFLAGS=$(CONFIGURE_CFLAGS) $(RAVL_GLOBAL_CFLAGS) $(RAVL_OPT_CFLAGS) $(RAVL_GPROF_CFLAGS)
  RAVL_CCFLAGS=$(CONFIGURE_CCFLAGS) $(RAVL_GLOBAL_CCFLAGS) $(RAVL_OPT_CCFLAGS) $(RAVL_GPROF_CCFLAGS)
  RAVL_LDFLAGS=$(CONFGIURE_LDFLAGS) $(RAVL_GLOBAL_LDFLAGS) $(RAVL_OPT_LDFLAGS) $(RAVL_GPROF_LDFLAGS)
  RAVL_LDLIBFLAGS=$(CONFGIURE_LDLIBFLAGS) $(RAVL_GLOBAL_LDLIBFLAGS) $(RAVL_OPT_LDLIBFLAGS) $(RAVL_GPROF_LDLIBFLAGS)
endif

ifdef SHAREDBUILD
  RAVL_CFLAGS+=$(RAVL_SHARED_CFLAGS)
  RAVL_CCFLAGS+=$(RAVL_SHARED_CCFLAGS)
  RAVL_LDFLAGS+=$(RAVL_SHARED_LDFLAGS)
  RAVL_LDLIBFLAGS+=$(RAVL_SHARED_LDLIBFLAGS)
endif

# Add in any USERxxFLAGS from .def file
# Bear in mind QMake sports only a USERCFLAGS that doubles for C++ as well
RAVL_CFLAGS+=$(USERCFLAGS)
RAVL_CCFLAGS+=$(USERCFLAGS)
RAVL_LDFLAGS+=$(USERLDFLAGS)

# Ravl uses CPPFLAGS directly (again QMake has no seperate C++ variable)
CPPFLAGS+=$(USERCPPFLAGS)
CCPPFLAGS+=$(USERCPPFLAGS)
