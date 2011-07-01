# This file is part of QMake, Quick Make System 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
################################
# Quick RAVL make system
# $Id: Clean.mk 5747 2006-07-19 07:57:14Z craftit $
#! rcsid="$Id: Clean.mk 5747 2006-07-19 07:57:14Z craftit $"
#! file="Ravl/QMake/Clean.mk"

ifndef MAKEHOME
  MAKEHOME=/path/set/by/QMake
endif

include $(MAKEHOME)/Definitions.mk

SLIB:=$(strip $(SLIB))

ifndef PLIB
ifdef SLIB
PLIB=$(SLIB)
endif
endif


.PHONY : clean cleandep cleanii

VPATH = $(QCWD)

ifdef USESLIBS
 ifndef LIBDEPS
  ifdef PLIB
   LOCAL_DEFBASE=$(PLIB)#
  endif
 else
  LOCAL_DEFBASE = $(patsubst %.def,%,$(LIBDEPS))
 endif
endif

TARG_NESTED =$(patsubst %.r,%,$(filter %.r,$(NESTED)))

#############################
# Clean up.

#	$(SHOWIT)if [ -d $(WORKTMP) ] ; then \
#	  $(RM) -r $(WORKTMP) ; \
#	fi ; 

#	if [ -r ii_files ] ; then \
#	  $(RM) ii_files ; \
#	fi ; 

clean: subdirs
	$(SHOWIT)if [ -d $(WORKTMP) ] ; then \
	  $(RM) -r $(WORKTMP) ; \
	fi ;  \
	if [ -d $(INST_DEPEND) ] ; then \
	  $(RM) -rf $(INST_DEPEND)/*.d ; \
	fi ; 

cleanlib: subdirs
	$(SHOWIT)if [ -d $(WORKTMP) ] ; then \
	  $(RM) -r $(WORKTMP) ; \
	fi ; \
	if [ -f $(INST_LIBDEF)/$(LOCAL_DEFBASE).def ] ; then \
	  $(RM) -f $(INST_LIBDEF)/$(LOCAL_DEFBASE).def ; \
        fi ; \
	if [ -f $(INST_LIB)/lib$(PLIB).a ] ; then \
	  $(RM) $(INST_LIB)/lib$(PLIB).a ; \
	fi ; 

cleandep: subdirs
	$(SHOWIT)if [ -d $(INST_DEPEND) ] ; then \
	  $(RM) -rf $(INST_DEPEND)/*.d ; \
	fi ; 

cleandoc: 
	$(SHOWIT) if [ -d $(INST_DOC) ] ; then \
	  $(RM) -rf $(INST_DOC) ; \
	fi ; 

subdirs:
	+ $(SHOWIT)echo "------ Cleaning $(ARC)/$(VAR)/$(BASENAME)   $(TARGET)" ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   $(MAKE) $(RAVL_MAKEFLAGS) $(CONFIG_MAKEFLAGS) $(TARGET) TARGET=$(TARGET) -C $$SUBDIR -f $(MAKEHOME)/Clean.mk $(DEF_INC) ; \
	  fi  \
	done

#include $(MAKEHOME)/rcs.mk
