# This file is part of QMake, Quick Make System 
# Copyright (C) 2001, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
################################
# Quick RAVL make system
# $Id: Util.mk 5747 2006-07-19 07:57:14Z craftit $
#! rcsid="$Id: Util.mk 5747 2006-07-19 07:57:14Z craftit $"
#! file="Ravl/QMake/Util.mk"

ifndef MAKEHOME
  MAKEHOME=/path/set/by/QMake
endif

include $(MAKEHOME)/Definitions.mk

TARG_HDRS:=$(patsubst %,$(INST_HEADER)/%,$(HEADERS)) $(LOCALHEADERS)

.PHONY : listoff listco listfix src_all srcfiles co ci mirror


CIPROG = $(LOCALBIN)/ci
COPROG = $(LOCALBIN)/co

VPATH = $(QCWD)

TARG_NESTED =$(patsubst %.r,%,$(filter %.r,$(NESTED)))
NESTED_OFF = $(patsubst %.r,,$(NESTED))

ifndef MIRROR
  MIRROR=$(PROJECT_OUT)/src
endif

#################################
# Check in.

ifndef CIMSG
  CIMSG='QMake ci: No message.'
endif

IFFYSOURCE = $(LIBDEPS)
SOURCESET = $(MAINS) $(SOURCES) $(HEADERS) $(LOCALHEADERS) $(LOCAL_FILES) $(EXAMPLES) $(TESTEXES) $(HTML) $(MAN1) $(MAN3) $(MAN5) $(AUXFILES) $(EHT) $(MUSTLINK) defs.mk

TARG_SRCINST=$(patsubst %,$(MIRROR)/$(DPATH)/%,$(SOURCESET))

listoff:
	$(SHOWIT) for FILE in stupid_for_loop_thing $(NESTED_OFF) ; do \
	  if [ -d $${FILE} ] ; then \
	    echo "$(DPATH)/"$${FILE}  ; \
          fi \
	done ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	    $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) listoff -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done	

listco:
	$(SHOWIT)for FILE in stupid_for_loop_thing $(SOURCESET) $(IFFYSOURCE) ; do \
	  if [ -f RCS/$${FILE},v ] ; then \
	    if [ -w $${FILE} ] ; then \
              echo "$(DPATH)"/$${FILE} ; \
	    fi ; \
	  else \
	    if [ -f $${FILE} ] ; then \
	      echo "WARNING: No RCS file for  $(DPATH)"/$${FILE} ; \
	    fi ; \
	  fi \
	done ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	    $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) listco -C $$SUBDIR  DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

listlocked:
	$(SHOWIT) echo "------ Locked files in $(DPATH)/"$$SUBDIR; \
	rlog -L -R $(SOURCESET) $(IFFYSOURCE) ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	    $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) listlocked -C $$SUBDIR  DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done


listfix:
	$(SHOWIT) $(GREP) -n "FIXME " $(SOURCESET) $(IFFYSOURCE) ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   echo "------ Fix's in $(DPATH)/"$$SUBDIR; \
	    $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) listfix -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

#	co $(COFLAGS) $(SOURCESET) ; \

co: 
	$(SHOWIT)for FILE in stupid_for_loop_thing $(IFFYSOURCE) $(SOURCESET); do \
	  if [ -f RCS/$${FILE},v ] ; then \
	    if [ -w $${FILE} ] ; then \
	      echo "--- $${FILE} is writable, skipping. " ; \
	    else \
	      echo "--- co $${FILE} " ; \
	       $(COPROG) $(COFLAGS) $$FILE ; \
	    fi \
	  fi \
	done ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   echo "------ Checking out $(DPATH)/"$$SUBDIR; \
	   $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) co -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done


ci:
	$(SHOWIT)$(CIPROG) -m"$(CIMSG)" -t"-RAVL Source." -u $(CIFLAGS) $(SOURCESET) ; \
	for FILE in stupid_for_loop_thing $(IFFYSOURCE) ; do \
	  if [ -f $$FILE ] ; then \
	     $(CIPROG) -m"$(CIMSG)" -t"-RAVL Source." -u $(CIFLAGS) $$FILE ; \
	  fi \
	done ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   echo "------ Checking in $(DPATH)/"$$SUBDIR; \
	   $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) ci -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

mirror:
	$(SHOWIT)echo "------ Mirror $(DPATH) to $(MIRROR)/$(DPATH)" ;\
	$(MKDIR_P) $(MIRROR)/$(DPATH); \
	if [ ! -h $(MIRROR)/$(DPATH)/RCS -a ! -d $(MIRROR)/$(DPATH)/RCS ] ; then \
	  cd $(MIRROR)/$(DPATH); $(LN_S) $(QCWD)/RCS RCS; \
	fi ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   $(MKDIR_P) $(MIRROR)/$(DPATH)/$$SUBDIR; \
	   if [ ! -h $(MIRROR)/$(DPATH)/$$SUBDIR/RCS -a ! -d $(MIRROR)/$(DPATH)/$$SUBDIR/RCS ] ; then \
	     cd $(MIRROR)/$(DPATH)/$$SUBDIR; $(LN_S) $(QCWD)/$$SUBDIR/RCS RCS; \
	   fi ; \
	   $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) mirror -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

############################
# Source install.

srcinst: $(TARG_SRCINST)
	@echo "----- Installing source $(DPATH) to $(MIRROR)/$(DPATH)" ;\
	for FILE in stupid_for_loop_thing $(IFFYSOURCE) ; do \
	  if [ -f ./RCS/$${FILE},v -a ! -w $${FILE} ]  ; then \
	     $(CO) -q $${FILE} ; \
	  fi ; \
	  if [ -f $${FILE} ] ; then \
	    if [ -f $(MIRROR)/$(DPATH)/$${FILE} ] ; then \
	      $(CHMOD) +w $(MIRROR)/$(DPATH)/$${FILE} ; \
	    fi ; \
	    $(CP) $${FILE} $(MIRROR)/$(DPATH)/$${FILE}; \
	    $(CHMOD) a-w $(MIRROR)/$(DPATH)/$${FILE} ; \
          fi ; \
	done ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) srcinst -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done


# Copy source to installation

$(MIRROR)/$(DPATH)/% : %
	@if [ ! -d $(MIRROR)/$(DPATH) ] ; then \
	  $(MKDIR_P) $(MIRROR)/$(DPATH) ; \
	else  \
	  if [ -f $(MIRROR)/$(DPATH)/$(@F) ] ; then \
	    $(CHMOD) +w $(MIRROR)/$(DPATH)/$(@F) ; \
	fi ; fi ; \
	echo "----- Installing $<" ;\
	$(CP) $< $(MIRROR)/$(DPATH)/$(@F) ; \
	$(CHMOD) a-w $(MIRROR)/$(DPATH)/$(@F)

#############################
# Update Change Log.
#
# LASTCLOGUPDATE  :- Date/Time of last update of change log.
# CHANGELOGFILE :- Where to list changes found.

udchangelog:
	$(SHOWIT)echo "----- Updating change log $(DPATH) -----";
	rlog -d'>$(LASTCLOGUPDATE)' $(SOURCESET) >> $(CHANGELOGFILE)
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	    $(MAKE) $(PKG_MAKEFLAGS) $(CONFIG_MAKEFLAGS) udchangelog -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done	


#	   $(MKDIR_P) $(MIRROR)/$(DPATH)/$$SUBDIR; \
#	   cd $(MIRROR)/$(DPATH)/$$SUBDIR; $(LN_S) $$SUBDIR/RCS RCS; \

# Use RCS.

#include $(MAKEHOME)/rcs.mk

