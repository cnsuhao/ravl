# This file is part of QMake, Quick Make System 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
################################
# Quick RAVL make system
#! file="Ravl/QMake/Util.mk"

TARG_HDRS:=$(patsubst %,$(INST_HEADER)/%,$(HEADERS)) $(LOCALHEADERS)

.PHONY : listoff listco listfix src_all srcfiles co ci mirror


CIPROG = $(LOCALBIN)/ci
COPROG = $(LOCALBIN)/co

-include $(QCWD)/defs.mk
include $(MAKEHOME)/Dirs.mk

PLIB:=$(strip $(PLIB))

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
	    $(MAKE) listoff -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
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
	    $(MAKE) listco -C $$SUBDIR  DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

listlocked:
	$(SHOWIT) echo "------ Locked files in $(DPATH)/"$$SUBDIR; \
	rlog -L -R $(SOURCESET) $(IFFYSOURCE) ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	    $(MAKE) listlocked -C $$SUBDIR  DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done


listfix:
	$(SHOWIT) grep -n "FIXME " $(SOURCESET) $(IFFYSOURCE) ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   echo "------ Fix's in $(DPATH)/"$$SUBDIR; \
	    $(MAKE) listfix -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
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
	   $(MAKE) co -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
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
	   $(MAKE) ci -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done

mirror:
	$(SHOWIT)echo "------ Mirror $(DPATH) to $(MIRROR)/$(DPATH)" ;\
	$(MKDIR_P) $(MIRROR)/$(DPATH); \
	if [ ! -h $(MIRROR)/$(DPATH)/RCS -a ! -d $(MIRROR)/$(DPATH)/RCS ] ; then \
	  $(LN_S) $(QCWD)/RCS $(MIRROR)/$(DPATH)/RCS; \
	fi ; \
	for SUBDIR in stupid_for_loop_thing $(TARG_NESTED) ; do \
	  if [ -d $$SUBDIR ] ; then \
	   $(MKDIR_P) $(MIRROR)/$(DPATH)/$$SUBDIR; \
	   if [ ! -h $(MIRROR)/$(DPATH)/$$SUBDIR/RCS -a ! -d $(MIRROR)/$(DPATH)/$$SUBDIR/RCS ] ; then \
	     $(LN_S) $(QCWD)/$$SUBDIR/RCS $(MIRROR)/$(DPATH)/$$SUBDIR/RCS; \
	   fi ; \
	   $(MAKE) mirror -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
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
	   $(MAKE) srcinst -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done


# Copy source to installation

$(MIRROR)/$(DPATH)/% : %
	@if [ ! -d $(MIRROR)/$(DPATH) ] ; then \
	  $(MKDIR) $(MIRROR)/$(DPATH) ; \
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
	    $(MAKE) udchangelog -C $$SUBDIR DPATH=$(DPATH)/$$SUBDIR -f $(MAKEHOME)/Util.mk $(DEF_INC) ; \
	  fi  \
	done	


