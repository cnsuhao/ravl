# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2015, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! file="Ravl/Core/Base/defs.mk"

$(INST_OBJS)/Resource$(OBJEXT) $(INST_DEPEND)/Resource.d: $(INST_OBJS)/Root.Used

ifndef DEFAULT_ROOT
 BUILTROOT=$(PROJECT_OUT)
else
 BUILTROOT=$(DEFAULT_ROOT)
endif

CFLAGS += $(if $(BUILTROOT),-DDEFAULT_ROOT=\"$(BUILTROOT)\")

CCFLAGS += $(if $(BUILTROOT),-DDEFAULT_ROOT=\"$(BUILTROOT)\")

$(INST_OBJS)/Root.Used: FORCE
	@echo $(BUILTROOT) > $(LOCALTMP)/$(@F); \
	 if [ -f $@ ] ; \
	 then \
	   diff $(LOCALTMP)/$(@F) $@ ; \
	   if [ $$? -ne 0 ] ; \
	   then \
             if [ X$(QMAKE_INFO) != X ] ; \
             then \
	       echo Updating BUILTROOT to $(BUILTROOT) ; \
	     fi ; \
	     cp $(LOCALTMP)/$(@F) $@ ; \
	   else \
             if [ X$(QMAKE_INFO) != X ] ; \
             then \
	       echo BUILTROOT unchanged ; \
	     fi ; \
	   fi ; \
	 else \
           if [ X$(QMAKE_INFO) != X ] ; \
           then \
             echo Recording BUILTROOT ; \
	   fi ; \
	   cp $(LOCALTMP)/$(@F) $@ ; \
	 fi


