# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2015, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here
#! rcsid="$Id$"

$(INST_OBJS)/Gnome$(OBJEXT) $(INST_DEPEND)/Gnome.d : $(INST_OBJS)/GnomeDirs.Used

ifndef PREFIX
 ifdef prefix
  PREFIX=$(prefix)
 else
  ifdef PROJECT_OUT
   PREFIX=$(PROJECT_OUT)
  endif
 endif
endif
ifndef DATADIR
 ifdef datadir
  DATADIR=$(datadir)/Ravl
 else
  ifdef PROJECT_OUT
   DATADIR=$(PROJECT_OUT)/share
  endif
 endif
endif
ifndef SYSCONFDIR
 ifdef sysconfdir
  SYSCONFDIR=$(sysconfdir)/Ravl
 else
  ifdef PROJECT_OUT
   SYSCONFDIR=$(PROJECT_OUT)/etc
  endif
 endif
endif
ifndef LIBDIR
 ifdef libdir
  LIBDIR=$(libdir)
 else
  ifdef PROJECT_OUT
   LIBDIR=$(PROJECT_OUT)/lib
  endif
 endif
endif


CFLAGS+= $(if $(PREFIX),-DPREFIX=\"$(PREFIX)\") \
         $(if $(DATADIR),-DDATADIR=\"$(DATADIR)\") \
         $(if $(SYSCONFDIR),-DSYSCONFDIR=\"$(SYSCONFDIR)\") \
         $(if $(LIBDIR), -DLIBDIR=\"$(LIBDIR)\")

CCFLAGS+= $(if $(PREFIX),-DPREFIX=\"$(PREFIX)\") \
         $(if $(DATADIR),-DDATADIR=\"$(DATADIR)\") \
         $(if $(SYSCONFDIR),-DSYSCONFDIR=\"$(SYSCONFDIR)\") \
         $(if $(LIBDIR), -DLIBDIR=\"$(LIBDIR)\")

$(INST_OBJS)/GnomeDirs.Used: FORCE
	@echo $(PREFIX) > $(LOCALTMP)/$(@F); \
         echo $(DATADIR) >> $(LOCALTMP)/$(@F); \
         echo $(SYSCONFDIR) >> $(LOCALTMP)/$(@F); \
         echo $(LIBDIR) >> $(LOCALTMP)/$(@F); \
         if [ -f $@ ] ; \
         then \
           diff $@ $(LOCALTMP)/$(@F) ; \
           if [ $$? -ne 0 ] ; \
           then \
             if [ X$(QMAKE_INFO) != X ] ; \
             then \
               echo Updating Gnome directories ; \
             fi ; \
             cp $(LOCALTMP)/$(@F) $@ ; \
           else \
             if [ X$(QMAKE_INFO) != X ] ; \
             then \
               echo Gnome directories ok ; \
             fi ; \
           fi ; \
         else \
           if [ X$(QMAKE_INFO) != X ] ; \
           then \
             echo Recording Gnome directories ; \
           fi ; \
           cp $(LOCALTMP)/$(@F) $@ ; \
         fi ; \

