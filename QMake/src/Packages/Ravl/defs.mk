# This file is part of QMake, Quick Make System 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU General 
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
#! file="Ravl/QMake/defs.mk"

#PACKAGENAME= QMake
DONOT_SUPPORT=VCPP

DESCRIPTION = Quick Make System

#PREBUILDSTEP=echo $(PROJECT_OUT)

LICENSE= GPL

MAINS=SysConf.cc untouch.cc findBuildTag.cc

AUXDIR=share/RAVL/QMake

AUXFILES =  QMake.mk \
  Definitions.mk Main.mk MainDep.mk Doc.mk Clean.mk \
  Install.pl dummymain.c \
  Help.txt Defs.txt \
  BinDep.pl mkdefs.pl \
  QLibs.pl GlobalMake qmake.cshrc qmake.sh \
  AutoBuild.pl AutoBuild.sample.conf project.qpr 

HTML=Example.def.html  Help.txt

EHT= exeSysConf.eht Ravl.QMake.html Ravl.QMake.Defs.html  \
 Ravl.QMake.Build_Structure.html Ravl.QMake.AutoBuild.html

SCRIPTS=qm QLibs

SCRIPT_INSTALL=$(PERL) ./Install.pl $(PROJECT_OUT)/share/RAVL/QMake $(PROJECT_OUT)

AUXINSTALL=$(PERL) ./Install.pl $(PROJECT_OUT)/share/RAVL/QMake $(PROJECT_OUT)
