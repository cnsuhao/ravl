# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

DESCRIPTION = Templates for CxxDoc documentation system.

AUXDIR = $(ROOTDIR)/transient/Ravl/CxxDoc/DocNode
# Would better be $(INST_ADMIN)/CxxDoc/DocNode but as Ravl does not build using
# PROJECT_NAME, we have to manually include the Ravl subdirectory in the path
# because the CxxDoc stage expects it (that phase must be run with PROJECT_NAME
# set, even for Ravl)

AUXFILES = develop.docnode.tmpl user.docnode.tmpl docnode.stmpl subNode.stmpl \
 sitemap.tmpl objects.stmpl

