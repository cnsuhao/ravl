# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

DESCRIPTION = Templates for CxxDoc documentation system.

AUXDIR = $(ROOTDIR)/transient/Ravl/CxxDoc/Class
# Would better be $(INST_ADMIN)/CxxDoc/Class but as Ravl does not build using
# PROJECT_NAME, we have to manually include the Ravl subdirectory in the path
# because the CxxDoc stage expects it (that phase must be run with PROJECT_NAME
# set, even for Ravl)

AUXFILES = user.class.tmpl user.namespace.tmpl develop.class.tmpl \
 develop.namespace.tmpl index.tmpl bugs.tmpl class.stmpl scope.stmpl \
 method.stmpl comment.stmpl namespace.stmpl basic.tmpl develop.index.tmpl \
 user.index.tmpl example.tmpl topbar.stmpl function.stmpl \
 develop.function.tmpl user.function.tmpl executable.stmpl \
 user.executable.tmpl develop.executable.tmpl footer.stmpl

#docentries.tmpl
