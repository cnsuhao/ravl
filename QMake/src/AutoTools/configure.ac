#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.65])

AC_INIT([QMake], [1.5.8030], [John.Field@Surrey.ac.UK])

# This file is part of QMake, Quick Make System
# Copyright (C) 2011, University of Surrey
# This code may be redistributed under the terms of the GNU General
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html

dnl Include standard QMake autoconf macros
m4_include([./AcMacros/QM_CHOOSE_GETCWD])
m4_include([./AcMacros/QM_CHOOSE_RPATH])
m4_include([./AcMacros/QM_COMPILE_CK_FOR_LIB])
m4_include([./AcMacros/QM_Debug_Fn])
m4_include([./AcMacros/QM_DEFINE_EXTLIB_CC])
m4_include([./AcMacros/QM_DEFINE_EXTLIB_PC])
m4_include([./AcMacros/QM_DEFINE_VIRTUAL_EXTLIB])
m4_include([./AcMacros/QM_LOCATE_KEY_TOOLS])
m4_include([./AcMacros/QM_LOCATE_LD])
m4_include([./AcMacros/QM_LOCATE_MAKE])
m4_include([./AcMacros/QM_LOCATE_QMAKE])
m4_include([./AcMacros/QM_PROCESS_EXTLIBS])
m4_include([./AcMacros/QM_SET_ARC])
m4_include([./AcMacros/QM_SET_ARCHIVER_ARFLAGS])
m4_include([./AcMacros/QM_SET_COMPILER_CFLAGS])
m4_include([./AcMacros/QM_SET_COMPILER_CCFLAGS])
m4_include([./AcMacros/QM_SET_LINKER_LDFLAGS])
m4_include([./AcMacros/QM_SET_MAKE_MAKEFLAGS])

dnl#JFi:ToDo-Check for ver 5 of Perl as a requirement

# Main body of configure script

# First determine the current platform
AC_CANONICAL_HOST


dnl It would be good to enable the following code, however _AC_... aren't 
dnl generally available macros and may dissappear/change in differing versions
dnl of autotools so its best not to rely on them. If only there was a supported
dnl means of doing this.. (you can check for the preprocessor (as we do, above)
dnl but not all of the calls that _AC_INCLUDES_DEFAULT_REQUIREMENTS expands to
dnl are individually available.
dnl
dnl#      Cause the checks for the C pre-processor and default headers to be done now
dnl#      (This can prevent multiple calls within the script's body)
dnl       _AC_INCLUDES_DEFAULT_REQUIREMENTS



# Locate the tools needed to run this script (shell utils, compilers etc.)
QM_LOCATE_KEY_TOOLS
dnl Declare key tools for output to the Config.X file.
dnl Ensure this tallies with the list of tools regarded as such in the
dnl locate macro. There must be no additions or subtractions here.
dnl
AC_SUBST([AR])
AC_SUBST([AWK])
AC_SUBST([CC])
AC_SUBST([CPP])
AC_SUBST([CXX])
AC_SUBST([CXXCPP])
AC_SUBST([EXPR])
AC_SUBST([GREP])
AC_SUBST([LD])
AC_SUBST([MAKE])
AC_SUBST([SED])
AC_SUBST([TR])

dnl Allow for output of the command-line switches initialised by the
dnl QM_LOCATE_KEY_TOOLS macro. 
AC_SUBST([GET_CWD])
AC_SUBST([with_rpath])


# Now record the correct switches for the individual tools in use
QM_SET_COMPILER_CFLAGS
QM_SET_COMPILER_CCFLAGS
QM_SET_LINKER_LDFLAGS
QM_SET_ARCHIVER_ARFLAGS
QM_SET_MAKE_MAKEFLAGS




# Check for remaining utils

# Check for programs needed by QMake
# First check for those programs that GNU coding standards regard as safe to use
# in a makefile if they are employed via make variables (and thus create the
# necessary variables)
AC_PATH_PROG([CHGRP],[chgrp],[missing])
AS_IF([test $CHGRP = missing],
      [AC_MSG_FAILURE([Cannot find a working chgrp required for using QMake])]
     )
AC_SUBST([CHGRP])

AC_PATH_PROG([CHMOD],[chmod],[missing])
AS_IF([test $CHMOD = missing],
      [AC_MSG_FAILURE([Cannot find a working chmod required for using QMake])]
     )
AC_SUBST([CHMOD])

AC_PATH_PROG([CHOWN],[chown],[missing])
AS_IF([test $CHOWN = missing],
      [AC_MSG_FAILURE([Cannot find a working chown required for using QMake])]
     )
AC_SUBST([CHOWN])

# Check for install-sh as it is needed directly by QMake and it also may be
# needed for mkdir -p support
AC_PROG_INSTALL

AC_PROG_LEX
AC_SUBST([LEX])
AC_SUBST([LEXLIB])

# Ensure we have a mkdir -p capability (will use install-sh on systems without
# a native mkdir -p)
AC_PROG_MKDIR_P
AC_SUBST([MKDIR_P])

# Ranlib program (if necessary)
AC_PROG_RANLIB
AC_SUBST([RANLIB])
# Slight weakness of this check is that if ranlib is not found, we assume it is
# not needed (set RANLIB to ":")

AC_PROG_YACC
AC_SUBST([YACC])

# GNU standards regard the following safe to call directly but as we already
# implemented them via a make variable, set those here...
AC_PATH_PROG([CP],[cp],[missing])
AS_IF([test $CP = missing],
      [AC_MSG_FAILURE([Cannot find a working cp required for using QMake])]
     )
AC_SUBST([CP])

AC_PATH_PROG([RM],[rm],[missing])
AS_IF([test $RM = missing],
      [AC_MSG_FAILURE([Cannot find a working rm required for using QMake])]
     )
AC_SUBST([RM])

AC_PATH_PROG([TOUCH],[touch],[missing])
AS_IF([test $TOUCH = missing],
      [AC_MSG_FAILURE([Cannot find a working touch required for using QMake])]
     )
AC_SUBST([TOUCH])



# Now check for those utilities we use in the build process that aren't recognised
# as being reliably portable by the GNU standard. For these, we will need to
# consider providing an alternative implementation in cases where any are absent.
AC_PROG_LN_S
AC_SUBST([LN_S])

AS_IF([test x$PAGER = x],
      [AC_PATH_PROG([QMAKE_PAGER],[more],[missing])
       AS_IF([test $QMAKE_PAGER = missing],
             [AC_PATH_PROG([QMAKE_PAGER],[pg],[missing])
              AS_IF([test $QMAKE_PAGER = missing],
                    [AC_MSG_FAILURE([Cannot find a working pager required for using QMake])]
                   )
             ]
            )
      ],
      [QMAKE_PAGER=$PAGER]
     )
AC_SUBST([QMAKE_PAGER])

AC_PATH_PROG([PERL],[perl],[missing])
AS_IF([test $PERL = missing], 
      [AC_MSG_FAILURE([Cannot find a working copy of Perl])]
     )
AC_SUBST([PERL])

AC_PATH_PROG([SYNC],[sync],[missing])
AS_IF([test $SYNC = missing],
      [AC_MSG_FAILURE([Cannot find a working sync required for using QMake])]
     )
AC_SUBST([SYNC])

AC_PATH_PROG([XARGS],[xargs],[missing])
AS_IF([test $XARGS = missing],
      [AC_MSG_FAILURE([Cannot find a working xargs required for using QMake])]
     )
AC_SUBST([XARGS])

# Remaining utils used in the build process are all regarded by the GNU
# standards as safe for direct use:
#       cat cmp diff ls mv rmdir tar test touch true



#
# Create proforma Config.QMake.in
#

# Create initial, empty file
echo > ./Config.QMake.in

echo \# QMake configuration file - regenerate using configure >> ./Config.QMake.in
echo >> ./Config.QMake.in

# Define location of compilers and helper programs for build system
echo >> ./Config.QMake.in
echo \# List of programs required for the build process >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo AR=@AR@ >> ./Config.QMake.in
echo AWK=@AWK@ >> ./Config.QMake.in
echo CC=@CC@ >> ./Config.QMake.in
echo CP=@CP@ >> ./Config.QMake.in
echo CPP=@CPP@ >> ./Config.QMake.in
echo CHGRP=@CHGRP@ >> ./Config.QMake.in
echo CHMOD=@CHMOD@ >> ./Config.QMake.in
echo CHOWN=@CHOWN@ >> ./Config.QMake.in
echo CXX=@CXX@ >> ./Config.QMake.in
echo CXXCPP=@CXXCPP@ >> ./Config.QMake.in
echo EXPR=@EXPR@ >> ./Config.QMake.in
echo GET_CWD=@GET_CWD@ >> ./Config.QMake.in
echo GREP=@GREP@ >> ./Config.QMake.in
echo INSTALL=@INSTALL@ >> ./Config.QMake.in
echo INSTALL_DATA=@INSTALL_DATA@ >> ./Config.QMake.in
echo INSTALL_PROGRAM=@INSTALL_PROGRAM@ >> ./Config.QMake.in
echo LD=@LD@ >> ./Config.QMake.in
echo LEX=@LEX@ >> ./Config.QMake.in
echo LEXLIB=@LEXLIB@ >> ./Config.QMake.in
echo LN_S=@LN_S@ >> ./Config.QMake.in
echo MAKE=@MAKE@ >> ./Config.QMake.in
echo MKDIR_P=@MKDIR_P@ >> ./Config.QMake.in
echo PERL=@PERL@ >> ./Config.QMake.in
echo QMAKE_PAGER=@QMAKE_PAGER@ >> ./Config.QMake.in
echo RANLIB=@RANLIB@ >> ./Config.QMake.in
echo RM=@RM@ >> ./Config.QMake.in
echo SED=@SED@ >> ./Config.QMake.in
echo SORT=@SORT@ >> ./Config.QMake.in
echo SYNC=@SYNC@ >> ./Config.QMake.in
echo TOUCH=@TOUCH@ >> ./Config.QMake.in
echo TR=@TR@ >> ./Config.QMake.in
echo XARGS=@XARGS@ >> ./Config.QMake.in
echo YACC=@YACC@ >> ./Config.QMake.in
echo >> ./Config.QMake.in


# Compiler, etc. switches
echo >> ./Config.QMake.in
echo \# Relevant QMake build process flags >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Declare platform architecture >> ./Config.QMake.in
echo ARC=@ARC@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the C compiler >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo QMAKE_CFLAGS=@CONFIGURE_CFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# QMake compiler settings >> ./Config.QMake.in
echo COMPILER_ANSI_CFLAGS=@COMPILER_ANSI_CFLAGS@ >> ./Config.QMake.in
echo COMPILER_DEBUG_CFLAGS=@COMPILER_DEBUG_CFLAGS@ >> ./Config.QMake.in
echo COMPILER_GPROF_CFLAGS=@COMPILER_GPROF_CFLAGS@ >> ./Config.QMake.in
echo COMPILER_PROF_CFLAGS=@COMPILER_PROF_CFLAGS@ >> ./Config.QMake.in
echo COMPILER_SHARED_CFLAGS=@COMPILER_SHARED_CFLAGS@ >> ./Config.QMake.in
echo COMPILER_SPECIFIC_CFLAGS=@COMPILER_SPECIFIC_CFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the C++ compiler >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo QMAKE_CCFLAGS=@CONFIGURE_CCFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# QMake compiler settings >> ./Config.QMake.in
echo COMPILER_ANSI_CCFLAGS=@COMPILER_ANSI_CCFLAGS@ >> ./Config.QMake.in
echo COMPILER_DEBUG_CCFLAGS=@COMPILER_DEBUG_CCFLAGS@ >> ./Config.QMake.in
echo COMPILER_GPROF_CCFLAGS=@COMPILER_GPROF_CCFLAGS@ >> ./Config.QMake.in
echo COMPILER_PROF_CCFLAGS=@COMPILER_PROF_CCFLAGS@ >> ./Config.QMake.in
echo COMPILER_SHARED_CCFLAGS=@COMPILER_SHARED_CCFLAGS@ >> ./Config.QMake.in
echo COMPILER_SPECIFIC_CCFLAGS=@COMPILER_SPECIFIC_CCFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the linker >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo QMAKE_LDFLAGS=@LDFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# General QMake linker settings >> ./Config.QMake.in
echo LINKER_SPECIFIC_LDFLAGS=@LINKER_SPECIFIC_LDFLAGS@ >> ./Config.QMake.in
echo LINKER_SHARED_LDFLAGS=@LINKER_SHARED_LDFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Linker settings for QMake libraries >> ./Config.QMake.in
echo LINKER_SPECIFIC_LDLIBFLAGS=@LINKER_SPECIFIC_LDLIBFLAGS@ >> ./Config.QMake.in
echo LINKER_SHARED_LDLIBFLAGS=@LINKER_SHARED_LDLIBFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Specific linker functionality >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo LINKER_LIBPATHSWITCH=@LINKER_LIBPATHSWITCH@ >> ./Config.QMake.in
echo USE_LIBPATHSWITCH=@with_rpath@ >> ./Config.QMake.in
echo LINKER_UNDEFSYMB=@LINKER_UNDEFSYMB@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the archiver >> ./Config.QMake.in
echo \# >> ./Config.QMake.in
echo \# No user supplied ARFLAGS are accepted owing to the nature of how ar works >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo AR_SPECIFIC_ARFLAGS=@AR_SPECIFIC_ARFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for make >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo QMAKE_MAKEFLAGS=@MAKEFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# General QMake make settings >> ./Config.QMake.in
echo MAKE_SPECIFIC_MAKEFLAGS=@MAKE_SPECIFIC_MAKEFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in



QM_SET_ARC
AC_SUBST([ARC])



dnl Create Makefiles and Config.QMake
AC_CONFIG_FILES([Config.QMake])
AC_CONFIG_FILES([Makefile:src/AutoTools/Makefile.in])
AC_CONFIG_FILES([Targets.QMake:src/AutoTools/Targets.QMake.in])
AC_OUTPUT
