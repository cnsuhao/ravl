#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.65])

AC_INIT([QMake], [1.5.8030], [John.Field@Surrey.ac.UK])

# This file is part of QMake, Quick Make System
# Copyright (C) 2011, University of Surrey
# This code may be redistributed under the terms of the GNU General
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html
# file-header-ends-here
#! file="Ravl/QMake/Main.mk"

dnl#JFi:ToDo-Check for ver 5 of Perl as a requirement


#     Define shell functions

# Trace debug function
#
# Set DBG=<level> in the environment for extra output messages. Arguments to
# the function are the 'level' of the trace message, and the message itself
# (which may be provided over multiple arguments). Current values of level
# are: 
#	1 Assertions, 
#	2 Progress tracing, 
#	3 Moderate data reporting
#       :
#       6 High data reporting
#
# If an indentation is required (say, for iterated code), set TRACE_INDENT to
# the indentation as you want if to appear on the screen (i.e. a number of
# spaces, or -->, etc.). Current scheme is to use 5 spaces per level of indent
trace ()
{
   if ( test X$DBG != X )
   then
      level=$1
      shift
      if ( test $DBG -ge $level )
      then
         if ( test X"$TRACE_INDENT" != X )
         then
            echo "$TRACE_INDENT" "$*"
         else
            echo "$*"
         fi
      fi
   fi
   TRACE_INDENT=""
}

#     End of shell functions


# Main body of configure script

# First determine the current platform
AC_CANONICAL_HOST

# Check for required programs.
trace 2 Checking for support programs

# First check for utils required for this script itself. However, it is probably
# too late to check for a lot of commands as the preceding code has already used
# them (test, print, etc.) but we can check for what we're about to use...
AC_PROG_AWK
AC_SUBST([AWK])

AC_PATH_PROG([EXPR],[expr],[Missing])
AS_IF([test $EXPR = Missing],
      [AC_MSG_FAILURE([Cannot find a working expr - configure cannot continue])]
     )
AC_SUBST([EXPR])

AC_PROG_GREP
AC_SUBST([GREP])

AC_PROG_SED
AC_SUBST([SED])

# Check for C Compiler and Pre-processor
#
# Record CFLAGS in case user has supplied them on the configure cmd-line;
# these will form CONFIGURE_CFLAGS setting in QMake build system. 
CONFIGURE_CFLAGS=$CFLAGS
AC_SUBST([CONFIGURE_CFLAGS])
# Now actually check for compiler and and allow Config.QMake to record the
# location found
AC_PROG_CC
AC_SUBST([CC])
# AC_PROG_CC won't actually have altered CFLAGS if a value was already set
# however, if an initial value was not provided, AC_PROG_CC may have set CFLAGS
# to include optimisation or -g. As we do not want this for a QMake build, we
# reinitialise CFLAGS to the original value (which may be null or what a user
# supplied on the command line).
CFLAGS=$CONFIGURE_CFLAGS
# Find and record C pre-processor
AC_PROG_CPP
AC_SUBST([CPP])

# Check for C++ Compiler and Pre-processor
#
# Allow recording of any user-supplied CCFLAGS in Config.QMake
CONFIGURE_CCFLAGS=$CCFLAGS
AC_SUBST([CONFIGURE_CCFLAGS])
# Now check for compiler and allow that result to be recorded
AC_PROG_CXX
AC_SUBST([CXX])
# Prevent optimisation or -g being to CCFLAGS if it was initially unset
CCFLAGS=$CONFIGURE_CCFLAGS
# Find and record C++ pre-processor
AC_PROG_CXXCPP
AC_SUBST([CXXCPP])

# Look for linker
#
# This code inspired by LT_PATH_LD from libtool

if ( test X$LD = X )
then
  # User has not asked for a specific linker, we have to look for one

  if ( test "$GCC" = yes )
  then
    # If we have gcc we can try: $CC -print-prog-name=ld
    AC_MSG_CHECKING([which ld is used by $CC])
    LD=`$CC -print-prog-name=ld`
    # Test host_os != mingw*
    if ( test  X`expr X$host_os : 'X\(mingw\).*'` != 'Xmingw' )
    then
      # gcc leaves a trailing carriage return which upsets mingw
      LD=`echo $LD | tr -d '\015'`
    fi
    
    if ( test X`expr X$LD : '\(X[[\\/]].*\)'` = XX$LD )
    then
      # $LD is an absolute path
      # Fold all \ to /
      LD=`$echo "$LD"| $SED 's%\\\\%/%g'`
      # $LD may be an absolute path but might still need cannonicalisation
      while test X`expr X$LD : 'X\(.*/\.\./\.*\)'` != X
      do
        LD=`echo $LD | $SED -e "s%/*[[^/]][[^/]]*/\.\./%/%g" -e "s%^/*\.\./%/%"`
      done
      AC_MSG_RESULT([$LD])
    else
      # $CC -print-prog-name=ld has not delivered us an absolute path
      # We need to use the searchlist
      LD=""
      AC_MSG_RESULT([unhelpful])
    fi
  else
    # Not using GCC so we don't know how to ask the compiler, we will have to
    # look for ld on the searchlist
    LD=""
  fi

  if ( test X`expr X$LD : '\(X[[\\/]].*\)'` != XX$LD )
  then
    # LD not set to an absolute path, must search for ld

    AC_MSG_CHECKING([for ld on the search-list])
    GNU_LD=""
    ANO_LD=""
    save_ifs="$IFS"
    IFS=$PATH_SEPARATOR
    for dir in $PATH
    do
      IFS="$save_ifs"
      if ( test X"$dir" != X && test $dir != '.' && test -x "$dir/ld" )
      then
        # Some variants of GNU ld only accept -v, so use that not --version
        case `"$dir/ld" -v 2>&1 </dev/null` in
        *GNU* | *'with BFD'*)   if ( test  X`expr X$host_os : 'X\(aix\).*'` != 'Xaix' )
                                then
                                  # We are not using AIX so favour the GNU linker
                                  # (that we have just found)
                                  LD=$dir/ld
                                  break
                                else
                                  # We are on AIX so should use the native
                                  # linker if possible but remember where the
                                  # GNU version is should that be all we find
                                  if ( test X$GNU_LD = X )
                                  then
                                    # This is the first GNU version of ld on the
                                    # PATH so record it
                                    GNU_LD=$dir/ld
                                  fi
                                fi
                                ;;
                           *)   if ( test  X`expr X$host_os : 'X\(aix\).*'` != 'Xaix' )
                                then
                                  # We are not on AIX so should favour the GUN
                                  # version of ld. As this is not a GNU version,
                                  # merely remember where the first non-GNU
                                  # version is should this be the only one we
                                  # actually find.
                                  if ( test X$ANO_LD = X )
                                  then
                                    ANO_LD=$dir/ld
                                  fi
                                else
                                  # As we are on AIX and have found the non-GNU
                                  # linker, we should favour that.
                                  LD=$dir/ld
                                fi
                                ;;
        esac
      fi
    done
    IFS="$save_ifs"
    if ( test X$LD = X )
    then
      # We haven't located our preferred linker, is there a fall-back?
      if ( test  X`expr X$host_os : 'X\(aix\).*'` != 'Xaix' )
      then
        # We are not using AIX but found no GNU linker, use any other we found
        LD=$ANO_LD
      else
        # We are on AIX and wanted the native linker which we did not locate.
        # No matter, try any GNU version we found
        LD=$GNU_LD
      fi
    fi
  fi # End of checking for ld on $PATH (LD not already an absolute pathname)

  # Have we actually found a linker?
  if ( test X$LD != X && test -x $LD )
  then
    # We have, what type is it?
    VER_STRING=`"$LD" -v 2>&1 </dev/null`
    if ( test `expr "$VER_STRING" : '.*GNU.*'` -gt 0 || test `expr "$VER_STRING" : '.*with BFD.*'` -gt 0 )
    then
      # GNU linker
      GNU_LD=yes
    else
      GNU_LD=no
    fi
    AC_MSG_RESULT([$LD])
  else
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Cannot find a working ld required for using QMake])
  fi
else
  # User specified LD
  if ( test -x $LD )
  then
    # What type is it?
    VER_STRING=`"$LD" -v 2>&1 </dev/null`
    if ( test `expr "$VER_STRING" : '.*GNU.*'` -gt 0 || test `expr "$VER_STRING" : '.*with BFD.*'` -gt 0 )
    then
      # GNU linker
      GNU_LD=yes
    else
      GNU_LD=no
    fi
  else
    AC_MSG_FAILURE([$LD does not appear to correctly specify a linker])
  fi
fi

AC_SUBST([LD])



# Look for archive tools

AC_PATH_PROG([AR],[ar],[Missing])
AS_IF([test $AR = Missing],
      [AC_MSG_FAILURE([Cannot find a working ar required for using QMake])]
     )
AC_SUBST([AR])


AC_PROG_RANLIB
AC_SUBST([RANLIB])
# Slight weakness of this check is that if ranlib is not found, we assume it is
# not needed (set RANLIB to ":")



# Look for the make utility

if ( test X$MAKE = X )
then
  # User has not asked for a specific make utility, look for make
  MAKE=make
fi

if ( test X`expr X$MAKE : '\(X[[\\/]].*\)'` != XX$MAKE )
then
  # User has not asked for a specific path for the make utility, we have to look
  # for it on the search list

  AC_MSG_CHECKING([for make utility])
  GNU_MAKE=""
  ANO_MAKE=""
  save_ifs="$IFS"
  IFS=$PATH_SEPARATOR
  for dir in $PATH
  do
    IFS="$save_ifs"
    if ( test X"$dir" != X && test $dir != '.' && test -x "$dir/$MAKE" )
    then
      # Run $MAKE -v to check if we have GNU Make
      case `"$dir/$MAKE" -v 2>&1 </dev/null` in
        *GNU* | *'with BFD'*)   GNU_MAKE=$dir/$MAKE
                                # Found GNU Make which we prefer so we need
                                # look no further
                                break
                                ;;
                           *)   if ( test X$ANO_MAKE = X )
                                then
                                  ANO_MAKE=$dir/$MAKE
                                fi
                                ;;
        esac
    fi
  done
  IFS="$save_ifs"
  if ( test X$GNU_MAKE = X )
  then
    # Preferred GNU Make not found, was there an alternative
    if ( test  X$ANO_MAKE = X)
    then
      AC_MSG_RESULT([no])
      AC_MSG_FAILURE([Cannot find a working make required for using QMake])
    else
      MAKE=$ANO_MAKE
      AC_MSG_RESULT([$MAKE])
      GNU_MAKE=no
    fi
  else
    MAKE=$GNU_MAKE
    AC_MSG_RESULT([$MAKE])
    GNU_MAKE=yes
  fi
else
  # User specified a full path to make (in MAKE environment variable)
  AC_MSG_CHECKING([$$MAKE specified make utility])
  if ( test -x $MAKE )
  then
    VER_STRING=`"$MAKE" -v 2>&1 </dev/null`
    if ( test `expr "$VER_STRING" : '.*GNU.*'` -gt 0 || test `expr "$VER_STRING" : '.*with BFD.*'` -gt 0 )
    then
      GNU_MAKE=yes
    else
      GNU_MAKE=no
    fi
    AC_MSG_RESULT([$MAKE])
  else
    AC_MSG_RESULT([unexecutable])
    AC_MSG_FAILURE([$$MAKE does not specify a usable make utility])
  fi
fi

AC_SUBST([MAKE])

# Set relevant flags dependant on the software tools we have found
#
dnl Most options will be package specific so will be set by the configure script
dnl for the individual package, Only set options that are required for QMake to
dnl operate correctly here; and even then, only select flags dependant on the
dnl actual tools in use. Set all options pertaining to platform specifics later 
dnl in this script
dnl

# Set CFLAGS
if (test X$GCC = Xyes)
then
   # GNU C Compiler
   QMAKE_GLOBAL_CFLAGS=""
   QMAKE_SHARED_CFLAGS=" -shared -fPIC "
   QMAKE_PROF_CFLAGS=" -p "
   QMAKE_GPROF_CFLAGS=" -pg "
else
   # No GNU compiler, try relevant machine specific options we have used before
dnl 
dnl Although testing the host OS, we are using that to identify the native tool
dnl set that may be in use. These are not platform specific settings per se.
dnl DO NOT PUT PLATFORM SPECIFIC SETTINGS HERE. This section is for settings
dnl that relate to specific tool sets. They may or may not relate to specific 
dnl platforms but that is coincidental. Platform specific settings must go in
dnl the section at the end of this script. If you have a specific compiler on
dnl a platform, put its settings here. If you have specific settings for a
dnl common tool (say the GNU compilers) when run on a particular platform, put
dnl those settings at the end of this script.
dnl 
   case X$host_os in
   Xsolaris* ) QMAKE_GLOBAL_CFLAGS=""
               QMAKE_SHARED_CFLAGS=" -KPIC "
               QMAKE_PROF_CFLAGS=" -p "
               QMAKE_GPROF_CFLAGS=" -pg "
               ;;
   Xirix* )    CC=$CC" -n32 -J4 -mp "
               # Not sure why this originally was done this way rather than use
               # CFLAGS, but assume there was a reason and propogate the method
               QMAKE_GLOBAL_CFLAGS=" -mips4 -r1000 "
               QMAKE_SHARED_CFLAGS=""
               QMAKE_PROF_CFLAGS=""
               QMAKE_GPROF_CFLAGS=""
               ;;
   * )         QMAKE_GLOBAL_CFLAGS=""
               QMAKE_SHARED_CFLAGS=""
               QMAKE_PROF_CFLAGS=""
               QMAKE_GPROF_CFLAGS=""
               ;;
   esac
fi

# Declare QMAKE_XXXXX_CFLAGS variables as output for Config.QMake
AC_SUBST([QMAKE_GLOBAL_CFLAGS])
AC_SUBST([QMAKE_SHARED_CFLAGS])
AC_SUBST([QMAKE_PROF_CFLAGS])
AC_SUBST([QMAKE_GPROF_CFLAGS])

# Set CCFLAGS
if (test X$GXX = Xyes)
then
   # GNU C++ Compiler
   QMAKE_GLOBAL_CCFLAGS=""
   QMAKE_SHARED_CCFLAGS=" -shared -fPIC "
   QMAKE_PROF_CCFLAGS=" -p "
   QMAKE_GPROF_CCFLAGS=" -pg "
else
   # No GNU compiler, try relevant machine specific options we have used before
dnl 
dnl Although testing the host OS, we are using that to identify the native tool
dnl set that may be in use. These are not platform specific settings per se.
dnl DO NOT PUT PLATFORM SPECIFIC SETTINGS HERE. This section is for settings
dnl that relate to specific tool sets. They may or may not relate to specific 
dnl platforms but that is coincidental. Platform specific settings must go in
dnl the section at the end of this script. If you have a specific compiler on
dnl a platform, put its settings here. If you have specific settings for a
dnl common tool (say the GNU compilers) when run on a particular platform, put
dnl those settings at the end of this script.
dnl 
   case X$host_os in
   Xsolaris* ) QMAKE_GLOBAL_CCFLAGS=""
               QMAKE_SHARED_CCFLAGS=" -KPIC "
               QMAKE_PROF_CCFLAGS=" -p "
               QMAKE_GPROF_CCFLAGS=" -pg "
               ;;
   Xirix* )    CXX=$CXX" -n32 -J4 -mp -LANG:ansi-for-init-scope=on "
               # Not sure why this originally was done this way rather than use
               # CCFLAGS, but assume there was a reason and propogate the method
               QMAKE_GLOBAL_CCFLAGS=" -mips4 -r1000 -woff 1021 -woff 1681 -woff 1882 "
               QMAKE_SHARED_CCFLAGS=""
               QMAKE_PROF_CCFLAGS=""
               QMAKE_GPROF_CCFLAGS=""
               ;;
   * )         QMAKE_GLOBAL_CCFLAGS=""
               QMAKE_SHARED_CCFLAGS=""
               QMAKE_PROF_CCFLAGS=""
               QMAKE_GPROF_CCFLAGS=""
               ;;
   esac
fi

# Declare QMAKE_XXXXX_CCFLAGS variables as output for Config.QMake
AC_SUBST([QMAKE_GLOBAL_CCFLAGS])
AC_SUBST([QMAKE_SHARED_CCFLAGS])
AC_SUBST([QMAKE_PROF_CCFLAGS])
AC_SUBST([QMAKE_GPROF_CCFLAGS])


# Set QMAKE_XXXXX_LDFLAGS; LDLIBFLAGS and other linker settings
if ( test X$GNU_LD = Xyes )
then
  # We are using a GNU linker

  QMAKE_GLOBAL_LDFLAGS=
  QMAKE_SHARED_LDFLAGS=" -rdynamic "

  QMAKE_GLOBAL_LDLIBFLAGS=
  QMAKE_SHARED_LDLIBFLAGS=" -shared "

  LIBPATHSWITCH=" -Wl,-rpath#"
  UNDEFSYMB=" -u#"
else
  # Not using a GNU linker, try native flags we have used before or default to
  # what works for GNU
dnl 
dnl Although testing the host OS, we are using that to identify the native tool
dnl set that may be in use. These are not platform specific settings per se.
dnl DO NOT PUT PLATFORM SPECIFIC SETTINGS HERE. This section is for settings
dnl that relate to specific tool sets. They may or may not relate to specific 
dnl platforms but that is coincidental. Platform specific settings must go in
dnl the section at the end of this script. If you have a specific compiler on
dnl a platform, put its settings here. If you have specific settings for a
dnl common tool (say the GNU compilers) when run on a particular platform, put
dnl those settings at the end of this script.
dnl 
  case X$host_os in
  Xcygwin )   QMAKE_GLOBAL_LDFLAGS=
              QMAKE_SHARED_LDFLAGS=" -Wl,-export-dynamic "
              # sic, this is an assumption on what compiler is in use

              QMAKE_GLOBAL_LDLIBFLAGS=
              QMAKE_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              # sic, this is an assumption on what compiler is in use
              UNDEFSYMB=" -u #"
              ;;
  Xirix*    ) QMAKE_GLOBAL_LDFLAGS=" -n32 -Wl,-woff 84 "
              QMAKE_SHARED_LDFLAGS=

              QMAKE_GLOBAL_LDLIBFLAGS=
              QMAKE_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              UNDEFSYMB=" -u#"
              ;;
  Xsolaris* ) QMAKE_GLOBAL_LDFLAGS=
              QMAKE_SHARED_LDFLAGS=

              QMAKE_GLOBAL_LDLIBFLAGS=
              QMAKE_SHARED_LDLIBFLAGS=" -dy -G "

              LIBPATHSWITCH=" -R#"
              UNDEFSYMB=" -u#"
              ;;
  * )         QMAKE_GLOBAL_LDFLAGS=
              QMAKE_SHARED_LDFLAGS=" -rdynamic "

              QMAKE_GLOBAL_LDLIBFLAGS=
              QMAKE_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              UNDEFSYMB=" -u#"
              ;;
  esac
fi

# Declare variables that the linker as output for Config.QMake
AC_SUBST([LDFLAGS])
# any user-provided LDFLAGS will be recorded in Config.QMake as CONFIGURE_LDFLAGS

AC_SUBST([QMAKE_GLOBAL_LDFLAGS])
AC_SUBST([QMAKE_SHARED_LDFLAGS])

AC_SUBST([QMAKE_GLOBAL_LDLIBFLAGS])
AC_SUBST([QMAKE_SHARED_LDLIBFLAGS])

AC_SUBST([LIBPATHSWITCH])
AC_SUBST([UNDEFSYMB])


# Declare flags for ar

dnl DO NOT PUT PLATFORM SPECIFICS HERE. If your platform has a non-standard ar
dnl that takes completely different flags then that is ok to list here (as it
dnl pertains to the tool itself, not the platform). If your platform uses the
dnl GNU version of ar but, for example requires an added 's' flag, that setting
dnl should be put in the platform specific code at the end of this script, not
dnl here.
QMAKE_ARFLAGS=ruc

AC_SUBST([QMAKE_ARFLAGS])

# Declare flags for make

dnl Only put make flags specific to any particular version of make here. DO NOT
dnl PUT PLATFORM SPECIFIC FLAGS HERE; they go in the final section of this
dnl script

if ( test X$GNU_MAKE = Xyes )
then
  QMAKE_MAKEFLAGS=" --no-print-directory -r "
else
  QMAKE_MAKEFLAGS=" -r "
fi

AC_SUBST([MAKEFLAGS])
# User-provided MAKEFLAGS will be recorded as CONFIGURE_MAKEFLAGS in Config.QMake 
AC_SUBST([QMAKE_MAKEFLAGS])

dnl End of software tools section


dnl# It would be good to enable the following code, however _AC_... aren't 
dnl# generally available macros and may dissappear/change in differing versions
dnl# of autotools so its best not to rely on them. If only there was a supported
dnl# means of doing this.. (you can check for the preprocessor (as we do, above)
dnl# but not all of the calls that _AC_INCLUDES_DEFAULT_REQUIREMENTS expands to
dnl# are individually available.
dnl#
dnl#      Cause the checks for the C pre-processor and default headers to be done now
dnl#      (This can prevent multiple calls within the script's body)
dnl       _AC_INCLUDES_DEFAULT_REQUIREMENTS


# Check for install-sh as it may be needed for mkdir -p support
AC_PROG_INSTALL

# Ensure we have a mkdir -p capability (will use install-sh on systems without
# a native mkdir -p)
AC_PROG_MKDIR_P
AC_SUBST([MKDIR_P])

# Check for remaining utils

# Check for pkg-config
PKG_PROG_PKG_CONFIG

# Check for programs needed by QMake
# First check for those programs that GNU coding standards regard as safe to use
# in a makefile if they are employed via make variables (and thus create the
# necessary variables)
AC_PATH_PROG([CHGRP],[chgrp],[Missing])
AS_IF([test $CHGRP = Missing],
      [AC_MSG_FAILURE([Cannot find a working chgrp required for using QMake])]
     )
AC_SUBST([CHGRP])

AC_PATH_PROG([CHMOD],[chmod],[Missing])
AS_IF([test $CHMOD = Missing],
      [AC_MSG_FAILURE([Cannot find a working chmod required for using QMake])]
     )
AC_SUBST([CHMOD])

AC_PATH_PROG([CHOWN],[chown],[Missing])
AS_IF([test $CHOWN = Missing],
      [AC_MSG_FAILURE([Cannot find a working chown required for using QMake])]
     )
AC_SUBST([CHOWN])

AC_PROG_LEX
AC_SUBST([LEX])
AC_SUBST([LEXLIB])

AC_PATH_PROG([SORT],[sort],[Missing])
AS_IF([test $SORT = Missing],
      [AC_MSG_FAILURE([Cannot find a working sort required for using QMake])]
     )
AC_SUBST([SORT])

AC_PROG_YACC
AC_SUBST([YACC])

# GNU standards regard the following safe to call directly but as we already
# implemented them via a make variable, set those here...
AC_PATH_PROG([CP],[cp],[Missing])
AS_IF([test $CP = Missing],
      [AC_MSG_FAILURE([Cannot find a working cp required for using QMake])]
     )
AC_SUBST([CP])

AC_PATH_PROG([RM],[rm],[Missing])
AS_IF([test $RM = Missing],
      [AC_MSG_FAILURE([Cannot find a working rm required for using QMake])]
     )
AC_SUBST([RM])

AC_PATH_PROG([TOUCH],[touch],[Missing])
AS_IF([test $TOUCH = Missing],
      [AC_MSG_FAILURE([Cannot find a working touch required for using QMake])]
     )
AC_SUBST([TOUCH])

AC_PATH_PROG([TR],[tr],[Missing])
AS_IF([test $TR = Missing],
      [AC_MSG_FAILURE([Cannot find a working tr required for using QMake])]
     )
AC_SUBST([TR])

# Provide switch to allow use of pawd instead of pwd
AC_ARG_WITH([pawd],
            [AS_HELP_STRING([--with-pawd],
                            [Employ pawd rather than pwd to return the current directory @<:@default=No@:>@]
                           )
            ],
            [case $with_pawd in
                        'n' | 'N' | 'no' | 'No' | 'NO' ) 
                               trace 4 --with-pawd disabling use of pawd
                               AC_PATH_PROG([GET_CWD],[pwd],[Missing])
                               AS_IF([test $GET_CWD = Missing],
                                     [AC_MSG_FAILURE([Cannot find a working pwd required for using QMake])]
                                    )
                               ;;
                        'y' |'Y' | 'yes' | 'Yes' | 'YES' ) 
                               trace 4 --with-pawd forcing use of pawd
                               AC_PATH_PROG([GET_CWD],[pawd],[Missing])
                               AS_IF([test $GET_CWD = Missing],
                                     [AC_MSG_FAILURE([Asking to enable pawd via --with-pawd but it cannot be found])]
                                    )
                               ;;
                        'c' | 'C' | 'ck'| 'Ck' | 'CK' | 'check' | 'Check' | 'CHECK' )
                               trace 4 --with-pawd asking to use pawd if available
                               AC_PATH_PROG([GET_CWD],[pawd],[Missing])
                               AS_IF([test $GET_CWD = Missing],
                                     [AC_PATH_PROG([GET_CWD],[pwd],[Missing])
                                      AS_IF([test $GET_CWD = Missing],
                                            [AC_MSG_FAILURE([Cannot find a working pawd or pwd required for using QMake])]
                                           )
                                     ]
                                    )
                               ;;
                        * )    AS_IF([test -x $with_pawd],
                                     [trace 4 --with-pawd asking to use "@<:@$with_pawd@:>@"
                                      GET_CWD=$with_pawd
                                     ],
                                     [AC_MSG_WARN([--with-pawd set to an unknown option, setting ignored])
                                      AC_PATH_PROG([GET_CWD],[pwd],[Missing])
                                      AS_IF([test $GET_CWD = Missing],
                                            [AC_MSG_FAILURE([Cannot find a working pwd required for using QMake])]
                                           )
                                     ]
                                    )
                               ;;
                       esac
            ],
            [trace 5 No --with-$1 specified
             AC_PATH_PROG([GET_CWD],[pwd],[Missing])
             AS_IF([test $GET_CWD = Missing],
                   [AC_MSG_FAILURE([Cannot find a working pwd required for using QMake])]
                  )
            ]
           )
AC_SUBST([GET_CWD])

# Now check for those utilities we use in the build process that aren't recognised
# as being reliably portable by the GNU standard. For these, we will need to
# consider providing an alternative implementation in cases where any are absent.
AC_PROG_LN_S
AC_SUBST([LN_S])

AS_IF([test x$PAGER = x],
      [AC_PATH_PROG([QMAKE_PAGER],[more],[Missing])
       AS_IF([test $QMAKE_PAGER = Missing],
             [AC_PATH_PROG([QMAKE_PAGER],[pg],[Missing])
              AS_IF([test $QMAKE_PAGER = Missing],
                    [AC_MSG_FAILURE([Cannot find a working pager required for using QMake])]
                   )
             ]
            )
      ],
      [QMAKE_PAGER=$PAGER]
     )
AC_SUBST([QMAKE_PAGER])

AC_PATH_PROG([PERL],[perl],[Missing])
AS_IF([test $PERL = Missing], 
      [AC_MSG_FAILURE([Cannot find a working copy of Perl])]
     )
AC_SUBST([PERL])

AC_PATH_PROG([SYNC],[sync],[Missing])
AS_IF([test $SYNC = Missing],
      [AC_MSG_FAILURE([Cannot find a working sync required for using QMake])]
     )
AC_SUBST([SYNC])

AC_PATH_PROG([XARGS],[xargs],[Missing])
AS_IF([test $XARGS = Missing],
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
echo CONFIGURE_CFLAGS=@CONFIGURE_CFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# QMake compiler settings >> ./Config.QMake.in
echo QMAKE_GLOBAL_CFLAGS=@QMAKE_GLOBAL_CFLAGS@ >> ./Config.QMake.in
echo QMAKE_GPROF_CFLAGS=@QMAKE_GPROF_CFLAGS@ >> ./Config.QMake.in
echo QMAKE_PROF_CFLAGS=@QMAKE_PROF_CFLAGS@ >> ./Config.QMake.in
echo QMAKE_SHARED_CFLAGS=@QMAKE_SHARED_CFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the C++ compiler >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo CONFIGURE_CCFLAGS=@CONFIGURE_CCFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# QMake compiler settings >> ./Config.QMake.in
echo QMAKE_GLOBAL_CCFLAGS=@QMAKE_GLOBAL_CCFLAGS@ >> ./Config.QMake.in
echo QMAKE_GPROF_CCFLAGS=@QMAKE_GPROF_CCFLAGS@ >> ./Config.QMake.in
echo QMAKE_PROF_CCFLAGS=@QMAKE_PROF_CCFLAGS@ >> ./Config.QMake.in
echo QMAKE_SHARED_CCFLAGS=@QMAKE_SHARED_CCFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the linker >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo CONFIGURE_LDFLAGS=@LDFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# General QMake linker settings >> ./Config.QMake.in
echo QMAKE_GLOBAL_LDFLAGS=@QMAKE_GLOBAL_LDFLAGS@ >> ./Config.QMake.in
echo QMAKE_SHARED_LDFLAGS=@QMAKE_SHARED_LDFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Linker settings for QMake libraries >> ./Config.QMake.in
echo QMAKE_GLOBAL_LDLIBFLAGS=@QMAKE_GLOBAL_LDLIBFLAGS@ >> ./Config.QMake.in
echo QMAKE_SHARED_LDLIBFLAGS=@QMAKE_SHARED_LDLIBFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Specific linker functionality >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo LIBPATHSWITCH=@LIBPATHSWITCH@ >> ./Config.QMake.in
echo UNDEFSYMB=@UNDEFSYMB@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for the archiver >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo QMAKE_ARFLAGS=@QMAKE_ARFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# Flags for make >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# User specified at configure time >> ./Config.QMake.in
echo CONFIGURE_MAKEFLAGS=@MAKEFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo \# General QMake make settings >> ./Config.QMake.in
echo QMAKE_MAKEFLAGS=@QMAKE_MAKEFLAGS@ >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in
echo >> ./Config.QMake.in



# Map Platform/OS into Ravl ARCs
case X$host_os in
Xarm )      ARC=arm
            # Unlikely to ever encounter this as an OS, most liekly to find arm
            # as a linux cpu type (should process that as a variation of linux).
            # This is only here as this is what we used to do
            ;;
Xcygwin )   ARC=cygwin
            # Ravl support assumes an old i*86 platform. Could possibly require
            # code changes to Ravl if we ever port to a more expansive
            # environment (e.g. 64 bit support)
            ;;
Xdarwin* )  case X`uname -m` in
               Xx86_64 )  ARC=macosx64
                          ;;
               Xi386 )    ARC=macosx32
                          ;;
               Xpowerpc ) ARC=macosxppc
                          ;;
            esac
            # The above is needed as host_os currently comes back as i386 even
            # for 64-bit installations of OSX
            ;;
Xirix* )    ARC=sgi
            ;;
Xlinux* )
            case X$host_cpu in
               Xi386* | Xi486* | Xi586* | Xi686* | Xi786* )
                          ARC=linux
                          ;;
               Xx86_64* ) ARC=linux64
                          ;;
               Xalpha* )  ARC=linux_alpha
                          # Will encompass more processors than the original
                          # config.arc accounted for but should not be an issue
                          ;;
               * )        ARC=unknown
                          ;;
            esac
            ;;
Xosf* )     ARC=alpha
            ;;
Xsolaris* ) VER_STRING=`expr X$host_os : 'Xsolaris2\.\([0-9]*\).*'`
            # config.guess returns solaris2.X regardless of if it is Solaris 2.5;
            #  Solaris 2.6, Solaris 7, Solaris 8, etc.
            if ( test X$VER_STRING= X )
            then
              # Just in case GNU change config.guess
              ARC=sol
            else
              # Original Ravl support was for Solaris 7 & 9 only. Later
              # versions may need further work on the Ravl source
              ARC=sol2_$VER_STRING
            fi
            ;;
* )         ARC=unknown
            ;;
esac
AC_SUBST([ARC])



dnl Create Makefiles and Config.QMake
AC_CONFIG_FILES([Config.QMake])
AC_CONFIG_FILES([Makefile:src/AutoTools/Makefile.in])
AC_CONFIG_FILES([Targets.QMake:src/AutoTools/Targets.QMake.in])
AC_OUTPUT
