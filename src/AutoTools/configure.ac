dnl#JFi:ToDo-Check for ver 5 of Perl as a requirement
dnl#JFi:ToDo-Change --with-<lib>=path to imply $path/lib and $path/include
dnl#JFi:ToDo-Extend --with-<lib>[-{library,include}] to allow multiple entries
dnl#JFi:ToDo-Maybe use AC_CONFIG_SRCDIR([include/ccmath/ccmath.h])
dnl#JFi:ToDo-Investigate position of 32 v 64 bit lib paths
dnl#JFi:ToDo-Investigate position of shared v static pkg-config results
dnl#JFi:ToDo-Extend --with-${lib} to specify location/name of .pc
dnl#JFi:ToDo-Extend --with-${lib}-pkgconf to specify path for known .pc

# Initialisation
AC_PREREQ([2.65])

AC_INIT([RAVL], [1.1.9], [ravl-devel@lists.sourceforge.net], [Ravl], [http://www.ee.surrey.ac.uk/CVSSP/Ravl])
dnl QMake doesn't use PACKAGE_TARNAME, it uses a hard-coded RAVL to set the 
dnl name of the dist archive/dir. However, the tarname parameter to AC_INIT
dnl (the 4th item) is used to prime the default for --docdir hence the value
dnl used here.

# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2011-16, University of Surrey
# This code may be redistributed under the terms of the GNU General
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html


dnl Include standard QMake autoconf macros
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CHOOSE_GETCWD])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CHOOSE_RPATH])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CK_COMPILE_CK_LIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CK_BESPOKE_CK_LIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CK_PREREQ_STATE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CHECK_CC_LIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_Debug_Fn])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_ALL_EXTLIBS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_CC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_PC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_VIRTUAL_EXTLIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_BESPOKE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_BESPOKE_CC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_RESOURCE_BASIC_SWITCHES])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_RESOURCE_CC_SWITCHES])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_FINISH_DEFINE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_KEY_TOOLS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_LD])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_MAKE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_QMAKE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_QMAKE_DEPS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_PROCESS_EXTLIBS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_READ_USER_ENV])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_ARC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_ARCHIVER_ARFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_COMPILER_CFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_COMPILER_CCFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_ID_GNU_COMPILER])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_LINKER_LDFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_MAKE_MAKEFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_SHAREDEXT])


dnl# We want to default to installing to /usr rather than /usr/local
dnl# This is primarily because of a number of distributions that do not
dnl# come configured to use /usr/local/lib as a default library location.
AC_PREFIX_DEFAULT([/usr])


# Main body of configure script


# First determine the current platform
AC_CANONICAL_HOST

# Locate the tools needed to run this script (locate QMake; shell utils;
# compilers etc.)
QM_LOCATE_KEY_TOOLS


   
# Locate other tools needed to run QMake (as a means of procesing any user
# settings rather than requirement - QMake will already be aware of them).
QM_LOCATE_QMAKE_DEPS



# Set relevant flags dependant on the software tools we have found
#
dnl Only select flags dependant on the tools in use here. Set all options
dnl pertaining to platform specifics later in this script
dnl
# First consider the C compiler
#
# If the user has specified the compiler, re-generate the compiler specific
# CFLAGS in case the ones generated into the QMake configuration relate to
# a different compiler.
if ( test X$USER_CC != X )
then
   # User has specified a different compiler, update CFLAGS
   QM_SET_COMPILER_CFLAGS
else
   # Bring in any configured CFLAGS from the QMake configuration
   CONFIGURE_CFLAGS=" \$(QMAKE_CFLAGS) $CONFIGURE_CFLAGS"
fi
#
# Set PKG_xx_CFLAGS to defaults or any package specific values
PKG_CANSIFLAG=" \$(COMPILER_ANSI_CFLAGS) "
PKG_CHECK_CFLAGS=" -DRAVL_CHECK "
PKG_DEBUG_CFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
PKG_DEFAULT_CFLAGS=" \$(COMPILER_NON_OPT_CFLAGS) "
PKG_GLOBAL_CFLAGS=" \$(COMPILER_SPECIFIC_CFLAGS) \$(COMPILER_DEBUG_CFLAGS) "
PKG_GPROF_CFLAGS=" \$(COMPILER_GPROF_CFLAGS) "
PKG_OPT_CFLAGS=" \$(COMPILER_OPT_CFLAGS) "
PKG_PROF_CFLAGS=" \$(COMPILER_PROF_CFLAGS) "
PKG_SHARED_CFLAGS=" \$(COMPILER_SHARED_CFLAGS) "
#
# Declare output variables for compiler ANSI switches
AC_SUBST([PKG_CANSIFLAG])
AC_SUBST([PKG_CCANSIFLAG])
#
# Declare PKG_XXXXX_CFLAGS variables as output for Config.Project
AC_SUBST([PKG_CHECK_CFLAGS])
AC_SUBST([PKG_DEBUG_CFLAGS])
AC_SUBST([PKG_DEFAULT_CFLAGS])
AC_SUBST([PKG_GLOBAL_CFLAGS])
AC_SUBST([PKG_GPROF_CFLAGS])
AC_SUBST([PKG_OPT_CFLAGS])
AC_SUBST([PKG_PROF_CFLAGS])
AC_SUBST([PKG_SHARED_CFLAGS])
#
# Allow output of any user provided CFLAGS (these will already have been
# preserved in the CONFIGURE_CFLAGS variable)
AC_SUBST([CONFIGURE_CFLAGS])

# Now consider the C+ compiler
#
# If the user has specified the compiler, re-generate the compiler specific
# CCFLAGS in case the ones generated into the QMake configuration relate to
# a different compiler.
if ( test X$USER_CXX != X )
then
   # User has specified a different compiler, update CCFLAGS
   QM_SET_COMPILER_CCFLAGS
else
   # Bring in any configured CCFLAGS from the QMake configuration
   CONFIGURE_CCFLAGS=" \$(QMAKE_CCFLAGS) $CONFIGURE_CCFLAGS"
fi
#
# Set PKG_xx_CFLAGS to defaults or any package specific values
PKG_CCANSIFLAG=" \$(COMPILER_ANSI_CCFLAGS) "
PKG_CHECK_CCFLAGS=" -DRAVL_CHECK "
PKG_DEBUG_CCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
PKG_DEFAULT_CCFLAGS=" \$(COMPILER_NON_OPT_CCFLAGS) "
PKG_GLOBAL_CCFLAGS=" \$(COMPILER_SPECIFIC_CCFLAGS) \$(COMPILER_DEBUG_CCFLAGS) "
PKG_GPROF_CCFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
PKG_OPT_CCFLAGS=" \$(COMPILER_OPT_CCFLAGS) "
PKG_PROF_CCFLAGS=" \$(COMPILER_PROF_CCFLAGS) "
PKG_SHARED_CCFLAGS=" \$(COMPILER_SHARED_CCFLAGS) "
#
# Declare PKG_XXXXX_CCFLAGS variables as output for Config.Project
AC_SUBST([PKG_CHECK_CCFLAGS])
AC_SUBST([PKG_DEBUG_CCFLAGS])
AC_SUBST([PKG_DEFAULT_CCFLAGS])
AC_SUBST([PKG_GLOBAL_CCFLAGS])
AC_SUBST([PKG_GPROF_CCFLAGS])
AC_SUBST([PKG_OPT_CCFLAGS])
AC_SUBST([PKG_PROF_CCFLAGS])
AC_SUBST([PKG_SHARED_CCFLAGS])
#
# Allow output of any user provided CCFLAGS (these will already have been
# preserved in the CONFIGURE_CCFLAGS variable)
AC_SUBST([CONFIGURE_CCFLAGS])


# Now consider the linker
#
# If the user has specified the linker, re-generate the linker specific
# LDFLAGS in case the ones generated into the QMake configuration relate to
# a different linker.
if ( test X$USER_LD != X )
then
   # User has specified a different linker, update LDFLAGS
   QM_SET_LINKER_LDFLAGS
   # Take the time to record any user-provided LDFLAGS for output to the
   # configuration file (as we have not recorded them yet)
   CONFIGURE_LDFLAGS=" $LDFLAGS "
   # Assume any LDFLAGS apply equally to libraries
   CONFIGURE_LDLIBFLAGS=" $LDFLAGS "
else
   # Bring in any configured LDFLAGS from the QMake configuration
   # plus any user-provided LDFLAGS
   CONFIGURE_LDFLAGS=" \$(QMAKE_LDFLAGS) $LDFLAGS "
   # Assume any LDFLAGS apply equally to libraries
   CONFIGURE_LDLIBFLAGS=" \$(QMAKE_LDLIBFLAGS) $LDFLAGS "
fi
#
# Set package specific LDFLAGS
if ( test X$GNU_LD = Xyes )
then
  # We are using a GNU linker

  # General Ravl link options
  PKG_CHECK_LDFLAGS=
  PKG_DEBUG_LDFLAGS=
  # -g does nothing on the linker so omit it
  PKG_DEFAULT_LDFLAGS=
  PKG_GLOBAL_LDFLAGS=" \$(LINKER_SPECIFIC_LDFLAGS) "
  PKG_GPROF_LDFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl .._LDFLAGS currently only ever used when the linker is called by the C++
dnl compiler, hence we use the C++ compiler's GPROF switch. If QMake is ever
dnl extended, we probably need extra versions of .._LDFLAGS for when the linker
dnl is called from the regular C compiler or when it is called directly. If
dnl using the linker directly, "GPROF" needs to cause /lib/gcrt0.o to be linked
dnl in as the first object file and to use -lc_p instead of defaulting to -lc.
  PKG_OPT_LDFLAGS=
  PKG_PROF_LDFLAGS=
  PKG_SHARED_LDFLAGS=" \$(LINKER_SHARED_LDFLAGS) "

  # Ravl link options used for libraries
  PKG_CHECK_LDLIBFLAGS=
  PKG_DEBUG_LDLIBFLAGS=
  PKG_DEFAULT_LDLIBFLAGS=
  PKG_GLOBAL_LDLIBFLAGS=" \$(LINKER_SPECIFIC_LDLIBFLAGS) "
  PKG_GPROF_LDLIBFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl As for .._LDFLAGS, .._LDLIBFLAGS is currently only ever used when the c++
dnl compiler calls the linker. All the caveats as per .._LDFLAGS apply (see
dnl above)
  PKG_OPT_LDLIBFLAGS=
  PKG_PROF_LDLIBFLAGS=
  PKG_SHARED_LDLIBFLAGS=" \$(LINKER_SHARED_LDLIBFLAGS) "
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
  Xcygwin )   PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=" -g "
              PKG_GLOBAL_LDFLAGS=" \$(LINKER_SPECIFIC_LDFLAGS) "
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDFLAGS
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=" \$(LINKER_SHARED_LDFLAGS) "

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=" \$(LINKER_SPECIFIC_LDLIBFLAGS) "
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDLIBFLAGS
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" \$(LINKER_SHARED_LDLIBFLAGS) "
              ;;
  Xirix*    ) PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=
              PKG_GLOBAL_LDFLAGS=" \$(LINKER_SPECIFIC_LDFLAGS) "
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDFLAGS
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=" \$(LINKER_SHARED_LDFLAGS) "

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=" \$(LINKER_SPECIFIC_LDLIBFLAGS) "
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDLIBFLAGS
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" \$(LINKER_SHARED_LDLIBFLAGS) "
              ;;
  Xsolaris* ) PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=" -g "
              PKG_GLOBAL_LDFLAGS=" \$(LINKER_SPECIFIC_LDFLAGS) "
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDFLAGS
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS= "\$(LINKER_SHARED_LDFLAGS) "

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=" \$(LINKER_SPECIFIC_LDLIBFLAGS) "
              PKG_DEFAULT_LDLIBFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDLIBFLAGS
              PKG_GPROF_LDLIBFLAGS=
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" \$(LINKER_SHARED_LDLIBFLAGS) "
              ;;
  * )         PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=
              PKG_GLOBAL_LDFLAGS=" \$(LINKER_SPECIFIC_LDFLAGS) "
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDFLAGS
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=" \$(LINKER_SHARED_LDFLAGS) "

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=" \$(LINKER_SPECIFIC_LDLIBFLAGS) "
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=" \$(COMPILER_GPROF_CCFLAGS) "
dnl See discussion in GNU linker section above, about caveats for .._LDLIBFLAGS
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" \$(LINKER_SHARED_LDLIBFLAGS) "
              ;;
  esac
fi
#
# Declare linker variables as output for Config.Project
AC_SUBST([CONFIGURE_LDFLAGS])
AC_SUBST([CONFIGURE_LDLIBFLAGS])
# any user-provided LDFLAGS will be recorded in Config.Project as
# CONFIGURE_LDFLAGS (for use linking binaries) and CONFIGURE_LDLIBFLAGS
# (for when linking libraries).
#
AC_SUBST([PKG_CHECK_LDFLAGS])
AC_SUBST([PKG_DEBUG_LDFLAGS])
AC_SUBST([PKG_DEFAULT_LDFLAGS])
AC_SUBST([PKG_GLOBAL_LDFLAGS])
AC_SUBST([PKG_GPROF_LDFLAGS])
AC_SUBST([PKG_OPT_LDFLAGS])
AC_SUBST([PKG_PROF_LDFLAGS])
AC_SUBST([PKG_SHARED_LDFLAGS])
#
AC_SUBST([PKG_CHECK_LDLIBFLAGS])
AC_SUBST([PKG_DEBUG_LDLIBFLAGS])
AC_SUBST([PKG_DEFAULT_LDLIBFLAGS])
AC_SUBST([PKG_GLOBAL_LDLIBFLAGS])
AC_SUBST([PKG_GPROF_LDLIBFLAGS])
AC_SUBST([PKG_OPT_LDLIBFLAGS])
AC_SUBST([PKG_PROF_LDLIBFLAGS])
AC_SUBST([PKG_SHARED_LDLIBFLAGS])

# CUDA Compiler
#
# Set PKG_xx_CFLAGS to defaults or any package specific values
PKG_NVCCANSIFLAG=" \$(NVCC_ANSI_NVCCFLAGS) "
PKG_CHECK_NVCCFLAGS=" -DRAVL_CHECK "
PKG_DEBUG_NVCCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
PKG_DEFAULT_NVCCFLAGS=" \$(NVCC_NON_OPT_NVCCFLAGS) "
PKG_GLOBAL_NVCCFLAGS=" \$(NVCC_SPECIFIC_NVCCFLAGS) \$(NVCC_DEBUG_NVCCFLAGS) "
PKG_GPROF_NVCCFLAGS=" \$(NVCC_GPROF_NVCCFLAGS) "
PKG_OPT_NVCCFLAGS=" \$(NVCC_OPT_NVCCFLAGS) "
PKG_PROF_NVCCFLAGS=" \$(NVCC_PROF_NVCCFLAGS) "
PKG_SHARED_NVCCFLAGS=" \$(NVCC_SHARED_NVCCFLAGS) "
dnl Declare PKG_XXXXX_NVCCFLAGS variables as output for Config.Project
AC_SUBST([PKG_NVCCANSIFLAG])
AC_SUBST([PKG_CHECK_NVCCFLAGS])
AC_SUBST([PKG_DEBUG_NVCCFLAGS])
AC_SUBST([PKG_DEFAULT_NVCCFLAGS])
AC_SUBST([PKG_GLOBAL_NVCCFLAGS])
AC_SUBST([PKG_GPROF_NVCCFLAGS])
AC_SUBST([PKG_OPT_NVCCFLAGS])
AC_SUBST([PKG_PROF_NVCCFLAGS])
AC_SUBST([PKG_SHARED_NVCCFLAGS])


# Consider the archiver
#
# If the user has specified the archiver, re-generate the archiver specific
# ARFLAGS in case the ones generated into the QMake configuration relate to
# a different archiver.
if ( test X$USER_AR != X )
then
   # User has specified a different archiver, update ARFLAGS
   QM_SET_ARCHIVER_ARFLAGS
fi
#
# Set package specific ARFLAGS
PKG_ARFLAGS=" \$(AR_SPECIFIC_ARFLAGS) "
# Allow recording in Config.Package
AC_SUBST([PKG_ARFLAGS])


# Lastly consider makefile
# Declare flags for make
#
# If the user has specified make, re-generate the specific MAKEFLAGS
# in case the ones generated into the QMake configuration relate to a
# different archiver.
if ( test X$USER_MAKE != X )
then
   # User has specified a different make, update MAKEFLAGS
   QM_SET_MAKE_MAKEFLAGS
   # We are yet to record any user-provided MAKEFLAGS, so do that now
   CONFIGURE_MAKEFLAGS=" $MAKEFLAGS "
else
   # Bring in any configured MAKEFLAGS from the QMake configuration
   CONFIGURE_MAKEFLAGS=" \$(QMAKE_MAKEFLAGS) $MAKEFLAGS"
fi
#
# Set package specific MAKEFLAGS
PKG_MAKEFLAGS=" \$(MAKE_SPECIFIC_MAKEFLAGS) "

AC_SUBST([CONFIGURE_MAKEFLAGS])
# User-provided MAKEFLAGS will be recorded as CONFIGURE_MAKEFLAGS in Config.Project 
AC_SUBST([PKG_MAKEFLAGS])


dnl End of software tools section





trace 2 Initial checks complete, now processing external libraries

# Define all Ravl Helper libraries and their associated cmd-line options
#
QM_DEFINE_ALL_EXTLIBS()

# Ravl cannot currently cope with the release versions of several libraries,
# declare these to be broken (and ultimatley ignore them in the Ravl build).
BrokenLibs="libdc1394 libavc1394 avifile"


# If debugging, output what libraries are defined
if ( test X$DBG != X && test $DBG -ge 3 )
then
   trace 3 Defined libraries
   for lib in $KnownLibs
   do
      eval lib_enablement=\$use_${lib} 
      TRACE_INDENT="     "
      eval check=\$with_${lib}_check
      case $check in
         'bespoke' | 'virtual'  )
                       eval prereq=\$with_${lib}_prq
                       trace 3 "$lib:$check:$prereq:$lib_enablement"
                       ;;
         'pkgconfig' ) eval pkgconf=\$with_${lib}_pkgconf
                       trace 3 "$lib:$check:$pkgconf:$lib_enablement"
                       ;;
         'compile' )   eval testwith="\$with_${lib}_tstcd"
                       eval prereq=\$with_${lib}_prq
                       eval lnklibs=\$with_${lib}_lnklibs
                       eval libpath=\$with_${lib}_libpath
                       eval hdr=\$with_${lib}_hdr
                       eval inc=\$with_${lib}_inc
                       trace 3 "$lib:$check:$testwith:$prereq:$lnklibs:$libpath:$hdr:$inc:$lib_enablement"
                       ;;
         * )           trace 1 "Assert - Unrecognised test-type for $lib"
                       ;;
      esac
   done
fi


# Default, from QMake's configuration, those libraries that the user has not
# explicitly configured.
# 
# First extract the list of RESOURCES configured into QMake
QMAKE_RESOURCES=`cat $QMAKE_CFG | $SED -n 's/^RESOURCES=\(.*\)/\1/p'`
trace 6 QMake Resources: @<:@$QMAKE_RESOURCES@:>@
#
# Now reconcile those with the user configured list
CheckOverride=""
for lib in $ConfiguredLibs
do
   if ( test x`$EXPR " $QMAKE_RESOURCES " : ".* \($lib\) .*"` == "x$lib" )
   then
      # Configured lib in QMake RESOURCE configuration, we may need to override
      # this setting later
      CheckOverride="$CheckOverride $lib"
   fi
done

# Now prime the un-configured packages
#
for lib in $UnconfiguredLibs
do
   if ( test x`$EXPR " $QMAKE_RESOURCES " : ".* \($lib\) .*"` != "x$lib" )
   then
      # Lib not available
      eval "use_${lib}=missing"
      eval "${lib}_CFLAGS="
      eval "${lib}_CCFLAGS="
      eval "${lib}_LIBS="
      dnl Extend this section to nullify the relevant Fortran make flags if
      dnl we ever add Fortran support
   else
      # Inherit QMake configuration settings
      eval "use_${lib}=present"
      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CFLAGS=\(.*\)/\1/p"`
      eval "${lib}_CFLAGS=\$QMakeSetting"
      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CCFLAGS=\(.*\)/\1/p"`
      eval "${lib}_CCFLAGS=\$QMakeSetting"
      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_LIBS=\(.*\)/\1/p"`
      eval "${lib}_LIBS=\$QMakeSetting"
      dnl Extend this section to read in the Fortran make flags if we ever
      dnl add Fortran support
      # We give no consideration to package pre-requisites here. If we inherit
      # from the QMake configuration, we inherit the package with the original
      # pre-requisite package(s). For example, if we inherit libTIFF, we inherit
      # it with the originally configured libZ regardless of if the user has
      # configured Ravl to use a different libZ. This is consitent with those
      # packages that are specified by pkg-configure as there is no way to
      # influence what dependant libraries they bring in (libGnome will use
      # whatever JPEG library it was set to use by the Gnome developers, even
      # if we configure Ravl to link with JPEGturbo)..
   fi
done

# Need to drop the RESOURCES Ravl cannot currently cope with
for lib in $BrokenLibs
do
   trace 6 Considering Broken lib: @<:@$lib@:>@
   if ( test x`$EXPR " $QMAKE_RESOURCES " : ".* \($lib\) .*"` == "x$lib" )
   then
      eval "use_${lib}=missing"
      eval "${lib}_CFLAGS="
      eval "${lib}_CCFLAGS="
      eval "${lib}_LIBS="
      dnl Extend this section to read in the Fortran make flags if we ever
      dnl add Fortran support
      CheckOverride="$CheckOverride $lib"
      trace 6 Broken lib @<:@$lib@:>@ removed
   fi
done


# If debugging, output what library definitions are after reading in the
# unconfigured packages
if ( test X$DBG != X && test $DBG -ge 3 )
then
   trace 3 Defaulted libraries
   for lib in $KnownLibs
   do
      eval lib_enablement=\$use_${lib} 
      TRACE_INDENT="     "
      eval check=\$with_${lib}_check
      case $check in
         'bespoke' | 'virtual' )
                       eval prereq=\$with_${lib}_prq
                       trace 3 "$lib:$check:$prereq:$lib_enablement"
                       ;;
         'pkgconfig' ) eval pkgconf=\$with_${lib}_pkgconf
                       trace 3 "$lib:$check:$pkgconf:$lib_enablement"
                       ;;
         'compile' )   eval testwith="\$with_${lib}_tstcd"
                       eval prereq=\$with_${lib}_prq
                       eval lnklibs=\$with_${lib}_lnklibs
                       eval libpath=\$with_${lib}_libpath
                       eval hdr=\$with_${lib}_hdr
                       eval inc=\$with_${lib}_inc
                       trace 3 "$lib:$check:$testwith:$prereq:$lnklibs:$libpath:$hdr:$inc:$lib_enablement"
                       ;;
         * )           trace 1 "Assert - Unrecognised test-type for $lib"
                       ;;
      esac
   done
fi

# Now process the libraries the user has explicitly configured.
LibsToProcess=$ConfiguredLibs
QM_PROCESS_EXTLIBS()

# OpenCV is a special case, we need to post-process its include path
if test x$use_OpenCV=xpresent
then
   # Add extra include directory for OpenCV so as to prevent any confusion over
   # the inclusion of opencv/xx.h (previously, Ravl code included the OpenCV
   # headers by specifying the opencv sub-dir; this was at odds with what the
   # .pc file was set up to expect the user to do). To combat this, we add the
   # parent directory (of the opencv directory) to the include path.
   OpenCV_CFLAGS=`echo "$OpenCV_CFLAGS " | $SED '/\(-I@<:@^ @:>@*\)\/opencv /s//\1\/opencv \1 /g'`
   dnl sed command is '/\(-I[^ ]*\)\/opencv /s//\1\/opencv \1 /g'
fi


# Now start checking for the existence of all we need or can use

trace 2 Checking for necessary dependancies

# Perl required by QMake 
AC_PATH_PROG([PERL],[perl])
AS_IF([test "X$PERL" != X], 
      [RESOURCES="$RESOURCES PERL "],
      [AC_MSG_FAILURE([Cannot find a working copy of Perl])
      ]
     )
AC_SUBST([PERL])
TRACE_INDENT="     "
trace 3 RESOURCES now set to "@<:@$RESOURCES@:>@"

# Check for other programs that can be used by RAVL itself

trace 2 Checking for programs that can be used by Ravl

AC_PATH_PROG([GnuPlot],[gnuplot])
AS_IF([test "X$GnuPlot" != X], 
      [ RESOURCES="$RESOURCES GnuPlot "],
      [AC_MSG_WARN([Cannot find a working copy of GNU Plot, some graph plotting functionality of Ravl requires this utility at run-time.])]
     )
AC_SUBST([GnuPlot])
TRACE_INDENT="     "
trace 3 RESOURCES now set to "@<:@$RESOURCES@:>@"


AC_PATH_PROG([SWIG],[swig])
AS_IF([test "X$SWIG" != X], [ RESOURCES="$RESOURCES Swig "])
AC_SUBST([SWIG])
TRACE_INDENT="     "
trace 3 RESOURCES now set to "@<:@$RESOURCES@:>@"


# Now check if we need to disallow any RESOURCES inherited from the QMake
# configuration
#
if ( test "x$CheckOverride" != x )
then
   # We have some QMake RESOURCES that the user re-configured, we need to
   # check they are still available
   for lib in $CheckOverride
   do
      eval "lib_enablement=\$use_${lib}" 
      if ( test $lib_enablement != present )
      then
         # package is no longer enabled, must override QMake RESOURCES
         ResetRESOURCES=yes
         QMAKE_RESOURCES=`echo " $QMAKE_RESOURCES "  | $SED "s/ $lib / /"`
      fi
   done
   if ( test x$ResetRESOURCES == xyes )
   then
      RESOURCES="$RESOURCES $QMAKE_RESOURCES"
   fi
fi


# Check if the user wishes to disable fast-install
AC_ARG_ENABLE([fast-install],
              [AS_HELP_STRING([--disable-fast-install],
                              [Do not build Ravl ready for immediate install, build it so it can be run from the build directory and rebuild as required on an install
                              ]
                             )
              ],
              [trace 5 fast-install set as $enableval],
              [trace 5 fast-install not explicitly disabled will use it
               enable_fast_install=yes
              ]
             )


#
# Create proforma Config.Project.in
#

# Create initial, empty file
echo > ./Config.Project.in

echo \# Ravl configuration file - regenerate using configure >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# We would like to set PROJECT_NAME=RAVL here. However, because Ravl pre-dates >> ./Config.Project.in
echo \# the full QMake PROJECT concept, we cannot currently set PROJECT_NAME owing to >> ./Config.Project.in
echo \# the knock on effect. Instead we use FALLBACK_PROJNAME as this provides the name >> ./Config.Project.in
echo \# for certain functionality while not introducing the full consequences of setting >> ./Config.Project.in
echo \# PROJECT_NAME. >> ./Config.Project.in
echo FALLBACK_PROJNAME=Ravl >> ./Config.Project.in
echo PROJECT_DESCRIPTION='RAVL, Recognition And Vision Library' >> ./Config.Project.in
echo PROJECT_URL=@PACKAGE_URL@ >> ./Config.Project.in
echo PROJECT_BUGREPORT=@PACKAGE_BUGREPORT@ >> ./Config.Project.in

echo \# Declare current version of RAVL >> ./Config.Project.in
echo PROJECT_VERSION=@PACKAGE_VERSION@ >> ./Config.Project.in
sedPattern='/^\@{:@@<:@0-9@:>@@<:@0-9@:>@*\@:}@\.\@{:@@<:@0-9@:>@@<:@0-9@:>@*\@:}@\.\@{:@@<:@0-9@:>@@<:@0-9@:>@*\@:}@ *@S|@'
dnl# Pattern minus quadrigraphs is '/^\([0-9][0-9]*\)\.\([0-9][0-9]*\)\.\([0-9][0-9]*\) *$'
PROJECT_MAJOR=`echo $PACKAGE_VERSION | $SED -n "$sedPattern/s//\1/p"`
AC_SUBST([PROJECT_MAJOR])
echo PROJECT_MAJOR=@PROJECT_MAJOR@ >> ./Config.Project.in
echo PROJECT_DISTDIR=RAVL-@PACKAGE_VERSION@ >> ./Config.Project.in
echo >> ./Config.Project.in

# Provide locations for QMake to work with
echo \# Location to which to build >> ./Config.Project.in
echo PROJECT_OUT=@abs_builddir@ >> ./Config.Project.in
echo INSTALLHOME=@abs_builddir@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Location for transient build files >> ./Config.Project.in
echo LOCALTMP=\$\(PROJECT_OUT\)/transient >> ./Config.Project.in
echo >> ./Config.Project.in

# Define location of compilers and helper programs for build system that
# override the QMake default settings
#
dnl Override comes from user supplying XXXX= on configure cmd line or
dnl selecting an appropriate configure switch.
dnl 
dnl Testing for XXX variable being set will detect those overrides for 
dnl command not used in this script. However, those commands used in this
dnl script (awk, sed, etc.) will have had the variable set up. For this
dnl reason, we check USER_XXX which will preserved any user setting prior
dnl to this script setting the XXX variable for internal use.
dnl
echo >> ./Config.Project.in
echo \# List of modified program settings >> ./Config.Project.in
echo >> ./Config.Project.in

if ( test "x$USER_AR" != x )
then
  AC_SUBST([USER_AR])
  echo AR=@USER_AR@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo AR_SPECIFIC_ARFLAGS=@AR_SPECIFIC_ARFLAGS@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi

if ( test "x$USER_AWK" != x )
then
  AC_SUBST([USER_AWK])
  echo AWK=@USER_AWK@ >> ./Config.Project.in
fi

if ( test "x$USER_CC" != x )
then
  AC_SUBST([USER_CC])
  echo CC=@USER_CC@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo COMPILER_ANSI_CFLAGS=@COMPILER_ANSI_CFLAGS@ >> ./Config.Project.in
  # Only output the standards flags if the variable has explicitly been set (this
  # allows QMake to distinguish between the standard needing no compiler flag (the
  # variable is set but null) and not being available (variable is not set at
  # all).
  if ( test "X${COMPILER_C90_CFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_C90_CFLAGS=@COMPILER_C90_CFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_C95_CFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_C95_CFLAGS=@COMPILER_C95_CFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_C99_CFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_C99_CFLAGS=@COMPILER_C99_CFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_C11_CFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_C11_CFLAGS=@COMPILER_C11_CFLAGS@ >> ./Config.Project.in
  fi
  echo COMPILER_DEBUG_CFLAGS=@COMPILER_DEBUG_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_GPROF_CFLAGS=@COMPILER_GPROF_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_NON_OPT_CFLAGS=@COMPILER_NON_OPT_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_OPT_CFLAGS=@COMPILER_OPT_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_PROF_CFLAGS=@COMPILER_PROF_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_SHARED_CFLAGS=@COMPILER_SHARED_CFLAGS@ >> ./Config.Project.in
  echo COMPILER_SPECIFIC_CFLAGS=@COMPILER_SPECIFIC_CFLAGS@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi

if ( test "x$USER_CP" != x )
then
  AC_SUBST([CP])
  echo CP=@CP@ >> ./Config.Project.in
fi

if ( test "x$USER_CPP" != x )
then
  AC_SUBST([USER_CPP])
  echo CPP=@USER_CPP@ >> ./Config.Project.in
fi

if ( test "x$USER_CHGRP" != x )
then
  AC_SUBST([CHGRP])
  echo CHGRP=@CHGRP@ >> ./Config.Project.in
fi

if ( test "x$USER_CHMOD" != x )
then
  AC_SUBST([CHMOD])
  echo CHMOD=@CHMOD@ >> ./Config.Project.in
fi

if ( test "x$USER_CHOWN" != x )
then
  AC_SUBST([CHOWN])
  echo CHOWN=@CHOWN@ >> ./Config.Project.in
fi

if ( test "x$USER_CXX" != x )
then
  AC_SUBST([USER_CXX])
  echo CXX=@USER_CXX@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo COMPILER_ANSI_CCFLAGS=@COMPILER_ANSI_CCFLAGS@ >> ./Config.Project.in
  # Only output the standards flags if the variable has explicitly been set (this
  # allows QMake to distinguish between the standard needing no compiler flag (the
  # variable is set but null) and not being available (variable is not set at
  # all).
  if ( test "X${COMPILER_CC98_CCFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_CC98_CCFLAGS=@COMPILER_CC98_CCFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_CC03_CCFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_CC03_CCFLAGS=@COMPILER_CC03_CCFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_CC11_CCFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_CC11_CCFLAGS=@COMPILER_CC11_CCFLAGS@ >> ./Config.Project.in
  fi
  if ( test "X${COMPILER_CC14_CCFLAGS-Unset}" != XUnset )
  then
     echo COMPILER_CC14_CCFLAGS=@COMPILER_CC14_CCFLAGS@ >> ./Config.Project.in
  fi
  echo COMPILER_DEBUG_CCFLAGS=@COMPILER_DEBUG_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_GPROF_CCFLAGS=@COMPILER_GPROF_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_NON_OPT_CCFLAGS=@COMPILER_NON_OPT_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_OPT_CCFLAGS=@COMPILER_OPT_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_PROF_CCFLAGS=@COMPILER_PROF_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_SHARED_CCFLAGS=@COMPILER_SHARED_CCFLAGS@ >> ./Config.Project.in
  echo COMPILER_SPECIFIC_CCFLAGS=@COMPILER_SPECIFIC_CCFLAGS@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi

if ( test "x$use_ccache" = xyes )
then
   # We do not use USER_CCACHE as the use of ccache is controlled by a 
   # command-line switch over and above the environment variable. If the
   # user has requested to use ccache via either the --with-ccache switch
   # or by setting the environment variable, use_ccache will be 'yes'.
   echo CCACHE=@CCACHE@ >> ./Config.Project.in
fi

if ( test "x$USER_CXXCPP" != x )
then
  AC_SUBST([USER_CXXCPP])
  echo CXXCPP=@USER_CXXCPP@ >> ./Config.Project.in
fi

if ( test "x$USER_EXPR" != x )
then
  AC_SUBST([USER_EXPR])
  echo EXPR=@USER_EXPR@ >> ./Config.Project.in
fi

if ( test "x$USER_GET_CWD" != x )
then
  AC_SUBST([USER_GET_CWD])
  echo GET_CWD=@USER_GET_CWD@ >> ./Config.Project.in
fi

if ( test "x$USER_GREP" != x )
then
  AC_SUBST([USER_GREP])
  echo GREP=@USER_GREP@ >> ./Config.Project.in
fi

if ( test "x$USER_INSTALL" != x )
then
  AC_SUBST([INSTALL])
  echo INSTALL=@INSTALL@ >> ./Config.Project.in
fi

if ( test "x$USER_INSTALL_DATA" != x )
then
  AC_SUBST([INSTALL_DATA])
  echo INSTALL_DATA=@INSTALL_DATA@ >> ./Config.Project.in
fi

if ( test "x$USER_INSTALL_PROGRAM" != x )
then
  AC_SUBST([INSTALL_PROGRAM])
  echo INSTALL=@INSTALL_PROGRAM@ >> ./Config.Project.in
fi

if ( test "x$USER_INSTALL_SCRIPT" != x )
then
  AC_SUBST([INSTALL_SCRIPT])
  echo INSTALL_SCRIPT=@INSTALL_SCRIPT@ >> ./Config.Project.in
fi

if ( test "x$USER_PKG_CONFIG" != x )
then
  AC_SUBST([USER_PKG_CONFIG])
  echo PKG_CONFIG=@USER_PKG_CONFIG@ >> ./Config.Project.in
fi

if ( test "x$USER_LD" != x )
then
  AC_SUBST([USER_LD])
  echo LD=@USER_LD@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo LINKER_SPECIFIC_LDFLAGS=@LINKER_SPECIFIC_LDFLAGS@ >> ./Config.Project.in
  echo LINKER_SHARED_LDFLAGS=@LINKER_SHARED_LDFLAGS@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo LINKER_SPECIFIC_LDLIBFLAGS=@LINKER_SPECIFIC_LDLIBFLAGS@ >> ./Config.Project.in
  echo LINKER_SHARED_LDLIBFLAGS=@LINKER_SHARED_LDLIBFLAGS@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo LINKER_LIBPATHSWITCH=@LINKER_LIBPATHSWITCH@ >> ./Config.Project.in
  echo LINKER_SET_SONAME=@LINKER_SET_SONAME@ >> ./Config.Project.in
  echo LINKER_UNDEFSYMB=@LINKER_UNDEFSYMB@ >> ./Config.Project.in
  echo >> ./Config.Project.in

fi

if ( test "x$USER_LEX" != x )
then
  AC_SUBST([LEX])
  echo LEX=@LEX@ >> ./Config.Project.in
fi

if ( test "x$USER_LEXLIB" != x )
then
  AC_SUBST([LEXLIB])
  echo LEXLIB=@LEXLIB@ >> ./Config.Project.in
fi

if ( test "x$USER_LN_S" != x )
then
  AC_SUBST([LN_S])
  echo LN_S=@LN_S@ >> ./Config.Project.in
fi

if ( test "x$USER_MAKE" != x )
then
  AC_SUBST([USER_MAKE])
  echo MAKE=@USER_MAKE@ >> ./Config.Project.in
  echo >> ./Config.Project.in
  echo MAKE_SPECIFIC_MAKEFLAGS=@MAKE_SPECIFIC_MAKEFLAGS@ >> ./Config.Project.in
  echo MAKE_SPECIFIC_PARALLEL=@MAKE_SPECIFIC_PARALLEL@ >> ./Config.Project.in
  echo >> ./Config.Project.in

fi

if ( test "x$USER_MKDIR_P" != x )
then
  AC_SUBST([MKDIR_P])
  echo MKDIR_P=@MKDIR_P@ >> ./Config.Project.in
fi

if ( test "x$USER_NVCC" != x )
then
  AC_SUBST([USER_NVCC])
  echo NVCC=@USER_NVCC@ >> ./Config.Project.in
fi

if ( test "x$USER_PAGER" != x )
then
  AC_SUBST([PAGER])
  echo PKG_PAGER=@PAGER@ >> ./Config.Project.in
fi

if ( test "x$USER_RANLIB" != x )
then
  AC_SUBST([RANLIB])
  echo RANLIB=@RANLIB@ >> ./Config.Project.in
fi

if ( test "x$USER_RM" != x )
then
  AC_SUBST([RM])
  echo RM=@RM@ >> ./Config.Project.in
fi

if ( test "x$USER_SED" != x )
then
  AC_SUBST([USER_SED])
  echo SED=@USER_SED@ >> ./Config.Project.in
fi

if ( test "x$USER_SORT" != x )
then
  AC_SUBST([USER_SORT])
  echo SORT=@USER_SORT@ >> ./Config.Project.in
fi

if ( test "x$USER_SYNC" != x )
then
  AC_SUBST([SYNC])
  echo SYNC=@SYNC@ >> ./Config.Project.in
fi

if ( test "x$USER_TOUCH" != x )
then
  AC_SUBST([TOUCH])
  echo TOUCH=@TOUCH@ >> ./Config.Project.in
fi

if ( test "x$USER_TR" != x )
then
  AC_SUBST([USER_TR])
  echo TR=@USER_TR@ >> ./Config.Project.in
fi

if ( test "x$USER_XARGS" != x )
then
  AC_SUBST([XARGS])
  echo XARGS=@XARGS@ >> ./Config.Project.in
fi

if ( test "x$USER_YACC" != x )
then
  AC_SUBST([YACC])
  echo YACC=@YACC@ >> ./Config.Project.in
fi

echo >> ./Config.Project.in


# Compiler, etc. switches
echo >> ./Config.Project.in
echo \# Relevant Ravl build process flags >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for the C compiler >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_CFLAGS=@CONFIGURE_CFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Ravl compiler settings >> ./Config.Project.in
echo PKG_CHECK_CFLAGS=@PKG_CHECK_CFLAGS@ >> ./Config.Project.in
echo PKG_DEBUG_CFLAGS=@PKG_DEBUG_CFLAGS@ >> ./Config.Project.in
echo PKG_DEFAULT_CFLAGS=@PKG_DEFAULT_CFLAGS@ >> ./Config.Project.in
echo PKG_GLOBAL_CFLAGS=@PKG_GLOBAL_CFLAGS@ >> ./Config.Project.in
echo PKG_GPROF_CFLAGS=@PKG_GPROF_CFLAGS@ >> ./Config.Project.in
echo PKG_OPT_CFLAGS=@PKG_OPT_CFLAGS@ >> ./Config.Project.in
echo PKG_PROF_CFLAGS=@PKG_PROF_CFLAGS@ >> ./Config.Project.in
echo PKG_SHARED_CFLAGS=@PKG_SHARED_CFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# ANSI switch settings >> ./Config.Project.in
echo PKG_CANSIFLAG=@PKG_CANSIFLAG@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Individual C standard switches >> ./Config.Project.in
if ( test X$PKG_C90_CFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_C90_CFLAGS=$PKG_C90_CFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_C90_CFLAGS" >> ./Config.Project.in
  echo "  PKG_C90_CFLAGS=\$(COMPILER_C90_CFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_C95_CFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_C95_CFLAGS=$PKG_C95_CFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_C95_CFLAGS" >> ./Config.Project.in
  echo "  PKG_C95_CFLAGS=\$(COMPILER_C95_CFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_C99_CFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_C99_CFLAGS=$PKG_C99_CFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_C99_CFLAGS" >> ./Config.Project.in
  echo "  PKG_C99_CFLAGS=\$(COMPILER_C99_CFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_C11_CFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_C11_CFLAGS=$PKG_C11_CFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_C11_CFLAGS" >> ./Config.Project.in
  echo "  PKG_C11_CFLAGS=\$(COMPILER_C11_CFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for the C++ compiler >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_CCFLAGS=@CONFIGURE_CCFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Ravl compiler settings >> ./Config.Project.in
echo PKG_CHECK_CCFLAGS=@PKG_CHECK_CCFLAGS@ >> ./Config.Project.in
echo PKG_DEBUG_CCFLAGS=@PKG_DEBUG_CCFLAGS@ >> ./Config.Project.in
echo PKG_DEFAULT_CCFLAGS=@PKG_DEFAULT_CCFLAGS@ >> ./Config.Project.in
echo PKG_GLOBAL_CCFLAGS=@PKG_GLOBAL_CCFLAGS@ >> ./Config.Project.in
echo PKG_GPROF_CCFLAGS=@PKG_GPROF_CCFLAGS@ >> ./Config.Project.in
echo PKG_OPT_CCFLAGS=@PKG_OPT_CCFLAGS@ >> ./Config.Project.in
echo PKG_PROF_CCFLAGS=@PKG_PROF_CCFLAGS@ >> ./Config.Project.in
echo PKG_SHARED_CCFLAGS=@PKG_SHARED_CCFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# ANSI switch settings >> ./Config.Project.in
echo PKG_CCANSIFLAG=@PKG_CCANSIFLAG@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Individual C+ standard switches >> ./Config.Project.in
if ( test X$PKG_CC98_CCFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_CC98_CCFLAGS=$PKG_CC98_CCFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_CC98_CCFLAGS" >> ./Config.Project.in
  echo "  PKG_CC98_CCFLAGS=\$(COMPILER_CC98_CCFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_CC03_CCFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_CC03_CCFLAGS=$PKG_CC03_CCFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_CC03_CCFLAGS" >> ./Config.Project.in
  echo "  PKG_CC03_CCFLAGS=\$(COMPILER_CC03_CCFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_CC11_CCFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_CC11_CCFLAGS=$PKG_CC11_CCFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_CC11_CCFLAGS" >> ./Config.Project.in
  echo "  PKG_CC11_CCFLAGS=\$(COMPILER_CC11_CCFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
if ( test X$PKG_CC14_CCFLAGS != X )
then
  # Explicit override from user - reflect this in Config.Project
  echo PKG_CC14_CCFLAGS=$PKG_CC14_CCFLAGS >> ./Config.Project.in
else
  echo "ifdef COMPILER_CC14_CCFLAGS" >> ./Config.Project.in
  echo "  PKG_CC14_CCFLAGS=\$(COMPILER_CC14_CCFLAGS)" >> ./Config.Project.in
  echo endif >> ./Config.Project.in
fi
echo >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for the linker >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for linking binaries >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_LDFLAGS=@CONFIGURE_LDFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Package specific linker settings >> ./Config.Project.in
echo PKG_CHECK_LDFLAGS=@PKG_CHECK_LDFLAGS@ >> ./Config.Project.in
echo PKG_DEBUG_LDFLAGS=@PKG_DEBUG_LDFLAGS@ >> ./Config.Project.in
echo PKG_DEFAULT_LDFLAGS=@PKG_DEFAULT_LDFLAGS@ >> ./Config.Project.in
echo PKG_GLOBAL_LDFLAGS=@PKG_GLOBAL_LDFLAGS@ >> ./Config.Project.in
echo PKG_GPROF_LDFLAGS=@PKG_GPROF_LDFLAGS@ >> ./Config.Project.in
echo PKG_OPT_LDFLAGS=@PKG_OPT_LDFLAGS@ >> ./Config.Project.in
echo PKG_PROF_LDFLAGS=@PKG_PROF_LDFLAGS@ >> ./Config.Project.in
echo PKG_SHARED_LDFLAGS=@PKG_SHARED_LDFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for linking libraries >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_LDLIBFLAGS=@CONFIGURE_LDLIBFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Package specific linker settings >> ./Config.Project.in
echo PKG_CHECK_LDLIBFLAGS=@PKG_CHECK_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_DEBUG_LDLIBFLAGS=@PKG_DEBUG_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_DEFAULT_LDLIBFLAGS=@PKG_DEFAULT_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_GLOBAL_LDLIBFLAGS=@PKG_GLOBAL_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_GPROF_LDLIBFLAGS=@PKG_GPROF_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_OPT_LDLIBFLAGS=@PKG_OPT_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_PROF_LDLIBFLAGS=@PKG_PROF_LDLIBFLAGS@ >> ./Config.Project.in
echo PKG_SHARED_LDLIBFLAGS=@PKG_SHARED_LDLIBFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Specific linker functionality >> ./Config.Project.in
echo >> ./Config.Project.in
if ( test x$USER_LIBPATHSWITCH != x )
then
  AC_SUBST([USER_LIBPATHSWITCH])
  echo LINKER_LIBPATHSWITCH=@USER_LIBPATHSWITCH@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi
if ( test x$USER_with_rpath != x )
then
  AC_SUBST([USER_with_rpath])
  echo USE_LIBPATHSWITCH=@USER_with_rpath@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi
if ( test x$USER_UNDEFSYMB != x )
then
  AC_SUBST([USER_UNDEFSYMB])
  echo LINKER_UNDEFSYMB=@USER_UNDEFSYMB@ >> ./Config.Project.in
  echo >> ./Config.Project.in
fi
echo >> ./Config.Project.in
echo >> ./Config.Project.in
# Only write out the NVCC flags if we have found the compiler driver
if ( test x$use_NVCC = xpresent )
then
   echo \# CUDA compiler settings >> ./Config.Project.in
   echo >> ./Config.Project.in
   echo \# User specified at configure time >> ./Config.Project.in
   if ( test "x$CONFIGURE_NVCCFLAGS" = x )
   then
      # No NVCCFLAGS specified on the configure, use any from QMake
      echo CONFIGURE_NVCCFLAGS=\$\(QMAKE_NVCCFLAGS\) >> ./Config.Project.in
   else
      echo CONFIGURE_NVCCFLAGS=@CONFIGURE_NVCCFLAGS@ >> ./Config.Project.in
   fi
   echo >> ./Config.Project.in
   echo \# Settings determined for current combination of C++ and CUDA compilers >> ./Config.Project.in
   if ( test "x$DEFAULT_NVCC_TARGS" = x )
   then
      echo PKG_NVCC_TARGS=\$\(QMAKE_NVCC_TARGS\) >> ./Config.Project.in
   else
      echo PKG_NVCC_TARGS=@DEFAULT_NVCC_TARGS@ >> ./Config.Project.in
   fi
   echo PKG_NVCCANSIFLAG=@PKG_NVCCANSIFLAG@ >> ./Config.Project.in
   echo PKG_CHECK_NVCCFLAGS=@PKG_CHECK_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_DEBUG_NVCCFLAGS=@PKG_DEBUG_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_DEFAULT_NVCCFLAGS=@PKG_DEFAULT_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_GLOBAL_NVCCFLAGS=@PKG_GLOBAL_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_GPROF_NVCCFLAGS=@PKG_GPROF_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_OPT_NVCCFLAGS=@PKG_OPT_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_PROF_NVCCFLAGS=@PKG_PROF_NVCCFLAGS@ >> ./Config.Project.in
   echo PKG_SHARED_NVCCFLAGS=@PKG_SHARED_NVCCFLAGS@ >> ./Config.Project.in
   echo >> ./Config.Project.in
   echo >> ./Config.Project.in
   AC_SUBST([CONFIGURE_NVCCFLAGS])
   AC_SUBST([DEFAULT_NVCC_TARGS])
fi
echo \# Flags for the archiver >> ./Config.Project.in
echo >> ./Config.Project.in
echo PKG_ARFLAGS=@PKG_ARFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Flags for make >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_MAKEFLAGS=@CONFIGURE_MAKEFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# General Ravl make settings >> ./Config.Project.in
echo PKG_MAKEFLAGS=@PKG_MAKEFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in


# Select if we build Ravl ready to install or runnable from the build area
echo \# Specify default for embedding paths into Ravl >> ./Config.Project.in
AC_SUBST([enable_fast_install])
echo USE_INSTALLED_PATHS=@enable_fast_install@ >> ./Config.Project.in
echo >> ./Config.Project.in


# Allow definition of RESOURCES
echo >> ./Config.Project.in
echo \# List of resources available to Ravl >> ./Config.Project.in
echo >> ./Config.Project.in
if ( test x$ResetRESOURCES == xyes )
then
   # We need to override the QMake base level of RESOURCES
   echo "RESOURCES=@RESOURCES@" >> ./Config.Project.in
else
   # We can supplement the QMake settings
   echo "RESOURCES+=@RESOURCES@" >> ./Config.Project.in
fi
echo >> ./Config.Project.in

# Declare where external executables used by Ravl may be found
#
echo >> ./Config.Project.in
echo \# List of external programs available to Ravl >> ./Config.Project.in
echo >> ./Config.Project.in
# Definition of Perl
echo "PERL=@PERL@" >> ./Config.Project.in
echo >> ./Config.Project.in
# Definition of SWIG
echo "SWIG=@SWIG@" >> ./Config.Project.in
echo >> ./Config.Project.in


# Apply any "rpath" setting to the user-configured libraries
#
if ( test "$with_rpath" = 'Y' )
then
  for lib in $ConfiguredLibs
  do
    # Virtual libraries pick up their configuration from their real
    # counterparts. We therefore only need to massage the _LIBS setting here
    # for non-virtual libraries.
    eval check=\$with_${lib}_check
    if ( test x$check != xvirtual )
    then
      eval lib_enablement=\$use_${lib}
      case $lib_enablement in
        'blocked' | 'missing' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} $lib_enablement
                   ;;
        'no' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} discarded
                   ;;
        'check' | 'yes' )
                   trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
                   ;;
        'present' )
                   eval setting=\$${lib}_LIBS
                   TRACE_INDENT="     "
                   trace 3 Applying rpath to $lib
                   TRACE_INDENT="     "
                   trace 5 Changing _LIBS from:
                   TRACE_INDENT="     "
                   trace 5 $setting
                   setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LINKER_LIBPATHSWITCH)\1 -L\1/g'`
dnl                # The sed command is actually:
dnl                #    's/-L *\([^ ][^ ]*\)\$(LINKER_LIBPATHSWITCH)\1 -L\1/g'
dnl                # the square brackets having been substituted to prevent
dnl                # m4 removing them. 
                   # The sed command looks for:
                   #
                   #        -L<optional spaces><something><space> 
                   # 
                   # if this pattern is found, each and every occurrence is
                   # replaced by the following:
                   #
                   #       $(LINKER_LIBPATHSWITCH)=<something> -L<something>
                   #
                   # <something> being the substring found in the original
                   # fragment.
                   TRACE_INDENT="     "
                   trace 5 to:
                   TRACE_INDENT="     "
                   trace 5 $setting
                   eval ${lib}_LIBS=\$setting
                   ;;
      esac
    fi
  done
fi

echo >> ./Config.Project.in
echo \# List of external libraries available to Ravl >> ./Config.Project.in

# We need to amend the rpath setting for the unconfigured libs if the user has
# changed the setting.
#
if ( test x$USER_with_rpath != x && test x`cat $QMAKE_CFG | $SED -n 's/^USE_LIBPATHSWITCH=\(.*\)/\1/p'` != x$USER_with_rpath)
then
  trace 3 Resetting rpath on un-configured external libraries
  for lib in $UnconfiguredLibs
  do
    # Virtual libraries pick up their configuration from their real
    # counterparts. We therefore only need to massage the _LIBS setting here
    # for non-virtual libraries.
    eval check=\$with_${lib}_check
    if ( test x$check != xvirtual )
    then
      eval lib_enablement=\$use_${lib}
      case $lib_enablement in
        'blocked' | 'missing' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} $lib_enablement
                   ;;
        'no' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} discarded
                   ;;
        'check' | 'yes' )
                   trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
                   ;;
        'present' )
                   eval setting=\$${lib}_LIBS
                   TRACE_INDENT="          "
                   trace 3 Considering ${lib}_LIBS 
                   TRACE_INDENT="               "
                   trace 5 $setting
                   if (test $with_rpath = Y)
                   then
                     # Add in new rpath setting
                     new_setting=`echo "$setting" | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LINKER_LIBPATHSWITCH)\1 -L\1/g'`
#dnl                 # The sed command is actually:
#dnl                 #    's/-L *\([^ ][^ ]*\)\$(LINKER_LIBPATHSWITCH)\1 -L\1/g'
#dnl                 # the square brackets having been substituted to prevent
#dnl                 # m4 removing them. 
                     # The sed command looks for:
                     #
                     #        -L<optional spaces><something><space> 
                     # 
                     # if this pattern is found, each and every occurrence is
                     # replaced by the following:
                     #
                     #       $(LINKER_LIBPATHSWITCH)<something> -L<something>
                     #
                     # <something> being the substring found in the original
                     # fragment.
                   else
                     # Strip existing setting
                     new_setting=`echo "$setting" | $SED 's/\$(LINKER_LIBPATHSWITCH)@<:@^ @:>@@<:@^ @:>@* //g'`
#dnl                 # The sed command is actually:
#dnl                 #    's/\$(LINKER_LIBPATHSWITCH)[^ ][^ ]* //g'
#dnl                 # the square brackets having been substituted to prevent
#dnl                 # m4 removing them. 
                     # The sed command looks for and deletes:
                     #
                     #        $(LINKER_LIBPATHSWITCH)<something><space> 
                     #
                   fi
                   if ( test "$setting" != "$new_setting" )
                   then
                     eval ${lib}_LIBS=\$new_setting
                     TRACE_INDENT="          "
                     trace 5 reset to
                     TRACE_INDENT="               "
                     trace 5 $new_setting
                     echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
                     echo >> ./Config.Project.in
                   fi
                   ;;
      esac
    fi
  done
fi


# Recalculate all virtual packages to pick up any changes made to the real
# libraries
for lib in $KnownLibs
do
  eval check=\$with_${lib}_check
  if ( test x$check = xvirtual )
  then
    eval lib_enablement=\$use_${lib}
    case $lib_enablement in
      'blocked' | 'missing' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} $lib_enablement
                   ;;
      'no' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} discarded
                   ;;
      'check' | 'yes' )
                   trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
                   ;;
      'present' )
                   TRACE_INDENT="     "
                   trace 3 Checking settings for virtual library $lib
                   cflags=""
                   expandedcflags=""
                   ccflags=""
                   expandedccflags=""
                   libs=""
                   expandedlibs=""
                   eval prereqs=\$with_${lib}_prq
                   for p in $prereqs
                   do
                     eval prereqcflags=\$${p}_CFLAGS
                     if ( test "X$prereqcflags" != X )
                     then
                       cflags="$cflags @${p}_CFLAGS@"
                       expandedcflags="$expandedcflags $prereqcflags"
                     fi
                     eval prereqccflags=\$${p}_CCFLAGS
                     if ( test "X$prereqccflags" != X )
                     then
                       ccflags="$ccflags @${p}_CCFLAGS@"
                       expandedccflags="$expandedccflags $prereqccflags"
                     fi
                     eval prereqlibs=\$${p}_LIBS
                     if ( test "X$prereqlibs" != X )
                     then
                       libs="$libs @${p}_LIBS@"
                       expandedlibs="$expandedlibs $prereqlibs"
                     fi
                     dnl Extend this section to process the Fortran make flags
                     dnl if we ever add Fortran support
                   done
                   need_newline=no
                   if ( test x`$EXPR " $UnconfiguredLibs " : ".* \($lib\) .*"` == "x$lib" )
                   then
                     # We have not directly configured this package, but may
                     # have implicitly changed its settings. See if we need
                     # to output the new settings...
                     original_cflags=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CFLAGS=\(.*\)/\1/p"`
                     if ( test "X$original_cflags" != "X$expandedcflags" )
                     then
                       TRACE_INDENT="     "
                       trace 5 Changing CFLAGS from:
                       TRACE_INDENT="     "
                       trace 5 $original_cflags
                       TRACE_INDENT="     "
                       trace 5 to:
                       TRACE_INDENT="     "
                       trace 5 $expandedcflags
                       echo ${lib}_CFLAGS=$cflags >> Config.Project.in
                       need_newline=yes
                     fi
                     original_ccflags=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CCFLAGS=\(.*\)/\1/p"`
                     if ( test "X$original_ccflags" != "X$expandedccflags" )
                     then
                       TRACE_INDENT="     "
                       trace 5 Changing CCFLAGS from:
                       TRACE_INDENT="     "
                       trace 5 $original_ccflags
                       TRACE_INDENT="     "
                       trace 5 to:
                       TRACE_INDENT="     "
                       trace 5 $expandedccflags
                       echo ${lib}_CCFLAGS=$ccflags >> Config.Project.in
                       need_newline=yes
                     fi
                     original_libs=`cat $QMAKE_CFG | $SED -n "s/^${lib}_LIBS=\(.*\)/\1/p"`
                     if ( test "X$original_libs" != "X$expandedlibs" )
                     then
                       TRACE_INDENT="     "
                       trace 5 Changing LIBS from:
                       TRACE_INDENT="     "
                       trace 5 $original_libs
                       TRACE_INDENT="     "
                       trace 5 to:
                       TRACE_INDENT="     "
                       trace 5 $expandedlibs
                       echo ${lib}_LIBS=$libs >> Config.Project.in
                       need_newline=yes
                     fi
                     dnl Extend this section to process the Fortran make flags
                     dnl if we ever add Fortran support
                     if ( test $need_newline == yes )
                     then
                       echo >> ./Config.Project.in
                     fi
                   else
                     # All configured packages have their settings output
                     # below. We just need to make sure the latest settings
                     # are copied across from the dependants of the virtual
                     # package
                     eval ${lib}_CFLAGS=\$expandedcflags
                     eval ${lib}_CCFLAGS=\$expandedccflags
                     eval ${lib}_LIBS=\$expandedlibs
                     dnl Extend this section to nullify the relevant Fortran
                     dnl make flags if we ever add Fortran support
                   fi
                   ;;
    esac
  fi
done


# Now set RESOURCES and output each external library's <lib>_CFLAGS and 
# <lib>_LIBS definition to Config.Project for those libraries the user has
# configured
#
echo >> ./Config.Project.in
trace 2 Calculating RESOURCES settings from library availability
trace 3 \(Initial value of RESOURCES "@<:@$RESOURCES@:>@"\)
for lib in $ConfiguredLibs
do
   eval lib_enablement=\$use_${lib}
   case $lib_enablement in
    'blocked' | 'missing' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} $lib_enablement
                   ;;
    'no' )
                   TRACE_INDENT="     "
                   trace 3 ${lib} discarded
                   ;;
    'check' | 'yes' )
                   trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
                   ;;
    'present' )
                   TRACE_INDENT="     "
                   trace 3 RESOURCES+=${lib}
                   RESOURCES="$RESOURCES $lib "
                   # Put the placeholders into Config.Project.in so we record the
                   # flags associated with the libraries that we can use
                   echo ${lib}_CFLAGS=@${lib}_CFLAGS@ >> Config.Project.in
                   eval ccflags=\$${lib}_CCFLAGS
                   if ( test "X$ccflags" != X )
                   then
                      # Output any CCFLAGS
                      echo ${lib}_CCFLAGS=@${lib}_CCFLAGS@ >> Config.Project.in
                   else
                      # No CCFLAGS - check we don't have any to dis-inherit
                      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CCFLAGS=\(.*\)/\1/p"`
                      if ( test "X$QMakeSetting" != X )
                      then
                         # QMake config provides CCFLAGS so we need to nullify
                         # the old setting
                         echo ${lib}_CCFLAGS= >> Config.Project.in
                      fi
                   fi
                   echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
                   echo >> ./Config.Project.in
                   dnl Extend this section to write out the Fortran make flags
                   dnl if we ever add Fortran support
                   ;;
   esac
done
trace 3 RESOURCES now "@<:@$RESOURCES@:>@"

echo >> ./Config.Project.in
echo >> ./Config.Project.in

# Process platform specific dependencies that are not directly driven by a 
# third-party library nor the programming tools in use.


dnl *****         DO NOT PUT COMPILER SPECIFIC SETTINGS HERE         *****
dnl ***** They should go in the compiler settings section earlier in *****
dnl ***** this script. The only settings here are for those that are *****
dnl ***** for common tools but only on a specific platform. If using *****
dnl ***** a common development tool on a particular platform needs a *****
dnl ***** specific setting, set it here; if you are using the native *****
dnl ***** compiler on your platform, set its switches in the section *****
dnl ***** earlier in this script.                                    *****


# Test host_os != linux*
#if ( test  `echo X$host_os | $SED 's/-.*$//'` != 'Xlinux' )
if ( test  X`$EXPR X$host_os : 'X\(linux\).*'` != 'Xlinux' )
then
   trace 3 Host OS is $host_os

   case X$host_os in
   Xarm )       # Not sure what sure what this platform actually was, but
                # include some settings as a means to record the historic data.
                # Certainly, this case will not get activated as the OS is not
                # going to be "arm". Most likely encounter with arm from now on
                # will be as a cpu type on linux and going forward and there is
                # no real reason why we would always kill some functionality if
                # we are running on arm. On some particular installations maybe
                # (due to resource considerations) but globally, no.
                trace 3 Amending compile settings for ARM
                # Probably would not want to do this today, but we used to... 
                # unroll loops and allow strict-aliasing optimisations (drop
                # the -fno-strict-aliasing flag)
                PKG_OPT_CFLAGS=`echo " $PKG_OPT_CFLAGS -funroll-loops" | $SED 's/ -fno-strict-aliasing / /'`
                PKG_OPT_CCFLAGS=`echo " $PKG_OPT_CCFLAGS -funroll-loops" | $SED 's/ -fno-strict-aliasing / /'`
                ;;
   Xcygwin )    trace 3 Amending compile settings for Cygwin
                # Drop -Wall from {C}CFLAGS
                PKG_GLOBAL_CFLAGS=`echo " $PKG_GLOBAL_CFLAGS " | $SED 's/ -Wall / /'`
                PKG_GLOBAL_CCFLAGS=`echo " $PKG_GLOBAL_CCFLAGS " | $SED 's/ -Wall / /'`
                # Drop -fPIC from {C}CFLAGS
                PKG_SHARED_CFLAGS=`echo " $PKG_SHARED_CFLAGS " | $SED 's/ -fPIC / /'`
                PKG_SHARED_CCFLAGS=`echo " $PKG_SHARED_CCFLAGS " | $SED 's/ -fPIC / /'`
                # .exe suffix
                PKG_EXEEXT=.exe#
                # Remember, source relies on __CYGWIN__ being set by the 
                # compiler; update the source or manually define this 
                # variable if this ceases to be the case
                ;;
   Xirix* )     # It is probable that this platform will no longer build 
                # correctly, configuration is really a starting point based
                # on old compiler settings. Both the Ravl source and any
                # platform compiler may well have moved on from these
                # settings being useful
                trace 3 Amending compile settings for SGI
                # Add s flag on ar
                PKG_ARFLAGS=crus
                # -M instead of -MM (hope that was not a typo)
                PKG_MKDEPFLAGS=-M
                # Code also relies on the compiler in use setting both __sgi__
                # and __mips. If this is not the case, you should manually set
                # these. 
                ;;
   Xosf* )      # Again, this is possibly a platform that will no longer build
                # correctly and the configuration is really a starting point
                # based on old compiler settings. This is due to both the Ravl
                # source and any platform compiler having evolved from when
                # when the last port was carried out
                trace 3 Amending compile settings for DEC Alpha
                # Drop -pipe fron {C}CFLAGS
                PKG_SHARED_CFLAGS=`echo " $PKG_SHARED_CFLAGS " | $SED 's/ -pipe / /'`
                PKG_SHARED_CCFLAGS=`echo " $PKG_SHARED_CCFLAGS " | $SED 's/ -pipe / /'`
                # Add s flag on ar
                PKG_ARFLAGS=rusc
                # Code also relies on the compiler in use setting both __osf__
                # and __alpha. If this is not the case, you should manually set
                # these or update the Ravl code. 
                ;;
   Xsolaris* )  # No specific options set here (all platform specific items come
                # from the toolset so are set earlier in this script). This is
                # really here just as a place-holder and to note  that the code
                # relies on the compiler appropriately setting the relevant
                # defines from the list:__sun; __SUNPRO_CC; __sparc_v9; __sparc
                # __sol2_7__ & __sol2_9__. If this is no longer the case, modify
                # Ravl as needed or manually define the relevant switch.
   esac

else
   # Linux 

   # Process Architecture specific issues
   trace 3 Host OS is $host_os, CPU is $host_cpu
   case X$host_cpu in
   Xi386* | Xi486* | Xi586* | Xi686* )
                 trace 3 Non-SSE 32-bit x86
                 # Case deliberately empty as no special processing is needed
                 # for these chips, but the case is needed to prevent the
                 # default actions being carried out
                 ;;
   Xi786* )      # i786 equates to Intel P4. Even Athlon XPs come back as i686
                 # from config.guess/sub

                 trace 3 Enabling SSE2 support as we are on a P4.
                 RESOURCES="$RESOURCES USE_SSE2 "
                 ;;
   Xx86_64* )    # 64 bit 'x86' architecture - includes both AMD & Intel but
                 # not Itanium

                 trace 3 64-bit x86 supports SSE2
                 RESOURCES="$RESOURCES USE_SSE2 "
                 ;;
   Xalpha* )     trace 3 DEC/Compaq/HP Alpha platform
                 # Case deliberately empty as no special processing is needed
                 # for these machines. The case is needed to differentiate
                 # this platform from the default.
                 ;;
   XNone-at-the-moment )
                 # There's no real difference between these architectures and
                 # the default case. This just makes a good place to list
                 # platforms we have actually used in the wild. The code here
                 # should be the same as the final default case (minus slight
                 # variations in the trace messages to allow execution path
                 # confirmation when debugging).
                 
                 trace 3 New platform
                 ;;
    X* )         # Catch-all for previously unseen platforms. Processing should
                 # be exactly the same as the above case for default platforms
                 # we have seen before. (minus a slight difference in the trace
                 # messages to see which path has been taken if debugging).

                 trace 3 Default platform
                 ;;
   esac

   # True for all Linuxes
fi

dnl Declare RESOURCES for output
AC_SUBST([RESOURCES])
if ( test x$ResetRESOURCES == xyes )
then
   # We have already determined that we are disabling resources enabled by
   # the QMake configuration. We are thus outputting a full redefinition
   # of RESOURCES which needs to include:
   #    + those resources already enabled in the QMake configuration minus
   #      any that have been configured out by this configure run (this is
   #      the current value of QMAKE_RESOURCES)
   #    + those resources configured in by this configure run (the current
   #      value of RESOURCES.
   # If we have configured in a resource that was both configured in by the
   # QMake configuration and this run, we will have the resource listed in
   # both variables. We should therefore delete the duplicate(s) before
   # creating our final resource list.
   for lib in $QMAKE_RESOURCES
   do
      RESOURCES=`echo " $RESOURCES "  | $SED "s/ $lib / /g"`
   done
   # We could have applied this logic if we are not resetting RESOURCES (i.e.
   # when we are only adding extra resources to those configured in at the
   # QMake configuration stage). However, this would make items specifically
   # configured at the project level subject to any subsequent reconfiguration
   # of QMake which is probably not what the use would require.
   RESOURCES=`echo "$QMAKE_RESOURCES $RESOURCES" | sed "s/  */ /g"`
   # Set final value for RESOURCES removing any multiple spaces
fi



dnl Create Makefiles and *.Project files
AC_CONFIG_FILES([../../Config.Project:Config.Project.in])
AC_CONFIG_FILES([../../Targets.Project:Targets.Project.in])
AC_CONFIG_FILES([../../Makefile:Makefile.in])
AC_OUTPUT
