dnl#JFi:ToDo-Check for ver 5 of Perl as a requirement
dnl#JFi:ToDo-Change --with-<lib>=path to imply $path/lib and $path/include
dnl#JFi:ToDo-Extend --with-<lib>[-{library,include}] to allow multipe entries
dnl#JFi:ToDo-Maybe use AC_CONFIG_SRCDIR([include/ccmath/ccmath.h])
dnl#JFi:ToDo-Investigate position of 32 v 64 bit lib paths
dnl#JFi:ToDo-Investigate position of shared v static pkg-config results
dnl#JFi:ToDo-Extend --with-${lib} to specify location/name of .pc
dnl#JFi:ToDo-Extend --with-${lib}-pkgconf to specify path for known .pc

# Initialisation
AC_PREREQ([2.65])

AC_INIT([RAVL], [1.1.3], [ravl-devel@lists.sourceforge.net], [Ravl], [http://www.ee.surrey.ac.uk/CVSSP/Ravl])
dnl QMake doesn't use PACKAGE_TARNAME, it uses a hard-coded RAVL to set the 
dnl name of the dist archive/dir. However, the tarname parameter to AC_INIT
dnl (the 4th item) is used to prime the default for --docdir hence the value
dnl used here.

# This file is part of RAVL, Recognition And Vision Library
# Copyright (C) 2011-13, University of Surrey
# This code may be redistributed under the terms of the GNU General
# Public License (GPL). See the gpl.licence file for details or
# see http://www.gnu.org/copyleft/gpl.html


dnl Include standard QMake autoconf macros
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CHOOSE_GETCWD])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CHOOSE_RPATH])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_CK_COMPILE_CK_LIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_COMPILE_CK_FOR_LIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_Debug_Fn])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_ALL_EXTLIBS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_CC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_EXTLIB_PC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_DEFINE_VIRTUAL_EXTLIB])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_KEY_TOOLS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_LD])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_MAKE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_LOCATE_QMAKE])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_PROCESS_EXTLIBS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_ARC])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_ARCHIVER_ARFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_COMPILER_CFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_COMPILER_CCFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_LINKER_LDFLAGS])
m4_include([../../QMake/src/AutoTools/AcMacros/QM_SET_MAKE_MAKEFLAGS])



# Reset list of external libraries
#
echo > ./ExtPkgSupp.$$
dnl# This file is used to prime Config.Project with the initialisation of the 
dnl# make variables needed to build the single library. It should be appended
dnl# to when the configure script declares each external library's XXX_SUPPORT
dnl# variable. Towards the end of the configure process, this file is sorted
dnl# and appended to the Config.Project file.
dnl#
dnl# Whenever you declare a new optional library/library-set in SingleLib.mk,
dnl# you need to add both the correponding AC_SUBST call to this script and add
dnl# the initialisation of the make variable to the ExtPkgSupp.$$ file. For
dnl# example, To add the support code for a new external library, say LibPKG,
dnl# the following lines need to be added to this configure script:
dnl#
dnl#      AC_SUBST([LibPKG_SUPPORT)
dnl#      echo LibPKG_SUPPORT=@LibPKG_SUPPORT@ >> ./ExtPkgSupp.$$
dnl#
dnl# For this example, you would use $(LibPKG_SUPPORT) in SingleLib.mk to
dnl# add the new support code to the single library.




# Main body of configure script


# First determine the current platform
AC_CANONICAL_HOST

# Locate the tools needed to run this script (locate QMake; shell utils;
# compilers etc.)
QM_LOCATE_KEY_TOOLS


   



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

# Declare the Ravl libraries that are dependant on the particular external
# libraries
dnl Modern versions of FireWire support currently unsupported by Ravl. Drop
dnl completely for now. Would still work for the old depreciated versions of
dnl FireWire (rev 1.X of the libs, Linux kernel <2.6.37)
dnl
dnl Uncomment the next line to re-add libdc1394
dnl libdc1394_RavlLib=libRavlImgIO1394dc
dnl
AC_SUBST([libdc1394_SUPPORT])
dnl On re-instatement, drop the following 2 lines 
libdc1394_SUPPORT=""
BrokenLibs="$BrokenLibs libdc1394"
echo libdc1394=@libdc1394_SUPPORT@ >> ./ExtPkgSupp.$$
dnl
dnl Similarly, uncomment the next line to re-instate libavc1394
dnl libavc1394_RavlLib=libRavlFireWire
dnl
AC_SUBST([libavc1394_SUPPORT])
dnl and drop the following 2 lines on restore
BrokenLibs="$BrokenLibs libavc1394"
libavc1394_SUPPORT=""
echo libavc1394=@libavc1394_SUPPORT@ >> ./ExtPkgSupp.$$
dnl
libAudioFile_RavlLib=libRavlAudioFile
AC_SUBST([libAudioFile_SUPPORT])
dnl Modern versions of libAVIFile currently unsupported by Ravl. Drop for now:
dnl To re-instate, replace these 3 lines
AC_SUBST([avifile_SUPPORT])
avifile_SUPPORT=""
BrokenLibs="$BrokenLibs avifile"
dnl with the un-commented version of these 2
dnl AC_SUBST([avifile_SUPPORT])
dnl avifile_RavlLib=libRavlAVIFile
echo avifile_SUPPORT=@avifile_SUPPORT@ >> ./ExtPkgSupp.$$
dnl
libCurl_RavlLib=libRavlURLIO
AC_SUBST([libCurl_SUPPORT])
libDV_RavlLib=libRavlDV
AC_SUBST([libDV_SUPPORT])
AC_SUBST([FFMPEGCDEC_SUPPORT])
AC_SUBST([FFMPEGCONT_SUPPORT])
AC_SUBST([FFMPEGSWS_SUPPORT])
AC_SUBST([FFMPEGUTIL_SUPPORT])
dnl     Ravl wants FFMPEG as only a single entity (libffmpeg). To support this,
dnl     this script will (later) define libffmpeg as a virtual package
iksemel_RavlLib=libRavlXMPPIksemel
AC_SUBST([iksemel_SUPPORT])
libglade2_RavlLib=libRavlLibGlade
AC_SUBST([libglade2_SUPPORT])
LibGnome_RavlLib=libRavlGUIGnome
AC_SUBST([LibGnome_SUPPORT])
AC_SUBST([libGTK2_SUPPORT])
dnl     The absence of libGTK2 will block the inclusion of some other libraries'
dnl     Ravl functionality (e.g. libRavlLibGlade, libRavlGUIGnome, etc.) but is
dnl     not solely responsible for the inclusion any Ravl libraries itself.
AC_SUBST([GTKGLExt_SUPPORT])
dnl     GTKGLExt does not relate directly to a Ravl library but it does add more
dnl     functionality to GUI3D and its absence blocks libRavlGUIOpenSceneGraph.
Loudmouth_RavlLib=libRavlXMPPLoudmouth
AC_SUBST([Loudmouth_SUPPORT])
OpenCV_RavlLib=libRavlOpenCV
AC_SUBST([OpenCV_SUPPORT])
AC_SUBST([libPNG_SUPPORT])
PortAudio_RavlLib=libRavlPortAudio
AC_SUBST([PortAudio_SUPPORT])
AC_SUBST([libGL_SUPPORT])
AC_SUBST([libGLU_SUPPORT])
dnl     Ravl sometimes wants libGL and libGLU as a single entity (OpenGL),
dnl     this configure script facilitates this by setting up an OpenGL
dnl     virtual lib. Singly, neither libGL or libGLU will cause Ravl to loose
dnl     a library in their absence, hence no Ravl-lib definition (although
dnl     the absence of libGL can block libRavlGUIOpenSceneGraph)
AC_SUBST([OpenSceneGraph_SUPPORT])
dnl     OpenSceneGraph does not relate directly to a Ravl library but it is a
dnl     required resource in the cvsspSoftware internal extensions to Ravl.
OpenSceneGraphGTK_RavlLib=libRavlGUIOpenSceneGraph
AC_SUBST([OpenSceneGraphGTK_SUPPORT])
RLog_RavlLib=libRavlRLog
AC_SUBST([RLog_SUPPORT])
devVideo4Linux2_RavlLib=libRavlImgIOV4L2
AC_SUBST([devVideo4Linux2_SUPPORT])
ZeroMQ_RavLib=libRavlZmq
AC_SUBST([ZeroMQ_SUPPORT])
dnl 
dnl Now for those libraries needing manual compilation checks:
dnl
# GLUT not directly used by Ravl
AC_SUBST([GLUT_SUPPORT])
LibJasper_RavlLib=libRavlImgIOJasper
AC_SUBST([LibJasper_SUPPORT])
AC_SUBST([JPEG_SUPPORT])
LAPACK_RavlLib="libRavlLapack libRavlLapackWraps"
AC_SUBST([LAPACK_SUPPORT])
AC_SUBST([posixThreads_SUPPORT])
AC_SUBST([LibTIFF_SUPPORT])
Zlib_RavlLib=libRavlZLib
AC_SUBST([Zlib_SUPPORT])

dnl Virtual Pkgs
libffmpeg_RavlLib="libRavlLibFFmpeg"
AC_SUBST([libffmpeg_SUPPORT])
OpenGL_RavlLib="libRavlDPDisplay3d libRavlGUI3D"
AC_SUBST([OpenGL_SUPPORT])


dnl#  Following detail optional Ravl libs that depend on external libs not
dnl#  currently supported by the QMake configure script
dnl#
dnl#      dvdread_RavlLib=libRavlDVDRead
dnl#      libRavlDVDRead also inhibited by lack of libmpeg2
dnl#
dnl#      libClipStationPro_RavlLib=CSPDriver
dnl#
dnl#      LibGd_RavlLib=RavlImgGd
dnl#
dnl#      libmkl_RavlLib=RavlIntelMKL
dnl#
dnl#      libmcrypt_RavlLib=RavlCrypto
dnl#
dnl#      libmpeg2_RavlLib=RavlLibMPEG2
dnl#      Also inhibits libRavlDVDRead on its absence
dnl#
dnl#      LibPython_RavlLib=RavlPython
dnl#
dnl#      NumPy_RavlLib=RavlNumPy
dnl#  
dnl#      SwigPython_RavlLib="RavlPythonSwig RavlPythonSwig2"
dnl#  
dnl#
dnl# Settings for external libs that are more or less dead
dnl#
dnl#      libehs_RavlLib=RavlEHS
dnl#
dnl#      libGuppi_RavlLib="RavlDPGraph RavlGuppi"
dnl#
dnl#      libmpeg_RavlLib=RavlMPEG
dnl#
dnl#      uEyeSDK_RavlLib=RavlImgIOuEye
dnl#      Also need to set uEyeSDK_CFLAGS=-D__LINUX__
dnl#
dnl#      UUIdTheo_RavlLib=RavlUUId
dnl#  
dnl#
dnl# Settings for OS specific externals
dnl# 
dnl#      SGIVL_RavlLib=RavlDMedia
dnl#
dnl#      LibDirectShow_RavlLib=RavlDirectShow
dnl#
dnl#     posixRT - Currently no use by Ravl, hence no LibposixRT_RavlLib and no
dnl#     interim variable defines below
dnl#
dnl#   Mac OSX Support:
dnl#     Currently global MacOSX REQUIRES flag. 
dnl#
dnl#     RavlMacOSXVideoCapture depends on RavlMacOSXRunLoop and all libraries
dnl#     Already use MacOSXVideoCapture_SUPPORT later in this script, so this needs to be kept in mind
dnl#
dnl#      MacOSXFoundation_RavlLib=RavlMacOSXRunLoop
dnl#
dnl# In the meantime...
dnl#
dnl#    declare dummy <Lib>_SUPPORT substitution variables
dnl#    here to prevent the single lib makefile trying to load unsupported
dnl#    libraries
dnl#
AC_SUBST([libClipStationPro_SUPPORT])
libClipStationPro_SUPPORT=""
echo libClipStationPro_SUPPORT=@libClipStationPro_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([LibDirectShow_SUPPORT])
LibDirectShow_SUPPORT=""
echo LibDirectShow_SUPPORT=@LibDirectShow_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([dvdread_SUPPORT])
dvdread_SUPPORT=""
echo dvdread_SUPPORT=@dvdread_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libehs_SUPPORT])
libehs_SUPPORT=""
echo libehs_SUPPORT=@libehs_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([uEyeSDK_SUPPORT])
uEyeSDK_SUPPORT=""
echo uEyeSDK_SUPPORT=@uEyeSDK_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([LibGd_SUPPORT])
LibGd_SUPPORT=""
echo LibGd_SUPPORT=@LibGd_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libGuppi_SUPPORT])
libGuppi_SUPPORT=""
echo libGuppi_SUPPORT=@libGuppi_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libmkl_SUPPORT])
libmkl_SUPPORT=""
echo libmkl_SUPPORT=@libmkl_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([MacOSXRunLoop_SUPPORT])
MacOSXRunLoop_SUPPORT=""
echo MacOSXRunLoop_SUPPORT=@MacOSXRunLoop_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([MacOSXVideoCapture_SUPPORT])
MacOSXVideoCapture_SUPPORT=""
echo MacOSXVideoCapture_SUPPORT=@MacOSXVideoCapture_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libmcrypt_SUPPORT])
libmcrypt_SUPPORT=""
echo libmcrypt_SUPPORT=@libmcrypt_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libmpeg_SUPPORT])
libmpeg_SUPPORT=""
echo libmpeg_SUPPORT=@libmpeg_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([libmpeg2_SUPPORT])
libmpeg2_SUPPORT=""
echo libmpeg2_SUPPORT=@libmpeg2_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([LibPython_SUPPORT])
LibPython_SUPPORT=""
echo LibPython_SUPPORT=@LibPython_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([NumPy_SUPPORT])
NumPy_SUPPORT=""
echo NumPy_SUPPORT=@NumPy_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([SGIVL_SUPPORT])
SGIVL_SUPPORT=""
echo SGIVL_SUPPORT=@SGIVL_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([SwigPython_SUPPORT])
SwigPython_SUPPORT=""
echo SwigPython_SUPPORT=@SwigPython_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([UUIdTheo_SUPPORT])
UUIdTheo_SUPPORT=""
echo UUIdTheo_SUPPORT=@UUIdTheo_SUPPORT@ >> ./ExtPkgSupp.$$
dnl#     End of workaround for currently unsupported libraries
dnl# 

dnl# Declare <Lib>_SUPPORT substitution variables for old packages no longer
dnl# automatically configured in (usually due to upstream support rendering 
dnl#the Ravl coding less useful).
dnl# These external pacakges may still be enabled (on systems where they will 
dnl# correctly build/function) by manually adding the appropriate RESOURCE 
dnl# identifier to RESOURCES and setting the <Lib>_SUPPORT variable to the name
dnl# of the appropriate Ravl library (usually the wrapper lib).
dnl#
dnl# Original Video4Linux depreciated from modern kernels
AC_SUBST([devVideo4Linux_SUPPORT])
devVideo4Linux_SUPPORT=""
echo devVideo4Linux_SUPPORT=@devVideo4Linux_SUPPORT@ >> ./ExtPkgSupp.$$
dnl# To re-instate (on old kernels):
dnl#     REQUIRES+=devVideo4Linux  devVideo4Linux_SUPPORT=libRavlImgIOV4L
dnl#
dnl# Dropped Meteor as this really is historic support.
AC_SUBST([Meteor1_SUPPORT])
Meteor1_SUPPORT=""
echo Meteor1_SUPPORT=@Meteor1_SUPPORT@ >> ./ExtPkgSupp.$$
dnl# To re-instate (for linux only)
dnl#     REQUIRES+=Meteor1  Meteor1_SUPPORT=RavlImgIOMeteor1


# If debugging, output what libraries are defined
if ( test X$DBG != X && test $DBG -ge 3 )
then
   trace 3 Defined libraries
   for lib in $KnownLibs
   do
      eval testwith=\$with_${lib}_fn
      eval prereq=\$with_${lib}_prq
      eval lnklibs=\$with_${lib}_lnklibs
      eval loc=\$with_${lib}_libpath
      eval hdr=\$with_${lib}_hdr
      eval inc=\$with_${lib}_inc
      eval pkgconf=\$with_${lib}_pkgconf
      eval lib_enablement=\$use_${lib} 
      TRACE_INDENT="     "
      eval virtual=\$with_${lib}_virtual
      AS_IF([test X$virtual = Xyes],
            [# Virtual package
             trace 3 $lib:Virtual:$prereq:$lib_enablement
            ],
            [trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$lib_enablement
            ]
           )
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
      eval "${lib}_LIBS="
   else
      # Inherit QMake configuration settings
      eval "use_${lib}=present"
      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_CFLAGS=\(.*\)/\1/p"`
      eval "${lib}_CFLAGS=\$QMakeSetting"
      QMakeSetting=`cat $QMAKE_CFG | $SED -n "s/^${lib}_LIBS=\(.*\)/\1/p"`
      eval "${lib}_LIBS=\$QMakeSetting"
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
      eval "${lib}_LIBS="
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
      eval testwith=\$with_${lib}_fn
      eval prereq=\$with_${lib}_prq
      eval lnklibs=\$with_${lib}_lnklibs
      eval loc=\$with_${lib}_libpath
      eval hdr=\$with_${lib}_hdr
      eval inc=\$with_${lib}_inc
      eval pkgconf=\$with_${lib}_pkgconf
      eval lib_enablement=\$use_${lib} 
      TRACE_INDENT="     "
      eval virtual=\$with_${lib}_virtual
      AS_IF([test X$virtual = Xyes],
            [# Virtual package
             trace 3 $lib:Virtual:$prereq:$lib_enablement
            ],
            [trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$lib_enablement
            ]
           )
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


# Now start checking for the existance of all we need or can use

trace 2 Checking for necessary dependancies

# Perl required by QMake 
AC_PATH_PROG([PERL],[perl])
AS_IF([test "X$PERL" != X], 
      [RESOURCES+="PERL "],
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
      [ RESOURCES+="GnuPlot "],
      [AC_MSG_WARN([Cannot find a working copy of GNU Plot, some graph plotting functionality of Ravl requires this utility at run-time.])]
     )
AC_SUBST([GnuPlot])
TRACE_INDENT="     "
trace 3 RESOURCES now set to "@<:@$RESOURCES@:>@"


AC_PATH_PROG([SWIG],[swig])
AS_IF([test "X$SWIG" != X], [ RESOURCES+="Swig "])
AC_SUBST([SWIG])
TRACE_INDENT="     "
trace 3 RESOURCES now set to "@<:@$RESOURCES@:>@"


# Check for common system functions that appear in different system libraries 
# dependant on the platform in use

# dlopen is usually in libdl but can vary (e.g. for BSD it's in libc)
AC_SEARCH_LIBS([dlopen],
               [dl],
               [],
               [AC_MSG_FAILURE([Cannot find a working dlopen function])],
               []
              )
if ( test "X$ac_cv_search_dlopen" != "Xnone required" )
then
        DynLink_LIBS="$ac_cv_search_dlopen"
else
        DynLink_LIBS=""
fi
AC_SUBST([DynLink_LIBS])



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




#
# Create proforma Config.Project.in
#

# Create initial, empty file
echo > ./Config.Project.in

echo \# Ravl configuration file - regenerate using configure >> ./Config.Project.in
echo >> ./Config.Project.in

echo \# Declare current version of RAVL >> ./Config.Project.in
echo PROJECT_VERSION=@PACKAGE_VERSION@ >> ./Config.Project.in
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
fi

if ( test "x$CP" != x )
then
  AC_SUBST([CP])
  echo CP=@CP@ >> ./Config.Project.in
fi

if ( test "x$USER_CPP" != x )
then
  AC_SUBST([USER_CPP])
  echo CPP=@USER_CPP@ >> ./Config.Project.in
fi

if ( test "x$CHGRP" != x )
then
  AC_SUBST([CHGRP])
  echo CHGRP=@CHGRP@ >> ./Config.Project.in
fi

if ( test "x$CHMOD" != x )
then
  AC_SUBST([CHMOD])
  echo CHMOD=@CHMOD@ >> ./Config.Project.in
fi

if ( test "x$CHOWN" != x )
then
  AC_SUBST([CHOWN])
  echo CHOWN=@CHOWN@ >> ./Config.Project.in
fi

if ( test "x$USER_CUT" != x )
then
  AC_SUBST([USER_CUT])
  echo CUT=@USER_CUT@ >> ./Config.Project.in
fi

if ( test "x$USER_CXX" != x )
then
  AC_SUBST([USER_CXX])
  echo CXX=@USER_CXX@ >> ./Config.Project.in
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

if ( test "x$INSTALL" != x )
then
  AC_SUBST([INSTALL])
  echo INSTALL=@INSTALL@ >> ./Config.Project.in
fi

if ( test "x$INSTALL_DATA" != x )
then
  AC_SUBST([INSTALL_DATA])
  echo INSTALL_DATA=@INSTALL_DATA@ >> ./Config.Project.in
fi

if ( test "x$INSTALL_PROGRAM" != x )
then
  AC_SUBST([INSTALL_PROGRAM])
  echo INSTALL=@INSTALL_PROGRAM@ >> ./Config.Project.in
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
fi

if ( test "x$LEX" != x )
then
  AC_SUBST([LEX])
  echo LEX=@LEX@ >> ./Config.Project.in
fi

if ( test "x$LEXLIB" != x )
then
  AC_SUBST([LEXLIB])
  echo LEXLIB=@LEXLIB@ >> ./Config.Project.in
fi

if ( test "x$LN_S" != x )
then
  AC_SUBST([LN_S])
  echo LN_S=@LN_S@ >> ./Config.Project.in
fi

if ( test "x$USER_MAKE" != x )
then
  AC_SUBST([USER_MAKE])
  echo MAKE=@USER_MAKE@ >> ./Config.Project.in
fi

if ( test "x$MKDIR_P" != x )
then
  AC_SUBST([MKDIR_P])
  echo MKDIR_P=@MKDIR_P@ >> ./Config.Project.in
fi

if ( test "x$PAGER" != x )
then
  AC_SUBST([PAGER])
  echo PKG_PAGER=@PAGER@ >> ./Config.Project.in
fi

if ( test "x$RANLIB" != x )
then
  AC_SUBST([RANLIB])
  echo RANLIB=@RANLIB@ >> ./Config.Project.in
fi

if ( test "x$RM" != x )
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

if ( test "x$SYNC" != x )
then
  AC_SUBST([SYNC])
  echo SYNC=@SYNC@ >> ./Config.Project.in
fi

if ( test "x$TOUCH" != x )
then
  AC_SUBST([TOUCH])
  echo TOUCH=@TOUCH@ >> ./Config.Project.in
fi

if ( test "x$USER_TR" != x )
then
  AC_SUBST([USER_TR])
  echo TR=@USER_TR@ >> ./Config.Project.in
fi

if ( test "x$XARGS" != x )
then
  AC_SUBST([XARGS])
  echo XARGS=@XARGS@ >> ./Config.Project.in
fi

if ( test "x$YACC" != x )
then
  AC_SUBST([YACC])
  echo YACC=@YACC@ >> ./Config.Project.in
fi

echo >> ./Config.Project.in


# Compiler, etc. switches
echo >> ./Config.Project.in
echo \# Relevant Ravl build process flags >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Declare platform architecture >> ./Config.Project.in
echo ARC=@ARC@ >> ./Config.Project.in
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


# Declare which libraries are needed for particular system calls that can move
# about on different platforms
#
# First initialise ALL_EXTERNAL_xxxS here as we need to record these libraries
# as well as the external libs
ALL_EXTERNAL_LIBS=""
ALL_EXTERNAL_CFLAGS=""
echo >> ./Config.Project.in
echo \# Location of system calls that can vary their parent library >> ./Config.Project.in
echo >> ./Config.Project.in
# Location of dlopen
echo "DynLink_LIBS=@DynLink_LIBS@" >> ./Config.Project.in
echo >> ./Config.Project.in
ALL_EXTERNAL_LIBS+=" "$DynLink_LIBS
ALL_EXTERNAL_CFLAGS+=" "$DynLink_CFLAGS


# Apply any "rpath" setting to the user-configured libraries
#
if ( test "$with_rpath" = 'Y' )
then
  for lib in $ConfiguredLibs
  do
    # Virtual libraries pick up their configuration from their real
    # counterparts. We therefore only need to massage the _LIBS setting here
    # for non-virtual libraries.
    eval virtual=\$with_${lib}_virtual
    if ( test x$virtual != xyes )
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
                   setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LIBPATHSWITCH)\=\1 -L\1/g'`
dnl                # The sed command is actually:
dnl                #    's/-L *\([^ ][^ ]*\)\$(LIBPATHSWITCH)\=\1 -L\1/g'
dnl                # the square brackets having been substituted to prevent
dnl                # m4 removing them. 
                   # The sed command looks for:
                   #
                   #        -L<optional spaces><something><space> 
                   # 
                   # if this pattern is found, each and every occurrence is
                   # replaced by the following:
                   #
                   #       $(LIBPATHSWITCH)=<something> -L<something>
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
# if with_rpath != QMake_rpath | LIBPATHSWITCH changed
#   if QMake_rpath=no
#     # Add in new rpath setting
#     for lib in $UnconfiguredLibs
#     do
#       # Virtual libraries pick up their configuration from their real
#       # counterparts. We therefore only need to massage the _LIBS setting here
#       # for non-virtual libraries.
#       eval virtual=\$with_${lib}_virtual
#       if ( test x$virtual != xyes )
#       then
#         eval lib_enablement=\$use_${lib}
#         case $lib_enablement in
#           'blocked' | 'missing' )
#                      TRACE_INDENT="     "
#                      trace 3 ${lib} $lib_enablement
#                      ;;
#           'no' )
#                      TRACE_INDENT="     "
#                      trace 3 ${lib} discarded
#                      ;;
#           'check' | 'yes' )
#                      trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
#                      ;;
#           'present' )
#                      eval setting=\$${lib}_LIBS
#                      setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LIBPATHSWITCH)\=\1 -L\1/g'`
#dnl                   # The sed command is actually:
#dnl                   #    's/-L *\([^ ][^ ]*\)\$(LIBPATHSWITCH)\=\1 -L\1/g'
#dnl                   # the square brackets having been substituted to prevent
#dnl                   # m4 removing them. 
#                      # The sed command looks for:
#                      #
#                      #        -L<optional spaces><something><space> 
#                      # 
#                      # if this pattern is found, each and every occurrence is
#                      # replaced by the following:
#                      #
#                      #       $(LIBPATHSWITCH)=<something> -L<something>
#                      #
#                      # <something> being the substring found in the original
#                      # fragment.
#                      eval ${lib}_LIBS=\$setting
#                      echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
#                      echo >> ./Config.Project.in
#                      ;;
#         esac
#       fi
#     done
#   else
#     # Strip existing setting; add any new setting and output the amended
#     # _LIBS entry
#     for lib in $UnconfiguredLibs
#     do
#       # Virtual libraries pick up their configuration from their real
#       # counterparts. We therefore only need to massage the _LIBS setting here
#       # for non-virtual libraries.
#       eval virtual=\$with_${lib}_virtual
#       if ( test x$virtual != xyes )
#       then
#         eval lib_enablement=\$use_${lib}
#         case $lib_enablement in
#           'blocked' | 'missing' )
#                      TRACE_INDENT="     "
#                      trace 3 ${lib} $lib_enablement
#                      ;;
#           'no' )
#                      TRACE_INDENT="     "
#                      trace 3 ${lib} discarded
#                      ;;
#           'check' | 'yes' )
#                      trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
#                      ;;
#           'present' )
#                      # First remove the old setting
#                      setting=......
#                      if LIBPATHSWITCH changed
#                        # Add in new switch
#                        setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LIBPATHSWITCH)\=\1 -L\1/g'`
#dnl                     # The sed command is actually:
#dnl                     #    's/-L *\([^ ][^ ]*\)\$(LIBPATHSWITCH)\=\1 -L\1/g'
#dnl                     # the square brackets having been substituted to prevent
#dnl                     # m4 removing them. 
#                        # The sed command looks for:
#                        #
#                        #        -L<optional spaces><something><space> 
#                        # 
#                        # if this pattern is found, each and every occurrence is
#                        # replaced by the following:
#                        #
#                        #       $(LIBPATHSWITCH)=<something> -L<something>
#                        #
#                        # <something> being the substring found in the original
#                        # fragment.
#                      fi
#                      eval ${lib}_LIBS=\$setting
#                      echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
#                      echo >> ./Config.Project.in
#                      ;;
#         esac
#       fi
#     done
#   fi
# fi


# Recalculate all virtual packages to pick up any changes made to the real
# libraries
for lib in $KnownLibs
do
  eval virtual=\$with_${lib}_virtual
  if ( test x$virtual = xyes )
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
                   libs=""
                   eval prereqs=\$with_${lib}_prq
                   for p in $prereqs
                   do
                     cflags="$cflags @${p}_CFLAGS@"
                     libs="$libs @${p}_LIBS@"
                   done
                   need_newline=no
                   eval original_cflags=\$${lib}_CFLAGS
                   if ( test "$original_cflags" != "$cflags" )
                   then
                     TRACE_INDENT="     "
                     trace 5 Changing CFLAGS from:
                     TRACE_INDENT="     "
                     trace 5 $original_cflags
                     TRACE_INDENT="     "
                     trace 5 to:
                     TRACE_INDENT="     "
                     trace 5 $cflags
                     echo ${lib}_CFLAGS=$cflags >> Config.Project.in
                     need_newline=yes
                   fi
                   eval original_libs=\$${lib}_LIBS
                   if ( test "$original_libs" != "$libs" )
                   then
                     TRACE_INDENT="     "
                     trace 5 Changing LIBS from:
                     TRACE_INDENT="     "
                     trace 5 $original_libs
                     TRACE_INDENT="     "
                     trace 5 to:
                     TRACE_INDENT="     "
                     trace 5 $libs
                     echo ${lib}_LIBS=$libs >> Config.Project.in
                     need_newline=yes
                   fi
                   if ( test $need_newline == yes )
                   then
                     echo >> ./Config.Project.in
                   fi
                   ;;
    esac
  fi
done


# Now set RESOURCES and output each external library's <Lib>_CFLAGS and 
# <Lib>_LIBS definition to Config.Project for those libraries the user has
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
                   RESOURCES+="${lib} "
                   # Put the placeholders into Config.Project.in so we record the
                   # flags associated with the libraries that we can use
                   echo ${lib}_CFLAGS=@${lib}_CFLAGS@ >> Config.Project.in
                   echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
                   echo >> ./Config.Project.in
                   ;;
   esac
done
trace 3 RESOURCES now "@<:@$RESOURCES@:>@"

# Need to calculate ALL_EXTERNAL_xxxS based on the libraries that Ravl can
# use. We need to include not just those libraries specifically configured
# by the user, but also those already configured by any existing QMake
# configuration. 
#
for lib in $KnownLibs
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
                   eval liblist=\$${lib}_LIBS
                   TRACE_INDENT="     "
                   trace 3 ALL_EXTERNAL_LIBS += $liblist
                   ALL_EXTERNAL_LIBS+=" "$liblist
                   eval cflglist=\$${lib}_CFLAGS
                   TRACE_INDENT="     "
                   trace 3 ALL_EXTERNAL_CFLAGS += $cfglist
                   ALL_EXTERNAL_CFLAGS+=" "$cflglist
                   ;;
   esac
done
trace 3 ALL_EXTERNAL_LIBS now "@<:@$ALL_EXTERNAL_LIBS@:>@"
trace 3 ALL_EXTERNAL_CFLAGS now "@<:@$ALL_EXTERNAL_CFLAGS@:>@"

# finish Config.Project.in external libraries section with ALL_EXTERNAL_LIBS and
# ALL_EXTERNAL_CFLAGS def
echo ALL_EXTERNAL_LIBS=@ALL_EXTERNAL_LIBS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo ALL_EXTERNAL_CFLAGS=@ALL_EXTERNAL_CFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in


echo >> ./Config.Project.in
echo >> ./Config.Project.in

# Process Ravl libraries that depend on the presence of particular external
# libraries. We need to process all libraries that Ravl will use, not just
# those specifically configured by the user.
for lib in $KnownLibs
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
    'present' )    # Allow the dependant Ravl libs to be listed out to the makefile
                   eval liblist="\$${lib}_RavlLib"
                   eval ${lib}_SUPPORT=\"$liblist\"
                   trace 3 ${lib}_SUPPORT set to @<:@$liblist@:>@
                   ;;
   esac
   echo ${lib}_SUPPORT=@${lib}_SUPPORT@ >> ./ExtPkgSupp.$$
done


# Apply additional requirements that affect the <Lib>_SUPPORT variables (drop
# any libs that have unmet complex dependancies on other external libs or on
# the current platform, etc.)

# First define the names of the optional parts of Ravl that do not directly
# relate to an external library (so do not currently have an _SUPPORT 
# variable). Initially set the variables to the name(s) of the libraries 
# that may be included; we'll strip them out if dependencies dictate..
#
# As XXXX_SUPPORT relates to libraries that interface to a third-party libraries
# use the convention of XXXX_LIBS to represent a set of wholly internal Ravl
# libraries, or XXXX on its own to indicate a single optional internal library.

ParallelIO_SUPPORT=libPPDev
Ravl2dGUI_LIBS="libRavlGUIUtil libRavlGUI2D"
Ravl3d_LIBS="libRavl3DIO libRavl3D"
RavlAAM="libRavlAAMGUI libRavlAAM"
RavlAutoPort=""
# libRavlAutoPort not currently implemented
RavlCameraCal=libRavlCameraCal
RavlChartDetector=libRavlChartDetector
RavlClassWizard=libRavlClassWizard
RavlDataServer=libRavlDataServer
RavlDevAudio=libRavlDevAudio
RavlDPDisplay=libRavlDPDisplay
RavlExtImgIO=libRavlExtImgIO
RavlFace_LIBS="libMarkupTool libRavlFace"
RavlGUI=libRavlGUI
RavlImageProc=libRavlImageProc
RavlLogic_LIBS="libRavlLogicAgent libRavlLogicNLP libRavlLogic"
RavlNet=libRavlNet
RavlRawVidIO=""
# libRavlRawVidIO currently dropped
RavlVDF=""
# libRavlVDF not currently implemented
RavlVPlay=libRavlVPlay
RavlXMPP=libRavlXMPP
SerialIO_SUPPORT=libRavlSerialIO
# SerialIO is currently included regardless..



dnl# Declare the substitution variables for the optional parts of Ravl
AC_SUBST([ParallelIO_SUPPORT])
echo ParallelIO_SUPPORT=@ParallelIO_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([Ravl2dGUI_LIBS])
echo Ravl2dGUI_LIBS=@Ravl2dGUI_LIBS@ >> ./ExtPkgSupp.$$
AC_SUBST([Ravl3d_LIBS])
echo Ravl3d_LIBS=@Ravl3d_LIBS@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlAAM])
echo RavlAAM=@RavlAAM@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlAutoPort])
echo RavlAutoPort=@RavlAutoPort@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlCameraCal])
echo RavlCameraCal=@RavlCameraCal@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlChartDetector])
echo RavlChartDetector=@RavlChartDetector@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlClassWizard])
echo RavlClassWizard=@RavlClassWizard@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlDataServer])
echo RavlDataServer=@RavlDataServer@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlDevAudio])
echo RavlDevAudio=@RavlDevAudio@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlDPDisplay])
echo RavlDPDisplay=@RavlDPDisplay@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlExtImgIO])
echo RavlExtImgIO=@RavlExtImgIO@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlFace_LIBS])
echo RavlFace_LIBS=@RavlFace_LIBS@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlGUI])
echo RavlGUI=@RavlGUI@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlImageProc])
echo RavlImageProc=@RavlImageProc@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlLogic_LIBS])
echo RavlLogic_LIBS=@RavlLogic_LIBS@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlNet])
echo RavlNet=@RavlNet@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlRawVidIO])
echo RavlRawVidIO=@RavlRawVidIO@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlVDF])
echo RavlVDF=@RavlVDF@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlVPlay])
echo RavlVPlay=@RavlVPlay@ >> ./ExtPkgSupp.$$
AC_SUBST([RavlXMPP])
echo RavlXMPP=@RavlXMPP@ >> ./ExtPkgSupp.$$
AC_SUBST([SerialIO_SUPPORT])
echo SerialIO_SUPPORT=@SerialIO_SUPPORT@ >> ./ExtPkgSupp.$$



# Process those libraries dependant on external libs

if ( test X$use_libDV != 'Xpresent' )
then
   # Lack of libDV blocks 1394 AV/C functionality in Ravl
   libavc1394_SUPPORT=""
   trace 3 Lack of libDV blocking libavc1394
fi

if ( test X$use_libGTK2 != 'Xpresent' )
then
   # Lack of libGTK2 blocks lots of libraries:
   libDV_SUPPORT=""
     # Lack of libDV blocks 1394 AV/C functionality
     libavc1394_SUPPORT=""
   libglade2_SUPPORT=""
   OpenGL_SUPPORT=""
   OpenSceneGraphGTK_SUPPORT=""
   LibGnome_SUPPORT=""
   Meteor1_SUPPORT=""
   RavlGUI=""
      # Lack of RavlGUI blocks the following:
         libGuppi_SUPPORT=""
         RavlVDF=""
         RavlVPlay=""
         # Also indirectly blocks LibGnome, OpenGL, OpenSceneGraphGTK, Ravl2dGUI
         # and libglade2 via dependancy but they are already disabled for their
         # direct dependancy on GTK
   # 2d-GUI support blocked both directly by lack of GTK2 and by lack of RavlGUI
   Ravl2dGUI_LIBS=""
      # RavlAAMGUI blocked by both lack of libGTK2 and 2d-GUI support
      RavlAAM="libRavlAAM"
      # Lack of 2d-GUI would also block RavlVDF (blocked by RavlGUI2D missing) and
      # RavlVPlay (blocked by the absense of either of the 2d libs) if they hadn't
      # already been blocked by RavlGUI being omitted
   # DPDisplay Blocked by lack of GTK2 directly or indirectly by the lack of 
   # either RavlGUI2D or RavlGUI
   RavlDPDisplay=""
      # DPDisplay blocks the following:
         MacOSXVideoCapture_SUPPORT=""
         # Also blocks RavlVDF which is already blocked by missing 2d-GUI
      
   trace 3 Lack of libGTK2 blocking Glade Gnome Meteor1 OpenGL 2dGUI etc.
fi

if ( ( test X$use_JPEG != 'Xpresent' ) && ( test X$use_LibTIFF != 'Xpresent' ) && ( test X$use_libPNG != 'Xpresent' ) )
then
   # Lack of any image support renders libRavlExtImgIO useless
   RavlExtImgIO=""
      # Lack of RavlExtImgIO blocks RavlPythonSwig2
      SwigPython_SUPPORT=`echo " $SwigPython_SUPPORT " | $SED 's/libRavlPythonSwig2/ /'`
   trace 3 Lack of image support blocking RavlExtImgIO and hence RavlPythonSwig2
fi

if ( test X$use_libGL != 'Xpresent' )
then
   # Lack of libGL blocks OpenSceneGraphGTK functionality in Ravl
   OpenSceneGraphGTK_SUPPORT=""
   trace 3 Lack of libGL blocking OpenSceneGraphGTK
fi

if ( test X$use_GTKGLExt != 'Xpresent' )
then
   # Lack of GTKGLExt blocks OpenSceneGraphGTK functionality in Ravl and will
   # also alter, but not remove, libRavlGUI3D in its absence
   OpenSceneGraphGTK_SUPPORT=""
   trace 3 Lack of GTKGLExt blocking OpenSceneGraphGTK
fi

if ( test X$use_libmpeg2 != 'Xpresent' )
then
   # Lack of libmpeg2 blocks dvdread functionality in Ravl
   dvdread_SUPPORT=""
   trace 3 Lack of libmpeg2 blocking dvdread
fi

if ( test X$use_NumPy != 'Xpresent' )
then
   # Lack of NumPy blocks RavlPythonSwig2
   SwigPython_SUPPORT=`echo " $SwigPython_SUPPORT " | $SED 's/libRavlPythonSwig2/ /'`
   trace 3 Lack of NumPy blocking RavlPythonSwig2
fi

if ( test X$use_RLog != 'Xpresent' )
then
   # Lack of RLog blocks libRavlFace and libMarkupTool; PortAudio_SUPPORT;
   # and blocks XMPP functionality (which in turn blocks iksemel support)
   RavlFace_LIBS=""
   PortAudio_SUPPORT=""
   RavlXMPP=""
   iksemel_SUPPORT=""
   trace 3 Lack of RLog blocking RavlFace PortAudio, XMPP and iksemel
fi


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
   trace 3 Disallowing all linux-only libraries
   Meteor1_SUPPORT=""
   ParallelIO_SUPPORT=""
   libavc1394_SUPPORT=""
   RavlDevAudio=""
   devVideo4Linux_SUPPORT=""
   devVideo4Linux2_SUPPORT=""
   libDV_SUPPORT=""
   uEyeSDK_SUPPORT=""

   case X$host_os in
   Xarm )       # Not sure what sure what this platform actually was, but
                # include some settings as a means to record the historic data.
                # Certainly, this case will not get activated as the OS is not
                # going to be "arm". Most likely encounter with arm from now on
                # will be as a cpu type on linux and going forward and there is
                # no real reason why we would always kill some functionality if
                # we are running on arm. On some particular installations maybe
                # (due to resource considerations) but globally, no.
                trace 3 Dropping libRavlAAM and libRavlVPlay under Arm
                RavlAAM=""
                RavlVPlay=""
                trace 3 Amending compile settings for ARM
                # Probably would not want to do this today, but we used to... 
                # unroll loops and allow strict-aliasing optimisations (drop
                # the -fno-strict-aliasing flag)
                PKG_OPT_CFLAGS=`echo " $PKG_OPT_CFLAGS -funroll-loops" | $SED 's/ -fno-strict-aliasing / /'`
                PKG_OPT_CCFLAGS=`echo " $PKG_OPT_CCFLAGS -funroll-loops" | $SED 's/ -fno-strict-aliasing / /'`
                ;;
   Xcygwin )    trace 3 Dropping functionality unsupported under Cygwin
                # 3D Support disabled under Cygwin
                   RavlCameraCal=""
                   Ravl3d_LIBS=""
                      # absence of libRavl3D blocks the OpenGL and OpenSceneGraphGTK support functionality
                         OpenGL_SUPPORT=""
                         OpenSceneGraphGTK_SUPPORT=""

                # Image Processing disabled under Cygwin
                   RavlImageProc=""

                # Logic Processing disabled under Cygwin
                   RavlLogic_LIBS=""

                # Applications disabled under Cygwin
                   RavlClassWizard=""
                   RavlDataServer=""
                   RavlVDF=""
                   RavlVPlay=""

                # RavlNet unavailable under Cygwin
                   RavlNet=""
                      # Absence of libRavlNet blocks libRavlPythonSwig & libRavlPythonSwig2 by dependancy
                      SwigPython_SUPPORT=""
                trace 3 Amending compile settings for Cygwin
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
                # relies on the compiler appropriatly setting the relevant 
                # defines from the list:__sun; __SUNPRO_CC; __sparc_v9; __sparc
                # __sol2_7__ & __sol2_9__. If this is no longer the case, modify
                # Ravl as needed or manually define the relevant switch.
   esac

   if ( test X$host_os != 'Xsgi' )
   then
      trace 3 SGIVL is only provided on SGI
      SGIVL_SUPPORT=""
   fi

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
   Xi786* )
                 # i786 equates to Intel P4. Even Athlon XPs come back as i686
                 # from config.guess/sub

                 trace 3 Enabling SSE2 support as we are on a P4.
                 RESOURCES+="USE_SSE2 "
                 ;;
   Xx86_64* )  
                 # 64 bit 'x86' architecture - includes both AMD & Intel but
                 # not Itanium

                 trace 3 64-bit x86 supports SSE2
                 RESOURCES+="USE_SSE2 "
                 
                 trace 3 Disallowing Meteor1 on 64-bit x86 platforms
                 Meteor1_SUPPORT=""

                 # ParallelIO currently unsupported off 32-bit x86
                 trace 3 Disallowing ParallelIO on 64-bit x86 platforms
                 ParallelIO_SUPPORT=""
                 ;;
   Xalpha* )
                 # DEC/Compaq/HP Alpha kit
                 
                 trace 3 Disallowing Meteor1 on non 32-bit x86 platforms
                 Meteor1_SUPPORT=""

                 # ParallelIO currently unsupported off 32-bit x86
                 trace 3 Disallowing ParallelIO on non 32-bit x86 platforms
                 ParallelIO_SUPPORT=""
                 ;;
   XNone-at-the-moment )
                 # There's no real difference between these architectures and
                 # the default case. This just makes a good place to list
                 # platforms we have actually used in the wild. The code here
                 # should be the same as the final default case (minus slight
                 # variations in the trace messages to allow execution path
                 # confirmation when debugging).
                 
                 trace 3 Disallowing Meteor1 on non 32-bit x86 platforms
                 Meteor1_SUPPORT=""

                 # ParallelIO currently unsupported off 32-bit x86
                 trace 3 Disallowing ParallelIO on non 32-bit x86 platforms
                 ParallelIO_SUPPORT=""
                 ;;
    X* )
                 # Catch-all for previously unseen platforms. Processing should
                 # be exactly the same as the above case for default platforms
                 # we have seen before. (minus a slight difference in the trace
                 # messages to see which path has been taken if debugging).

                 trace 3 Disallowing Meteor1 on default platforms
                 Meteor1_SUPPORT=""

                 # ParallelIO currently unsupported off 32-bit x86
                 trace 3 Disallowing ParallelIO on default platforms
                 ParallelIO_SUPPORT=""
                 ;;
   esac

   # True for all Linuxes

   trace 3 SGIVL is not supported on Linux
   SGIVL_SUPPORT=""
fi

# Map Platform/OS into Ravl ARCs
QM_SET_ARC()
AC_SUBST([ARC])

# Add, to Config.Project.in, the initialisation of the make variables used by
# SingleLib.mk. These have been recorded in ExtPkgSupp.$$ as this script has
# been running; so, sort this file and append it with a suitable heading.
echo >> ./Config.Project.in
echo \# Optional Ravl sub-libraries that may or may not be included in the >> ./Config.Project.in
echo \# single library>> ./Config.Project.in
$SORT ./ExtPkgSupp.$$ >> ./Config.Project.in
rm ./ExtPkgSupp.$$
echo >> ./Config.Project.in
echo \# Ordered list of the libraries we want in libRavl.\*  >> ./Config.Project.in
echo SINGLE_RECIPE=\$\(NumPy_SUPPORT\) \$\(avifile_SUPPORT\) libRavlMPEG2 \$\(devVideo4Linux_SUPPORT\) \$\(libffmpeg_SUPPORT\) libRavlVideoIO \$\(dvdread_SUPPORT\) \$\(libmpeg2_SUPPORT\) \$\(libavc1394_SUPPORT\) \$\(libDV_SUPPORT\) \$\(RavlRawVidIO\) \$\(RavlExtImgIO\) \$\(MacOSXVideoCapture_SUPPORT\) \$\(MacOSXRunLoop_SUPPORT\) \$\(RavlVDF\) \$\(SwigPython_SUPPORT\) \$\(RavlDPDisplay\) \$\(Meteor1_SUPPORT\) libRavlAV \$\(RavlChartDetector\) \$\(RavlAAM\) \$\(RavlFace_LIBS\) libRavlTwoWayMap \$\(RavlImageProc\) \$\(uEyeSDK_SUPPORT\) \$\(libmpeg_SUPPORT\) \$\(OpenGL_SUPPORT\) libRavlImageXMLFactory \$\(libClipStationPro_SUPPORT\) libRavlImageIO \$\(LibJasper_SUPPORT\) \$\(OpenCV_SUPPORT\) \$\(RavlVPlay\) \$\(Ravl2dGUI_LIBS\) \$\(libdc1394_SUPPORT\) libRavlVideo \$\(LibGd_SUPPORT\) \$\(devVideo4Linux2_SUPPORT\) \$\(OpenSceneGraphGTK_SUPPORT\) \$\(Ravl3d_LIBS\) \$\(LibDirectShow_SUPPORT\) libRavlImage \$\(libAudioFile_SUPPORT\) \$\(RavlDevAudio\) libRavlAudioFeatures \$\(PortAudio_SUPPORT\) libRavlAudioIO libRavlAudioUtil \$\(iksemel_SUPPORT\) \$\(Loudmouth_SUPPORT\) \$\(RavlXMPP\) \$\(RLog_SUPPORT\) \$\(SGIVL_SUPPORT\) \$\(RavlDataServer\) libOptimisation \$\(RavlLogic_LIBS\) libRavlPatternRecIO libRavlKalmanFilter libRavlSVM libRavlGnuPlot libRavlPatternRec libRavlDataSet libRavlOSIO  \$\(ParallelIO_SUPPORT\) \$\(SerialIO_SUPPORT\) \$\(libGuppi_SUPPORT\) libRavlPlot \$\(RavlClassWizard\) libRavlCxxDoc \$\(RavlAutoPort\) libRavlAutoPort2 libRavlSourceTools \$\(LibPython_SUPPORT\) \$\(RavlNet\) \$\(libehs_SUPPORT\) libRavlDPMT \$\(UUIdTheo_SUPPORT\) \$\(LibGnome_SUPPORT\) \$\(libglade2_SUPPORT\) \$\(RavlGUI\) libRavlMathIO libRavlGeneticProgram libRavlGeneticOptimisation \$\(ZeroMQ_SUPPORT\) libRavlService libRavlXMLFactory \$\(libCurl_SUPPORT\) libRavlThreads libRavlOS libRavlIO \$\(libmcrypt_SUPPORT\) libRavlOptimise \$\(RavlCameraCal\) \$\(libmkl_SUPPORT\) \$\(LAPACK_SUPPORT\) libRavlMath libccmath libfann libArffIO \$\(Zlib_SUPPORT\) libRavlHMM libRavlCore  >> ./Config.Project.in
echo >> ./Config.Project.in
echo >> ./Config.Project.in


dnl Declare RESOURCES and ALL_EXTERNAL_xxx as outputs
AC_SUBST([RESOURCES])
AC_SUBST([ALL_EXTERNAL_LIBS])
AC_SUBST([ALL_EXTERNAL_CFLAGS])

dnl Create Makefiles and *.Project files
AC_CONFIG_FILES([../../Config.Project:Config.Project.in])
AC_CONFIG_FILES([../../Targets.Project:Targets.Project.in])
AC_CONFIG_FILES([../../Makefile:Makefile.in])
AC_OUTPUT
