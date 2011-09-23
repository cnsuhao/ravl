#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.

AC_PREREQ([2.65])

AC_INIT([RAVL], [1.5.8030], [John.Field@Surrey.ac.UK])

dnl#JFi:ToDo-Check for ver 5 of Perl as a requirement
dnl#JFi:ToDo-Change --with-<lib>=path to imply $path/lib and $path/include
dnl#JFi:ToDo-Extend --with-<lib>[-{library,include}] to allow multipe entries
dnl#JFi:ToDo-Maybe use AC_CONFIG_SRCDIR([include/ccmath/ccmath.h])
dnl#JFi:ToDo-Implement use of ./src/Core/Defs/*.def for pre-reqs
dnl#JFi:ToDo-Delay fatal error from libs not being present until all checked
dnl#JFi:ToDo-Investigate position of 32 v 64 bit lib paths
dnl#JFi:ToDo-Investigate position of shared v static pkg-config results
dnl#JFi:ToDo-Extend --with-${lib} to specify location/name of .pc
dnl#JFi:ToDo-Extend --with-${lib}-pkgconf to specify path for known .pc
dnl#JFi:ToDo-Implement use of existing QMake installation


dnl     Define service Macros
dnl
dnl
dnl
dnl AX_RAVL_HELPER_PC
dnl
dnl Macro to define all helper libs that rely on pkg-config to see if they
dnl exist. This macro will declare the --with-<lib> command-line switch and
dnl create the simple --help text. The command-line switch will support a
dnl Yes/No (upper, lower and sentence case accepted; plus Y,N) to explicitly 
dnl en/dis-able use of the library; or 'check' (C, Ck, or Check etc.) to ask
dnl to utilise the library if it is available (the default option). The macro
dnl will go on to declare the --with-<lib>-pkgconf switch that explicitly sets
dnl the pkg-config control file to use to check for the helper library. This
dnl switch should be set to the full path of the control file.
dnl
dnl Macro parameters are:
dnl       p1: Ravl name for library. This is used for naming the cmd-line
dnl           switch (LIB => --with-LIB, --with-LIB-pkgconf). It should be
dnl           the name Ravl expects the RESOURCES string to take to enable
dnl           the library's functionality. It will also be the name used
dnl           for the LIB_CFLAGS and LIB_LIBS used in the relevant .def
dnl           file and Config.Project. 
dnl       p2: Default name by which pkg-config knows the library.
dnl       p3: Name(s) of the Ravl libraries that use the external library
dnl
dnl Global variables used:
dnl
dnl       with_<Lib>                  Value of similarly named cmd-line switch
dnl                                   Enables (actually requires) the use of the
dnl                                   particular library. Can also be used to
dnl                                   specify the location of the library and
dnl                                   header files when the specific commandline
dnl                                   switches are not used (ideal if both file
dnl                                   types reside in the same directory)
dnl       with_<Lib>_pkgconf          Value of similarly named cmd-line switch
dnl                                   Specifies what name pkg-config will know
dnl                                   the library as. This variable receives
dnl                                   the default if the cmd-line switch isn't 
dnl                                   used but the library is declared as using
dnl                                   pkg-config
dnl       use_<Lib>                   Flag specifying useage of library,
dnl                                   primed initially by --with-<Lib> or
dnl                                   as part of a dependancy check. 
dnl       <Lib>_RavlLib               Receives the name(s) of the Ravl libraries
dnl                                   that rely on the external library
dnl       <Lib>_SUPPORT               Declared here, this variable is used to
dnl                                   export the names of the dependant Ravl libs
dnl                                   to the makefiles. This variable is set to
dnl                                   null here and later set to <Lib>_RavlLib
dnl                                   if the external library is present and
dnl                                   usable
dnl
AC_DEFUN([AX_RAVL_HELPER_PC],
         [AC_ARG_WITH([$1-pkgconf],
                      [AS_HELP_STRING([--with-$1-pkgconf],
                                      [Enable the $1 library package via pkg-config and declare the name by which the $1 library package is known to pkg-config]
                                     )
                      ],
                      [case $with_$1_pkgconf in
                        'n' | 'N' | 'no' | 'No' | 'NO' )
                                          use_$1=no
                                          trace 4 --with-$1-pkgconf disabling $1
                                          ;;
                        'y' |'Y' | 'yes' | 'Yes' | 'YES' ) 
                                          AS_IF([test X$2 != X],
                                                [with_$1_pkgconf=$2
                                                 use_$1=yes
                                                 trace 4 --with-$1-pkgconf set to default @<:@$with_$1_pkgconf@:>@
                                                ],
                                                [AC_MSG_FAILURE([Asking to configure $1 via pkg-config but configure is unaware of the name of the DOTpc metadata file])
                                                ]
                                               )
                                          ;;
                        * )               use_$1=yes
                                          trace 4 --with-$1-pkgconf set to @<:@$with_$1_pkgconf@:>@
                                          ;;
                       esac
                      ],
                      [AS_IF([test X$2 != X],
                             [with_$1_pkgconf=$2
                              use_$1=check
                              trace 4 No --with-$1-pkgconf specified using default @<:@$with_$1_pkgconf@:>@
                             ]
                            )
                      ]
                     )
          AC_ARG_WITH([$1],
                      [AS_HELP_STRING([--with-$1],
                                      [Enable functionality that relies on the $1 library @<:@default=check@:>@]
                                     )
                      ],
                      [case $with_$1 in
                        'n' | 'N' | 'no' | 'No' | 'NO' ) 
                                       trace 4 --with-$1 disabling $1 libraries
                                       AS_IF([test X$use_$1 = Xyes ],
                                             [AC_MSG_FAILURE([Asking to disable $1 by --with-$1 but --with-$1-pkgconf has specifically enabled it])
                                             ],
                                             [use_$1=no]
                                            )
                                       ;;
                        'y' |'Y' | 'yes' | 'Yes' | 'YES' ) 
                                       trace 4 --with-$1 enabling $1 libraries
                                       AS_IF([test X$use_$1 = Xno],
                                             [AC_MSG_FAILURE([Asking to enable $1 by --with-$1 but --with-$1-pkgconf has specifically disabled it])
                                             ],
                                             [use_$1=yes]
                                            )
                                       ;;
                        'c' | 'C' | 'ck'| 'Ck' | 'CK' | 'check' | 'Check' | 'CHECK' ) use_$1=check
                                       trace 4 --with-$1 checking for $1 libraries
                                       AS_IF([test X$use_$1 != Xcheck],
                                             [AC_MSG_WARN([--with-$1 asking to check for the libs but --with-$1-pkgconf setting over-rides this])
                                             ],
                                             [use_$1=check]
                                            )
                                       ;;
                        * )            AC_MSG_WARN([--with-$1 set to an unknown option, setting ignored])
                                       ;;
                       esac
                      ],
                      [trace 5 No --with-$1 specified]
                     )
          HelperLibs=$HelperLibs" "$1
          $1_SUPPORT=""
          AS_IF([ test "$3" = NA],
                [$1_RavlLib=""],
                [$1_RavlLib="$3"]
               )
          AC_SUBST([$1_CFLAGS])
          AC_SUBST([$1_LIBS])
          AC_SUBST([$1_SUPPORT])
          echo $1_SUPPORT=@$1_SUPPORT@ >> ./ExtPkgSupp.$$
         ]
        )

dnl AX_RAVL_HELPER_CC
dnl
dnl Macro to define all helper libs that require a manual, compilation, check
dnl to see if they are available . Will declare the --with-<lib> command-line
dnl switch and create the simple --help text. The command-line switch will 
dnl support simple Yes/No (upper, lower and sentence case accepted; plus Y,N)
dnl to explicitly en/dis-able use of the library; check (C, Ck, or Check etc.)
dnl to ask to utilise the library if available (the default option); and a path 
dnl to a directory where a specific version of the library resides. The macro
dnl will go on to declare the --with-<lib>-library and --with-<lib>-include
dnl cmd-line swithes (and their help text). These two switches accept a path to
dnl the relevant support file from the external library (the library and header
dnl files respectivly). The include path must point to a directory while the 
dnl path to the library may just be the library's name (minus extension, with
dnl or without the lib prefix); a path to the containing directory; or both
dnl aspects in the form 'path/name'. This method of library path specification
dnl also applies to --with-<lib>. The switch, --with-<lib>-pkgconf, is also
dnl created to allow the user to ask configure to use pkg-config with this
dnl particular helper package. The value of --with-<lib>-pkgconf must be set
dnl to the full path of the relevant pkg-config file.
dnl
dnl Macro parameters are:
dnl       p1: Ravl name for library. This is used for naming the cmd-line
dnl           switch (LIB => --with-LIB, --with-LIB-include, etc.). It 
dnl           should be the name Ravl expects the RESOURCES string to take
dnl           to enable the library's functionality in Ravl. It will also
dnl           be the name used for the LIB_CFLAGS and LIB_LIBS used in the
dnl           relevant .def file and Config.
dnl       p2: Name of function in that library for the test link
dnl       p3: Name of all prerequisite libraries that this one depends on
dnl           Specify as a space-separated list of Ravl names
dnl       p4: Name(s) of the library as specified on the link line i.e. in
dnl           -lXXXX -lYYYY form
dnl       p5: Location (if different from the system defaults) where the 
dnl           library file resides 
dnl       p6: Name of any header files provided by the library package. It is
dnl           usually sufficient to only test for one file unless it is known
dnl           that incomplete subsets of the header files are regularly found
dnl       p7: Location (if different from the system defaults) where the 
dnl           library header files reside 
dnl       p8: Name(s) of the Ravl libraries that use the external library
dnl
dnl Global variables used:
dnl
dnl       with_<Lib>                  Value of similarly named cmd-line switch
dnl                                   Enables (actually requires) the use of the
dnl                                   particular library. Can also be used to
dnl                                   specify the location of the library and
dnl                                   header files when the specific commandline
dnl                                   switches are not used (ideal if both file
dnl                                   types reside in the same directory)
dnl       with_<Lib>_include          Value of similarly named cmd-line switch
dnl                                   Specifies where the header files for the
dnl                                   library reside.
dnl       with_<Lib>_library          Value of similarly named cmd-line switch
dnl                                   Specifies where the library file of the
dnl                                   library resides. Can be used to alter the
dnl                                   name of the library to be used.
dnl       with_<Lib>_pkgconf          Value of similarly named cmd-line switch
dnl                                   Specifies what name pkg-config will know
dnl                                   the library as. 
dnl       use_<Lib>                   Flag specifying useage of library,
dnl                                   primed initially by --with-<Lib> or
dnl                                   as part of a dependancy check. 
dnl       with_<Lib>_libpath          Path to the location of the library
dnl       with_<Lib>_lnklibs          List of the library files (in -lXXXX form)
dnl       with_<Lib>_hdr              Name(s) of the library package's headers
dnl       with_<Lib>_inc              Path to the library's headers
dnl       with_<Lib>_fn               Function for the test link
dnl       with_<Lib>_prq              Space separated list of pre-requisites
dnl                                   of the library
dnl       <Lib>_RavlLib               Receives the name(s) of the Ravl libraries
dnl                                   that rely on the external library
dnl       <Lib>_SUPPORT               Declared here, this variable is used to
dnl                                   export the names of the dependant Ravl libs
dnl                                   to the makefiles. This variable is set to
dnl                                   null here and later set to <Lib>_RavlLib
dnl                                   if the external library is present and
dnl                                   usable
dnl
AC_DEFUN([AX_RAVL_HELPER_CC],
         [AC_ARG_WITH([$1-pkgconf],
                      [AS_HELP_STRING([--with-$1-pkgconf],
                                      [Enable the $1 library package via pkg-config and declare the name by which the $1 library package is know to pkg-config]
                                     )
                      ],
                      [case $with_$1_pkgconf in
                        'n' | 'N' | 'no' | 'No' | 'NO' )   use_$1=no
                                          trace 4 --with-$1-pkgconf disabling $1
                                          ;;
                        'y' |'Y' | 'yes' | 'Yes' | 'YES' ) 
                                          AC_MSG_FAILURE([Asking to configure $1 via pkg-config but configure is unaware of the name of the DOTpc metadata file])
                                          ;;
                        * )               use_$1=yes
                                          trace 4 --with-$1-pkgconf set to @<:@$with_$1_pkgconf@:>@
                                          ;;
                       esac
                      ],
                      [trace 4 No --with-$1-pkgconf specified continuing with manual checks]
                     )
          AS_IF([test X$with_$1_pkgconf = X],
                [AC_ARG_WITH([$1],
                             [AS_HELP_STRING([--with-$1],
                                             [Enable functionality that relies on the $1 library @<:@default=check@:>@]
                                            )
                             ],
                             [trace 4 Not using pkg-config for the $1 library package
                              case $with_$1 in
                              'n' | 'N' | 'no' | 'No' | 'NO' ) use_$1=no
                                       ;;
                              'y' |'Y' | 'yes' | 'Yes' | 'YES' ) use_$1=yes
                                       ;;
                              'c' | 'C' | 'ck'| 'Ck' | 'CK' | 'check' | 'Check' | 'CHECK' ) use_$1=check
                                       ;;
                              * ) if ( test -d $with_$1 ) 
                                  then
                                       # Parameter is the directory path
                                       if ( test X$with_$1_libpath != X && test X$with_$1_libpath != X$with_$1 )
                                       then
                                          AC_MSG_WARN([--with-$1 and --with-$1-library disagree,])
                                          AC_MSG_WARN([--with-$1 ignored. ])
                                          use_$1=yes
                                       else
                                          with_$1_libpath=$with_$1
                                          use_$1=yes
                                       fi
                                  else
                                       path = AS_DIRNAME([$with_$1])
                                       if ( test -d $path  )
                                       then
                                          #file=`basename $with_$1`
                                          file=`$EXPR "//$with_$1" : '.*/\([[^/]]*\)'`
                                          if ( test X$with_$1 = X$path/$file )
                                          then
                                             # Parameter is a file path
                                             # Does it have a '.a' extension
                                             #lib=`basename $file .a`
                                             lib=`$EXPR "//$file" : '\(.*\)\.[[^.]]a$'`
                                             if ( test X$lib != X$file  )
                                             then
                                                lib='-l'`echo $lib | $SED 's/^lib//'`
                                                AC_MSG_WARN([Specifying --with-$1 pointing to a .a will])
                                                AC_MSG_WARN([cause problems building with shared libraries. We will])
                                                AC_MSG_WARN([use $lib and LDFLAGS=$path instead. ])
                                             else
                                                #lib=`basename $file .so`
                                                # Does it have a '.so' extension
                                                lib=`$EXPR "//$file" : '\(.*\)\.[[^.]]so$'`
                                                if ( test X$lib != X$file )
                                                then
                                                   lib='-l'`echo $lib | $SED 's/^lib//'`
                                                   AC_MSG_WARN([Specifying --with-$1 pointing to a .so will])
                                                   AC_MSG_WARN([cause problems building with shared libraries. We will])
                                                   AC_MSG_WARN([use $lib and LDFLAGS=$path instead. ])
                                                else
                                                   # Parameter is path and name fragment
                                                   lib=-l`echo $lib | $SED 's/^lib//'`
                                                   trace 3 with_$1 asks to use $lib and LDFLAGS=$path
                                                fi
                                             fi
                                             if ( test X$with_$1_libpath != X && test X$with_$1_libpath != X$path )
                                             then
                                                AC_MSG_WARN([The path component of --with-$1])
                                                AC_MSG_WARN([and --with_$1_library disagree.])
                                                AC_MSG_WARN([Setting from --with-$1 ignored. ])
                                                $path=""
                                                use_$1=yes
                                             else
                                                with_$1_libpath=$path
                                                path=${path}/
                                                use_$1=yes
                                             fi
                                             if ( test X$with_$1_lnklibs != X && test X$with_$1_lnklibs != X$lib )
                                             then
                                                AC_MSG_WARN([The name component of --with-$1])
                                                AC_MSG_WARN([and --with_$1_library disagree.])
                                                AC_MSG_WARN([ Setting from --with-$1 ignored. ])
                                                lib=""
                                                use_$1=yes
                                             else
                                                with_$1_lnklibs=$lib
                                                use_$1=yes
                                             fi
                                          else
                                             AC_MSG_WARN([Illegal cmd-line option for --with-$1 ignored])
                                             use_$1=check
                                          fi
                                       else
                                          if (test X$path != X$with_$1 )
                                          then
                                             # --with-$1=/$path/is/not/a/dir/something
                                             AC_MSG_WARN([Illegal cmd-line option for --with-$1 ignored])
                                             use_$1=check
                                          else
                                             # --with-$1=assumed-library-file
                                             if ( test X$with_$1_lnklibs != X && test X$with_$1_lnklibs != X$with_$1 )
                                             then
                                                AC_MSG_WARN([The name component of --with-$1])
                                                AC_MSG_WARN([and --with_$1_library disagree,])
                                                AC_MSG_WARN([Setting from --with-$1 ignored. ])
                                                use_$1=yes
                                             else
                                                with_$1_lnklibs=$with_$1
                                                use_$1=yes
                                             fi
                                          fi
                                       fi
                                  fi;;
                              esac
                             ],
                             [if ( test X$with_$1 = X )
                              then
                                 use_$1=check
                              fi
                             ]
                            )
                 AC_ARG_WITH([$1-library],
                             [AS_HELP_STRING([--with-$1-library],
                                             [Declare the location of the $1 library and enable its functionality]
                                            )
                             ],
                             [if ( test X$use_$1 = Xno )
                              then
                                 AC_MSG_WARN([--with-$1 specifically disabled])
                                 AC_MSG_WARN([--with-$1-library ignored. ])
                              else
                                 if ( test -d $with_$1_library )
                                 then
                                    if ( test X$with_$1_libpath != X && test X$with_$1_library != X$with_$1_libpath )
                                    then
                                       AC_MSG_WARN([--with-$1-library redefines library location:])
                                       AC_MSG_WARN([   old location: $with_$1_libpath discarded ])
                                       AC_MSG_WARN([   for new location: $with_$1_library ])
                                    fi
                                    with_$1_libpath=$with_$1_library
                                    use_$1=yes
                                 else
                                    path = AS_DIRNAME([$with_$1_library])
                                    if ( test -d $path  )
                                    then
                                       #file=`basename $with_$1_library`
                                       # strip leading path
                                       file=`$EXPR "//$with_$1_library" : '.*/\([[^/]]*\)'`
                                       if ( test X$with_$1_library = X$path/$file )
                                       then
                                          # Parameter is a file path
                                          # Does it have a '.a' extension
                                          #lib=`basename $file .a`
                                          lib=`$EXPR "//$file" : '\(.*\)\.[[^.]]a$'`
                                          if ( test X$lib != X$file  )
                                          then
                                             lib='-l'`echo $lib | $SED 's/^lib//'`
                                             AC_MSG_WARN([Specifying --with-$1_library pointing to a .a])
                                             AC_MSG_WARN([will cause problems building with shared libraries.])
                                             AC_MSG_WARN([Instead we will use:])
                                             AC_MSG_WARN([   $lib])
                                             AC_MSG_WARN([   LDFLAGS=$path])
                                          else
                                             # Does it have a '.so' extension
                                             #lib=`basename $file .so`
                                             lib=`$EXPR "//$file" : '\(.*\)\.[[^.]]so$'`
                                             if ( test X$lib != X$file )
                                             then
                                                lib='-l'`echo $lib | $SED 's/^lib//'`
                                                AC_MSG_WARN([Specifying --with-$1_library pointing to])
                                                AC_MSG_WARN([will cause problems building with a .so])
                                                AC_MSG_WARN([static libraries. Instead, we will use:])
                                                AC_MSG_WARN([   $lib and])
                                                AC_MSG_WARN([   LDFLAGS=$path])
                                             else
                                                # Parameter is path and name fragment
                                                lib='-l'`echo $lib | $SED 's/^lib//'`
                                                trace 3 $with_$1_library asks to use $lib and LDFLAGS=$path
                                             fi
                                          fi
                                          if ( test X$with_$1_libpath != X && test X$with_$1_libpath != X$path )
                                          then
                                             AC_MSG_WARN([The existing path for --with_$1_library])
                                             AC_MSG_WARN([disagrees with the new value.])
                                             AC_MSG_WARN([Setting from --with-$1_library ignored. ])
                                             use_$1=yes
                                          else
                                             with_$1_libpath=$path
                                             use_$1=yes
                                          fi
                                          if ( test X$with_$1_lnklibs != X && test X$with_$1_lnklibs != X$lib )
                                          then
                                             AC_MSG_WARN([The name component of --with-$1 and])
                                             AC_MSG_WARN([--with_$1_library disagree.])
                                             AC_MSG_WARN([Setting from --with-$1_library ignored. ])
                                          else
                                             with_$1_lnklibs=$lib
                                          fi
                                       else
                                          AC_MSG_WARN([Illegal cmd-line option for --with-$1_library])
                                          AC_MSG_WARN([Setting from --with-$1_library ignored. ])
                                          use_$1=check
                                       fi
                                    else
                                       if ( test X$with_$1_lnklibs != X && test X$with_$1_lnklibs != X$lib )
                                       then
                                          AC_MSG_WARN([The name component of --with-$1 and])
                                          AC_MSG_WARN([--with_$1_library disagree,])
                                          AC_MSG_WARN([Setting from --with-$1_library ignored. ])
                                       else
                                          with_$1_lnklibs=$lib
                                       fi
                                    fi
                                 fi
                              fi
                             ],
                             []
                            )
                 AC_ARG_WITH([$1-include],
                             [AS_HELP_STRING([--with-$1-include],
                                             [Declare the location of the $1 header files and enable their functionality]
                                            )
                             ],
                             [if (test X$use_$1 = no )
                              then
                                 AC_MSG_WARN([--with-$1 specifically disabled,])
                                 AC_MSG_WARN([--with-$1-include ignored. ])
                              else
                                 if ( test -d $with_$1_include )
                                 then
                                    if ( test X$with_$1_inc != X && test X$with_$1_include != X$with_$1_inc )
                                    then
                                       AC_MSG_WARN([--with-$1-include redefines header location:])
                                       AC_MSG_WARN([   old location: $with_$1_inc discarded ])
                                       AC_MSG_WARN([   for new location: $with_$1_include ])
                                    fi
                                    with_$1_inc=$with_$1_include
                                    use_$1=yes
                                 else
                                    AC_MSG_WARN([--with-$1-include doesnt specify a directory.])
                                    AC_MSG_WARN([--with-$1-include ignored.])
                                 fi
                              fi
                             ],
                             []
                            )
                 if ( test X$with_$1 = Xno )
                 then
                    trace 4 Library $1 disabled
                 fi
                 if ( test X$with_$1_lnklibs = X )
                 then
                    with_$1_lnklibs="$4"
                    trace 4 No library name specified using "@<:@$with_$1_lnklibs@:>@"
                 fi
                 if ( test X$with_$1_libpath = X )
                 then
                    with_$1_libpath="$5"
                    trace 4 No --with-$1-library specified using "@<:@$with_$1_libpath@:>@"
                 fi
                 if ( test X$with_$1_inc = X )
                 then
                    case ""$with_$1 in
                    "" | 'no' | 'yes' | 'check' ) with_$1_inc="$7"
                                                  trace 4 No --with-$1-include specified default to "@<:@$with_$1_inc@:>@"
                                                  ;;
                    * )                           with_$1_inc="$with_$1"
                                                  if [ test -d $with_$1_inc ]
                                                  then
                                                     # Whole of --with-$1 was a directory
                                                     trace 4 No --with-$1-include specified using "@<:@$with_$1_inc@:>@"
                                                  else
                                                     with_$1_inc = AS_DIRNAME([$with_$1_inc])
                                                     if [ test -d $with_$1_inc ]
                                                     then
                                                        # Take the path component of --with-$1
                                                        trace 4 No --with-$1-include specified using "@<:@$with_$1_inc@:>@" fragment
                                                     else
                                                        # --with-$1 not relevant, use default
                                                        with_$1_inc="$7"
                                                        trace 4 No --with-$1-include specified using "@<:@$with_$1_inc@:>@" fragment
                                                     fi
                                                  fi
                                                  ;;
                    esac
                 fi
                ]
               ) 
          HelperLibs=$HelperLibs" "$1
          with_$1_fn=$2
          with_$1_prq="$3"
          with_$1_hdr="$6"
          $1_SUPPORT=""
          AS_IF([ test "$8" = NA],
                [$1_RavlLib=""],
                [$1_RavlLib="$8"]
               )
          AC_SUBST([$1_CFLAGS])
          AC_SUBST([$1_LIBS])
          AC_SUBST([$1_SUPPORT])
          echo $1_SUPPORT=@$1_SUPPORT@ >> ./ExtPkgSupp.$$
         ]
        )

dnl AX_RAVL_CHECK_FOR_LIB
dnl
dnl Macro to check if the given library package exits on the system. This will
dnl check for both the library itself and its header files.
dnl
dnl Macro parameters are:
dnl       p1: Ravl name for library
dnl       p2: Library to test for (-lxx)
dnl       p3: Function within that library to be used for the link test
dnl       p4: Path to the library if not on system defaults
dnl       p5: List of header files to test for
dnl       p6: Path to the library's headers if not in system defaults
dnl       p7: Action to take if the library package is present and usable
dnl       p8: Action to take if the library package is missing or unusable
dnl       p9: Other libraries that are needed for the link test
dnl
AC_DEFUN([AX_RAVL_CHECK_FOR_LIB],
         [
         trace 5 Checking for library:
         TRACE_INDENT="     "
         trace 5 "@<:@$1@:>@"
         TRACE_INDENT="     "
         trace 5 "@<:@$2@:>@"
         TRACE_INDENT="     "
         trace 5 "@<:@$3@:>@"
         TRACE_INDENT="     "
         trace 5 "@<:@$4@:>@"
         TRACE_INDENT="     "
         trace 5 "@<:@$5@:>@"
         TRACE_INDENT="     "
         trace 5 "@<:@$6@:>@"
         TRACE_INDENT="     "
         trace 5 Code...
         TRACE_INDENT="     "
         trace 5 Code...
         TRACE_INDENT="     "
         stripspaces=$9
         # As $9 will have acquired trailing characters from m4 substitutions
         trace 5 "@<:@$stripspaces@:>@"
         if ( test "X$4" != X )
         then
            TRACE_INDENT="     "
            trace 3 Setting LDFLAGS for library to -L$4
            LIB_LDFLAGS=-L$4
         else
            LIB_LDFLAGS=""
         fi
         if ( test "X$6" != X )
         then
            TRACE_INDENT="     "
            trace 3 Setting CPPFLAGS for library to -I$6
            LIB_CPPFLAGS=-I$6
         else
            LIB_CPPFLAGS=""
         fi
         prereqlibs=""
         for prereqlib in $9
         do
            eval new_prereqlib="\$with_${prereqlib}_lnklibs"
            new_prereqlib=`echo $new_prereqlib | $SED 's/^-l//' | $SED 's/^lib//'`
            TRACE_INDENT="     "
            trace 3 Appending -l$new_prereqlib to link line
            prereqlibs=$prereqlibs" -l"$new_prereqlib
            eval new_prereqlibpath=\$with_${prereqlib}_libpath
            if ( test "X$new_prereqlibpath" != X )
            then
               TRACE_INDENT="     "
               trace 3 Appending -L$new_prereqlibpath to LDFLAGS
               LIB_LDFLAGS=$LIB_LDFLAGS" "-L$new_prereqlibpath
            fi
         done
         TRACE_INDENT="     "
         preserve_LDFLAGS=$LDFLAGS
         preserve_CPPFLAGS=$CPPFLAGS
         LDFLAGS=$LIB_LDFLAGS
         CPPFLAGS=$LIB_CPPFLAGS
         trace 3 Checking for fn $3 in library $2
         AC_CHECK_LIB([$2], [$3], [found=true], [found=false], [$prereqlibs])
         if ( test X$found = Xtrue )
         then
            if ( test X"$5" != X )
            then
               for hf in $5 
               do
                  if ( test X$found = Xtrue )
                  then
                     TRACE_INDENT="     "
                     trace 3 Checking for header file $hf
                     AC_CHECK_HEADER([$hf], 
                                     [found=true], 
                                     [found=false], 
                                     [AC_INCLUDES_DEFAULT]
                                    )
                  fi
               done
            fi
         fi
         CPPFLAGS=$preserve_CPPFLAGS
         LDFLAGS=$preserve_LDFLAGS
         if ( test X$found = Xtrue )
         then
            TRACE_INDENT="     "
            trace 4 Library package $1 found
            fix_expan=$1
            AS_VAR_SET(${fix_expan}_CFLAGS, [$LIB_CPPFLAGS])
            if ( test X$LIB_LDFLAGS != X )
            then
               LIB_LDFLAGS=$LIB_LDFLAGS" -l$2"
            else
               LIB_LDFLAGS=-l$2
            fi
            if ( test "X$prereqlibs" != X )
            then
               LIB_LDFLAGS=$LIB_LDFLAGS" "$prereqlibs
            fi
            AS_VAR_SET(${fix_expan}_LIBS, [$LIB_LDFLAGS])
            $7
         else
            TRACE_INDENT="     "
            trace 3 Library package $1 not found
            $8
         fi
         ]
        )
dnl
dnl     End of Macro definitions

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


# Initialisation

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


# Now Check for pkg-config as we may need that next to check for QMake and
# will certainly need it later
PKG_PROG_PKG_CONFIG


# Look for QMake
#

trace 2 Looking for QMake
WORKING_QMAKE=""
QMAKE_CONFIGFILES=""

AC_ARG_WITH([QMake-pkgconf],
            [AS_HELP_STRING([--with-QMake-pkgconf],
                            [Declare where to find the pkg-config file for the version of QMake with which to build Ravl
                            ]
                           )
            ],
            [trace 4 "--with-QMake-pkgconf=@<:@$with_QMake_pkgconf@:>@"
             if ( test x$with_QMake_pkgconf = x )
             then
               # Assume the setting: "--with_QMake_pkgconf=" implies the
               # user doesn't want to use pkgconf
               with_QMake_pkgconf=no
             fi
             case $with_QMake_pkgconf in
                  'n' | 'N' | 'no' | 'No' | 'NO' | 'l' | 'L' | 'local' | 'Local' | 'LOCAL' ) 
                            # Disable the use of pkg-config and use the 
                            # version shipped with Ravl
                            with_QMake_pkgconf=UseLocal
                            ;;
                  'y' | 'Y' | 'yes' | 'Yes' | 'YES' )
                            with_QMake_pkgconf=QMake.pc
                            Require_QMake_pkgconf=Yes
                            ;;
                  * )       # Name of .pc for relevant install of QMake
                            Require_QMake_pkgconf=Yes
                            ;;
             esac
            ],
            [# Not specified, look for default on system paths
             trace 4 "--with-QMake-pkgconf unset"
             with_QMake_pkgconf=QMake.pc
            ]
           )
 
if ( test $with_QMake_pkgconf != UseLocal )
then
  AC_MSG_CHECKING(pkg-config for QMake)
  PKG_CHECK_EXISTS([$with_QMake_pkgconf],
                   [WORKING_QMAKE=`$PKG_CONFIG --variable=datadir $with_QMake_pkgconf 2>/dev/null`/QMake
                    #  "      "  = ${datadir}/QMake
                    QMAKE_CONFIGFILES=`$PKG_CONFIG --variable=sysconfdir $with_QMake_pkgconf 2>/dev/null`/QMake
                    # "        "     = ${sysconfdir}/QMake
                    AC_MSG_RESULT([yes])
                   ],
                   [AC_MSG_RESULT([no])
                    if ( test x$Require_QMake_pkgconf = xYes )
                    then
                      if ( test $with_QMake_pkgconf = QMake.pc )
                      then
                        AC_MSG_FAILURE([Cannot find the QMake pkg-config file)])
                      else
                        AC_MSG_FAILURE([Cannot find the specified QMake pkg-config file)])
                      fi
                    else
                      AC_MSG_WARN([Cannot find an installed QMake, will use the embedded version])
                    fi
                   ]
                  )
fi

if ( test x$WORKING_QMAKE = x || test x$WORKING_QMAKE = xUseLocal )
then
   trace 2 "Will use local QMake:"

   # No installed QMake found or we explicitly want to use the use one staged
   # from the source included with the Ravl distribution. 
   WORKING_QMAKE=@abs_top_builddir@/share/QMake
   QMAKE_CONFIGFILES=@abs_top_builddir@/QMake

   # We will also need to stage this copy as part of the Ravl build, so add
   # LocalQMake to the target for the required QMake builds
   NeededQMake="LocalQMake QMake"
else
   trace 2 "Will use pre-installed QMake:"

   # We have an installed version of QMake to use so we don't need to stage
   # our own copy for building Ravl. The Ravl build should therefore only
   # build the updated QMake for possible later installation.
   NeededQMake="QMake"
fi
trace 3 "   Config file $QMAKE_CONFIGFILES/Config.QMake"
trace 3 "   Live QMake under $WORKING_QMAKE"
trace 4 "   Build target(s) $NeededQMake"
   
# We need to configure the QMake source supplied with Ravl as (even if we
# are not using it for the Ravl build) 
#
# N.B.
#                              ************
#
# The actual configure will be carried out only at the end of this script
# Do not rely on Config.QMake, etc. being available until then.
#
#                              ************
#
AC_CONFIG_SUBDIRS([QMake])

AC_SUBST([WORKING_QMAKE])
AC_SUBST([QMAKE_CONFIGFILES])
AC_SUBST([NeededQMake])


# Declare non library specific cmd-line flags

AC_ARG_WITH([rpath],
            [AS_HELP_STRING([--with-rpath],
                            [Add all library include paths (-L) to the runtime search path of the binary
                            ]
                           )
            ],
            [case $with_rpath in
                  'y' | 'Y' | 'yes' | 'Yes' ) with_rpath=Y
                                              ;;
                  * )                         with_rpath=N
                                              ;;
             esac
            ],
            [with_rpath=N
            ]
           )


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
# these will form CONFIGURE_CFLAGS setting in Ravl (QMake) build system. 
CONFIGURE_CFLAGS=$CFLAGS
AC_SUBST([CONFIGURE_CFLAGS])
# Now actually check for compiler and and allow Config.Project to record the
# location found
AC_PROG_CC
AC_SUBST([CC])
# AC_PROG_CC won't actually have altered CFLAGS if a value was already set
# however, if an initial value was not provided, AC_PROG_CC may have set CFLAGS
# to include optimisation or -g. As we do not want this for the Ravl build, we
# reinitialise CFLAGS to the original value (which may be null or what a user
# supplied on the command line).
CFLAGS=$CONFIGURE_CFLAGS
# Find and record C pre-processor
AC_PROG_CPP
AC_SUBST([CPP])

# Check for C++ Compiler and Pre-processor
#
# Allow recording of any user-supplied CCFLAGS in Config.Project
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
    if ( test  X`$EXPR X$host_os : 'X\(mingw\).*'` != 'Xmingw' )
    then
      # gcc leaves a trailing carriage return which upsets mingw
      LD=`echo $LD | tr -d '\015'`
    fi
    
    if ( test X`$EXPR X$LD : '\(X[[\\/]].*\)'` = XX$LD )
    then
      # $LD is an absolute path
      # Fold all \ to /
      LD=`$echo "$LD"| $SED 's%\\\\%/%g'`
      # $LD may be an absolute path but might still need cannonicalisation
      while test X`$EXPR X$LD : 'X\(.*/\.\./\.*\)'` != X
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

  if ( test X`$EXPR X$LD : '\(X[[\\/]].*\)'` != XX$LD )
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
        *GNU* | *'with BFD'*)   if ( test  X`$EXPR X$host_os : 'X\(aix\).*'` != 'Xaix' )
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
                           *)   if ( test  X`$EXPR X$host_os : 'X\(aix\).*'` != 'Xaix' )
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
      if ( test  X`$EXPR X$host_os : 'X\(aix\).*'` != 'Xaix' )
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
    if ( test `$EXPR "$VER_STRING" : '.*GNU.*'` -gt 0 || test `$EXPR "$VER_STRING" : '.*with BFD.*'` -gt 0 )
    then
      # GNU linker
      GNU_LD=yes
    else
      GNU_LD=no
    fi
    AC_MSG_RESULT([$LD])
  else
    AC_MSG_RESULT([no])
    AC_MSG_FAILURE([Cannot find a working ld required for building Ravl])
  fi
else
  # User specified LD
  if ( test -x $LD )
  then
    # What type is it?
    VER_STRING=`"$LD" -v 2>&1 </dev/null`
    if ( test `$EXPR "$VER_STRING" : '.*GNU.*'` -gt 0 || test `$EXPR "$VER_STRING" : '.*with BFD.*'` -gt 0 )
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
      [AC_MSG_FAILURE([Cannot find a working ar required for building Ravl])]
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

if ( test X`$EXPR X$MAKE : '\(X[[\\/]].*\)'` != XX$MAKE )
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
      AC_MSG_FAILURE([Cannot find a working make required for building Ravl])
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
    if ( test `$EXPR "$VER_STRING" : '.*GNU.*'` -gt 0 || test `$EXPR "$VER_STRING" : '.*with BFD.*'` -gt 0 )
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
dnl Only select flags dependant on the tools in use here. Set all options
dnl pertaining to platform specifics later in this script
dnl

# Set CFLAGS
if (test X$GCC = Xyes)
then
   # GNU C Compiler
   PKG_GLOBAL_CFLAGS=" -Wall -pipe -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE "
   PKG_CANSIFLAG=" -ansi "
   PKG_DEFAULT_CFLAGS=" -O "
   CC_VERSION=`$CC --v 2>&1 | $GREP 'gcc version' | $AWK '{ print $3 }'`
   CC_MAJOR=`echo $CC_VERSION | $AWK -F. '{ print $1 }'`
   CC_MINOR=`echo $CC_VERSION | $AWK -F. '{ print $2 }'`
   CC_PATCH=`echo $CC_VERSION | $AWK -F. '{ print $3 }'`
   if test X$CC_MAJOR=X4 && test $(($CC_MINOR)) -lt 3
   then
      # -funsafe-math-optimisation (implied by -ffast-math) was indeed, unsafe
      # on versions 4.0/1/2 of GCC, so just implement the parts of -ffast-math
      # that work with the Ravl source and employ -O3 as compensation
      PKG_OPT_CFLAGS=" -O3 -fno-math-error -fno-trapping-math -fno-signalling-nans -D__FAST_MATH__ -Wstrict-aliasing=2 -fno-strict-aliasing "
   else
      PKG_OPT_CFLAGS=" -O2 -ffast-math -fno-strict-aliasing "
   fi
   PKG_OPT_CFLAGS=$PKG_OPT_CFLAGS" -march=native -mtune=native "
   PKG_DEBUG_CFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
   if test $ac_cv_prog_cc_g = yes
   then
      PKG_DEBUG_CFLAGS=" -g "$PKG_DEBUG_CFLAGS
   fi
   PKG_CHECK_CFLAGS=" -DRAVL_CHECK "
   PKG_SHARED_CFLAGS=" -shared -fPIC "
   PKG_PROF_CFLAGS=" -p "
   PKG_GPROF_CFLAGS=" -pg "
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
   Xsolaris* ) PKG_GLOBAL_CFLAGS=""
               PKG_CANSIFLAG=" -ansi "
               PKG_DEFAULT_CFLAGS=" -O "
               PKG_OPT_CFLAGS=" -O2 "
               PKG_DEBUG_CFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CFLAGS=" -g "$PKG_DEBUG_CFLAGS
               fi
               PKG_SHARED_CFLAGS=" -KPIC "
               PKG_PROF_CFLAGS=" -p "
               PKG_GPROF_CFLAGS=" -pg "
               PKG_CHECK_CFLAGS=" -DRAVL_CHECK "
               ;;
   Xirix* )    CC=$CC" -n32 -J4 -mp "
               # Not sure why this originally was done this way rather than use
               # CFLAGS, but assume there was a reason and propogate the method
               PKG_GLOBAL_CFLAGS=" -mips4 -r1000 "
               PKG_CANSIFLAG=" -ansi "
               PKG_DEFAULT_CFLAGS=" -O1 -OPT:Olimit_opt=on "
               PKG_OPT_CFLAGS=" -O3 -OPT:const_copy_limit=50000:Olimit=80000 "
               PKG_DEBUG_CFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CFLAGS=" -g "$PKG_DEBUG_CFLAGS
               fi
               PKG_SHARED_CFLAGS=""
               PKG_PROF_CFLAGS=""
               PKG_GPROF_CFLAGS=""
               PKG_CHECK_CFLAGS=" -DRAVL_CHECK "
               ;;
   * )         PKG_GLOBAL_CFLAGS=""
               PKG_CANSIFLAG=" -ansi "
               PKG_DEFAULT_CFLAGS=" -O "
               PKG_OPT_CFLAGS=" -O2 "
               PKG_DEBUG_CFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CFLAGS=" -g "$PKG_DEBUG_CFLAGS
               fi
               PKG_SHARED_CFLAGS=""
               PKG_PROF_CFLAGS=""
               PKG_GPROF_CFLAGS=""
               PKG_CHECK_CFLAGS=" -DRAVL_CHECK "
               ;;
   esac
fi

# Declare output variables for compiler ANSI switches
AC_SUBST([PKG_CANSIFLAG])
AC_SUBST([PKG_CCANSIFLAG])

# Declare PKG_XXXXX_CFLAGS variables as output for Config.Project
AC_SUBST([PKG_CHECK_CFLAGS])
AC_SUBST([PKG_DEBUG_CFLAGS])
AC_SUBST([PKG_DEFAULT_CFLAGS])
AC_SUBST([PKG_GLOBAL_CFLAGS])
AC_SUBST([PKG_GPROF_CFLAGS])
AC_SUBST([PKG_OPT_CFLAGS])
AC_SUBST([PKG_PROF_CFLAGS])
AC_SUBST([PKG_SHARED_CFLAGS])

# Set CCFLAGS
if (test X$GXX = Xyes)
then
   # GNU C++ Compiler
   PKG_GLOBAL_CCFLAGS=" -Wall -pipe -D_FILE_OFFSET_BITS=64 -D_LARGEFILE_SOURCE "
   PKG_CCANSIFLAG=" -ansi "
   PKG_DEFAULT_CCFLAGS=" -O "
   CXX_VERSION=`$CXX --v 2>&1 | $GREP 'gcc version' | $AWK '{ print $3 }'`
   CXX_MAJOR=`echo $CXX_VERSION | $AWK -F. '{ print $1 }'`
   CXX_MINOR=`echo $CXX_VERSION | $AWK -F. '{ print $2 }'`
   CXX_PATCH=`echo $CXX_VERSION | $AWK -F. '{ print $3 }'`
   if test X$CXX_MAJOR=X4 && test $(($CXX_MINOR)) -lt 3
   then
      # -funsafe-math-optimisation (implied by -ffast-math) was indeed, unsafe
      # on versions 4.0/1/2 of GCC, so just implement the parts of -ffast-math
      # that work with the Ravl source and employ -O3 as compensation
      PKG_OPT_CCFLAGS=" -O3 -fno-math-error -fno-trapping-math -fno-signalling-nans -D__FAST_MATH__ -Wstrict-aliasing=2 -fno-strict-aliasing "
   else
      PKG_OPT_CCFLAGS=" -O2 -ffast-math -fno-strict-aliasing "
   fi
   PKG_OPT_CCFLAGS=$PKG_OPT_CCFLAGS" -march=native -mtune=native "
   PKG_DEBUG_CCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
   if test $ac_cv_prog_cc_g = yes
   then
      PKG_DEBUG_CCFLAGS=" -g "$PKG_DEBUG_CCFLAGS
   fi
   PKG_CHECK_CCFLAGS=" -DRAVL_CHECK "
   PKG_SHARED_CCFLAGS=" -shared -fPIC "
   PKG_PROF_CCFLAGS=" -p "
   PKG_GPROF_CCFLAGS=" -pg "
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
   Xsolaris* ) PKG_GLOBAL_CCFLAGS=""
               PKG_CCANSIFLAG=" -ansi "
               PKG_DEFAULT_CCFLAGS=" -O "
               PKG_OPT_CCFLAGS=" -O2 "
               PKG_DEBUG_CCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CCFLAGS=" -g "$PKG_DEBUG_CCFLAGS
               fi
               PKG_SHARED_CCFLAGS=" -KPIC "
               PKG_PROF_CCFLAGS=" -p "
               PKG_GPROF_CCFLAGS=" -pg "
               PKG_CHECK_CCFLAGS=" -DRAVL_CHECK "
               ;;
   Xirix* )    CXX=$CXX" -n32 -J4 -mp -LANG:ansi-for-init-scope=on "
               # Not sure why this originally was done this way rather than use
               # CCFLAGS, but assume there was a reason and propogate the method
               PKG_GLOBAL_CCFLAGS=" -mips4 -r1000 -woff 1021 -woff 1681 -woff 1882 "
               PKG_CCANSIFLAG=" -ansi "
               PKG_DEFAULT_CCFLAGS=" -O1 -OPT:Olimit_opt=on "
               PKG_OPT_CCFLAGS=" -O3 -OPT:const_copy_limit=50000:Olimit=80000 "
               PKG_DEBUG_CCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CCFLAGS=" -g "$PKG_DEBUG_CCFLAGS
               fi
               PKG_SHARED_CCFLAGS=""
               PKG_PROF_CCFLAGS=""
               PKG_GPROF_CCFLAGS=""
               PKG_CHECK_CCFLAGS=" -DRAVL_CHECK "
               ;;
   * )         PKG_GLOBAL_CCFLAGS=""
               PKG_CCANSIFLAG=" -ansi "
               PKG_DEFAULT_CCFLAGS=" -O "
               PKG_OPT_CCFLAGS=""
               PKG_DEBUG_CCFLAGS=" -DRAVL_CHECK -DQMAKE_PARANOID "
               if test $ac_cv_prog_cc_g = yes
               then
                  PKG_DEBUG_CCFLAGS=" -g "$PKG_DEBUG_CCFLAGS
               fi
               PKG_SHARED_CCFLAGS=""
               PKG_PROF_CCFLAGS=""
               PKG_GPROF_CCFLAGS=""
               PKG_CHECK_CCFLAGS=" -DRAVL_CHECK "
               ;;
   esac
fi

# Declare PKG_XXXXX_CCFLAGS variables as output for Config.Project
AC_SUBST([PKG_CHECK_CCFLAGS])
AC_SUBST([PKG_DEBUG_CCFLAGS])
AC_SUBST([PKG_DEFAULT_CCFLAGS])
AC_SUBST([PKG_GLOBAL_CCFLAGS])
AC_SUBST([PKG_GPROF_CCFLAGS])
AC_SUBST([PKG_OPT_CCFLAGS])
AC_SUBST([PKG_PROF_CCFLAGS])
AC_SUBST([PKG_SHARED_CCFLAGS])


# Set PKG_XXXXX_LDFLAGS; LDLIBFLAGS and other linker settings
if ( test X$GNU_LD = Xyes )
then
  # We are using a GNU linker

  # General Ravl link options
  PKG_CHECK_LDFLAGS=
  PKG_DEBUG_LDFLAGS=
  # -g does nothing on the linker so omit it
  PKG_DEFAULT_LDFLAGS=
  PKG_GLOBAL_LDFLAGS=
  PKG_GPROF_LDFLAGS=
  PKG_OPT_LDFLAGS=
  PKG_PROF_LDFLAGS=
  PKG_SHARED_LDFLAGS=" -rdynamic "

  # Ravl link options used for libraries
  PKG_CHECK_LDLIBFLAGS=
  PKG_DEBUG_LDLIBFLAGS=
  PKG_DEFAULT_LDLIBFLAGS=
  PKG_GLOBAL_LDLIBFLAGS=
  PKG_GPROF_LDLIBFLAGS=
  PKG_OPT_LDLIBFLAGS=
  PKG_PROF_LDLIBFLAGS=
  PKG_SHARED_LDLIBFLAGS=" -shared "

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
  Xcygwin )   PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=" -g "
              PKG_GLOBAL_LDFLAGS=
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=" -Wl,-export-dynamic "
              # sic, this is an assumption on what compiler is in use

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              # sic, this is an assumption on what compiler is in use
              UNDEFSYMB=" -u #"
              ;;
  Xirix*    ) PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=
              PKG_GLOBAL_LDFLAGS=" -n32 -Wl,-woff 84 "
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              UNDEFSYMB=" -u#"
              ;;
  Xsolaris* ) PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=" -g "
              PKG_GLOBAL_LDFLAGS=
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" -dy -G "

              LIBPATHSWITCH=" -R#"
              UNDEFSYMB=" -u#"
              ;;
  * )         PKG_CHECK_LDFLAGS=
              PKG_DEBUG_LDFLAGS=
              PKG_GLOBAL_LDFLAGS=
              PKG_DEFAULT_LDFLAGS=
              PKG_GPROF_LDFLAGS=
              PKG_PROF_LDFLAGS=
              PKG_OPT_LDFLAGS=
              PKG_SHARED_LDFLAGS=" -rdynamic "

              PKG_CHECK_LDLIBFLAGS=
              PKG_DEBUG_LDLIBFLAGS=
              PKG_GLOBAL_LDLIBFLAGS=
              PKG_DEFAULT_LDLIBFLAGS=
              PKG_GPROF_LDLIBFLAGS=
              PKG_PROF_LDLIBFLAGS=
              PKG_OPT_LDLIBFLAGS=
              PKG_SHARED_LDLIBFLAGS=" -shared "

              LIBPATHSWITCH=" -Wl,-rpath#"
              UNDEFSYMB=" -u#"
              ;;
  esac
fi

# Declare variables that the linker as output for Config.Project
AC_SUBST([LDFLAGS])
# any user-provided LDFLAGS will be recorded in Config.Project as CONFIGURE_LDFLAGS

AC_SUBST([PKG_CHECK_LDFLAGS])
AC_SUBST([PKG_DEBUG_LDFLAGS])
AC_SUBST([PKG_DEFAULT_LDFLAGS])
AC_SUBST([PKG_GLOBAL_LDFLAGS])
AC_SUBST([PKG_GPROF_LDFLAGS])
AC_SUBST([PKG_OPT_LDFLAGS])
AC_SUBST([PKG_PROF_LDFLAGS])
AC_SUBST([PKG_SHARED_LDFLAGS])

AC_SUBST([PKG_CHECK_LDLIBFLAGS])
AC_SUBST([PKG_DEBUG_LDLIBFLAGS])
AC_SUBST([PKG_DEFAULT_LDLIBFLAGS])
AC_SUBST([PKG_GLOBAL_LDLIBFLAGS])
AC_SUBST([PKG_GPROF_LDLIBFLAGS])
AC_SUBST([PKG_OPT_LDLIBFLAGS])
AC_SUBST([PKG_PROF_LDLIBFLAGS])
AC_SUBST([PKG_SHARED_LDLIBFLAGS])

AC_SUBST([LIBPATHSWITCH])
AC_SUBST([UNDEFSYMB])


# Declare flags for ar

dnl DO NOT PUT PLATFORM SPECIFICS HERE. If your platform has a non-standard ar
dnl that takes completely different flags then that is ok to list here (as it
dnl pertains to the tool itself, not the platform). If your platform uses the
dnl GNU version of ar but, for example requires an added 's' flag, that setting
dnl should be put in the platform specific code at the end of this script, not
dnl here.
PKG_ARFLAGS=ruc

AC_SUBST([PKG_ARFLAGS])

# Declare flags for make

dnl Only put make flags specific to any particular version of make here. DO NOT
dnl PUT PLATFORM SPECIFIC FLAGS HERE; they go in the final section of this
dnl script

if ( test X$GNU_MAKE = Xyes )
then
  PKG_MAKEFLAGS=" --no-print-directory -r "
else
  PKG_MAKEFLAGS=" -r "
fi

AC_SUBST([MAKEFLAGS])
# User-provided MAKEFLAGS will be recorded as CONFIGURE_MAKEFLAGS in Config.Project 
AC_SUBST([PKG_MAKEFLAGS])

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

# Check for programs needed for the Ravl build
# First check for those programs that GNU coding standards regard as safe to use
# in a makefile if they are employed via make variables (and thus create the
# necessary variables)
AC_PATH_PROG([CHGRP],[chgrp],[Missing])
AS_IF([test $CHGRP = Missing],
      [AC_MSG_FAILURE([Cannot find a working chgrp required for building Ravl])]
     )
AC_SUBST([CHGRP])

AC_PATH_PROG([CHMOD],[chmod],[Missing])
AS_IF([test $CHMOD = Missing],
      [AC_MSG_FAILURE([Cannot find a working chmod required for building Ravl])]
     )
AC_SUBST([CHMOD])

AC_PATH_PROG([CHOWN],[chown],[Missing])
AS_IF([test $CHOWN = Missing],
      [AC_MSG_FAILURE([Cannot find a working chown required for building Ravl])]
     )
AC_SUBST([CHOWN])

AC_PROG_LEX
AC_SUBST([LEX])
AC_SUBST([LEXLIB])

AC_PATH_PROG([SORT],[sort],[Missing])
AS_IF([test $SORT = Missing],
      [AC_MSG_FAILURE([Cannot find a working sort required for building Ravl])]
     )
AC_SUBST([SORT])

AC_PROG_YACC
AC_SUBST([YACC])

# GNU standards regard the following safe to call directly but as we already
# implemented them via a make variable, set those here...
AC_PATH_PROG([CP],[cp],[Missing])
AS_IF([test $CP = Missing],
      [AC_MSG_FAILURE([Cannot find a working cp required for building Ravl])]
     )
AC_SUBST([CP])

AC_PATH_PROG([RM],[rm],[Missing])
AS_IF([test $RM = Missing],
      [AC_MSG_FAILURE([Cannot find a working rm required for building Ravl])]
     )
AC_SUBST([RM])

AC_PATH_PROG([TOUCH],[touch],[Missing])
AS_IF([test $TOUCH = Missing],
      [AC_MSG_FAILURE([Cannot find a working touch required for building Ravl])]
     )
AC_SUBST([TOUCH])

AC_PATH_PROG([TR],[tr],[Missing])
AS_IF([test $TR = Missing],
      [AC_MSG_FAILURE([Cannot find a working tr required for building Ravl])]
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
                                     [AC_MSG_FAILURE([Cannot find a working pwd required for building Ravl])]
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
                                            [AC_MSG_FAILURE([Cannot find a working pawd or pwd required for building Ravl])]
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
                                            [AC_MSG_FAILURE([Cannot find a working pwd required for building Ravl])]
                                           )
                                     ]
                                    )
                               ;;
                       esac
            ],
            [trace 5 No --with-$1 specified
             AC_PATH_PROG([GET_CWD],[pwd],[Missing])
             AS_IF([test $GET_CWD = Missing],
                   [AC_MSG_FAILURE([Cannot find a working pwd required for building Ravl])]
                  )
            ]
           )
AC_SUBST([GET_CWD])

# Now check for those utilities we use in the Ravl build that aren't recognised
# as being reliably portable by the GNU standard. For these, we will need to
# consider providing an alternative implementation in cases where any are absent.
AC_PROG_LN_S
AC_SUBST([LN_S])

AS_IF([test x$PAGER = x],
      [AC_PATH_PROG([PKG_PAGER],[more],[Missing])
       AS_IF([test $PKG_PAGER = Missing],
             [AC_PATH_PROG([PKG_PAGER],[pg],[Missing])
              AS_IF([test $PKG_PAGER = Missing],
                    [AC_MSG_FAILURE([Cannot find a working pager required for building Ravl])]
                   )
             ]
            )
      ],
      [PKG_PAGER=$PAGER]
     )
AC_SUBST([PKG_PAGER])

AC_PATH_PROG([SYNC],[sync],[Missing])
AS_IF([test $SYNC = Missing],
      [AC_MSG_FAILURE([Cannot find a working sync required for building Ravl])]
     )
AC_SUBST([SYNC])

AC_PATH_PROG([XARGS],[xargs],[Missing])
AS_IF([test $XARGS = Missing],
      [AC_MSG_FAILURE([Cannot find a working xargs required for building Ravl])]
     )
AC_SUBST([XARGS])

# Perl is also required by the Ravl (really QMake) build process but as it is 
# also required for Ravl functionality, this is addressed by a check further on
# in this script (where RESOURCES is being set-up)

# Remaining utils used to build Ravl are all regarded by the GNU standards as
# safe for direct use:
#       cat cmp diff ls mv rmdir tar test touch true


trace 2 Initial checks complete, now processing cmd-line

# Define all Ravl Helper libraries and their associated cmd-line options
#

dnl
dnl Declare external helper libraries (i.e. those outside of RAVL) here: 
dnl
dnl
dnl First declare all those libraries implemented via pkg-config
dnl
dnl Parameters are:
dnl
dnl  1 Ravl name for library. The name used for RESOURCES and also for 
dnl    LIB_CFLAGS, LIB_LIBS, etc. used in Config.Project and the .def files.
dnl  2 Name by which pkg-config knows the library. Can be over-ridden by 
dnl    --with-<lib>-pkgconf
dnl  3 Name(s) of the Ravl libraries that provide functionality dependant on
dnl    the third-party library. Only lists libraries that are omitted from the
dnl    Ravl build if the external library is not present. Any Ravl library is 
dnl    associated with the most appropriate external library, this does not
dnl    imply this is the only external library that can inhibit the Ravl libs
dnl    inclusion. If additional requirements may block the Ravl library, the
dnl    associated <lib>_SUPPORT variable will be suitably amended towards the
dnl    end of this script. This may include blockage by another external pkg
dnl    being missing or even a platfom dependancy 
dnl
dnl
dnl               Name,              pkg-config       Ravl-lib
dnl
AX_RAVL_HELPER_PC([libdc1394],       [libdc1394],   [libRavlImgIO1394dc])
AX_RAVL_HELPER_PC([libAudioFile],    [audiofile],     [libRavlAudioFile])
AX_RAVL_HELPER_PC([libavc1394],      [libavc1394],    [libRavlFireWire])
dnl Modern versions of libAVIFile currently unsupported by Ravl. Drop for now:
dnl  .._HELPER_PC([avifile],         [avifile],       [libRavlAVIFile])
AX_RAVL_HELPER_PC([libCurl],         [libcurl],       [libRavlURLIO])
AX_RAVL_HELPER_PC([libDV],           [libdv],         [libRavlDV])
AX_RAVL_HELPER_PC([FFMPEGCDEC],      [libavcodec],    [NA])
AX_RAVL_HELPER_PC([FFMPEGCONT],      [libavformat],   [NA])
AX_RAVL_HELPER_PC([FFMPEGSWS],       [libswscale],    [NA])
AX_RAVL_HELPER_PC([FFMPEGUTIL],      [libavutil],     [NA])
dnl     Ravl wants FFMPEG as only a single entity (libffmpeg). To support this,
dnl     this script will (later) set up the appropriate flags if all four FFMPEG
dnl     components exist on the target system
AX_RAVL_HELPER_PC([iksemel],         [iksemel],       [libRavlXMPPIksemel])
AX_RAVL_HELPER_PC([libglade2],       [libglade-2.0],  [libRavlLibGlade])
AX_RAVL_HELPER_PC([LibGnome],        [libgnome-2.0],  [libRavlGUIGnome])
AX_RAVL_HELPER_PC([libGTK2],         [gtk+-2.0],      [NA])
dnl     The absence of libGTK2 will block the inclusion of some other libraries'
dnl     Ravl functionality (e.g. libRavlLibGlade, libRavlGUIGnome, etc.) but is
dnl     not solely responsible for the inclusion any Ravl libraries itself.
AX_RAVL_HELPER_PC([GTKGLExt],        [gtkglext-1.0],  [NA])
dnl     GTKGLExt does not relate directly to a Ravl library but it does add more
dnl     functionality to GUI3D and its absence will block OpenSceneGraph support
AX_RAVL_HELPER_PC([Loudmouth],       [loudmouth-1.0], [libRavlXMPPLoudmouth])
AX_RAVL_HELPER_PC([OpenCV],          [opencv],        [libRavlOpenCV])
AX_RAVL_HELPER_PC([libPNG],          [libpng],        [NA])
AX_RAVL_HELPER_PC([libGL],           [gl],            [NA])
AX_RAVL_HELPER_PC([libGLU],          [glu],           [NA])
dnl     Ravl sometimes wants libGL and libGLU as a single entity (OpenGL), this
dnl     configure script facilitates this by setting up an OpenGL set of flags
dnl     if both libGL and libGLU exist. Singly, neither libGL or libGLU will
dnl     cause Ravl to loose a library in their absence, hence no Ravl-lib defn
dnl     (although the absence of libGL can block libRavlGUIOpenSceneGraph)
AX_RAVL_HELPER_PC([OpenSceneGraph],  [osgGtk-1.0],    [libRavlGUIOpenSceneGraph])
AX_RAVL_HELPER_PC([RLog],            [librlog],       [libRavlRLog])
AX_RAVL_HELPER_PC([devVideo4Linux],  [libv4l1],       [libRavlImgIOV4L])
AX_RAVL_HELPER_PC([devVideo4Linux2], [libv4l2],       [libRavlImgIOV4L2])
dnl 
dnl Now define those libraries needing manual compilation checks:
dnl
dnl Parameters are:
dnl
dnl  1 Ravl name for library. The name used for RESOURCES and also for 
dnl    LIB_CFLAGS, LIB_LIBS, etc. used in Config.Project and the .def files.
dnl  2 Name of a function in the library that can be used as a test call
dnl  3 Space separated list of dependancies of the library. Specify the Ravl
dnl    name of the library - e.g. for dependancies on Zlib specify Zlib not 
dnl    -lz or z. 
dnl    *** N.B. ***
dnl         Dependancies on libraries reliant on pkg-config for their
dnl         location are not supported. I.E. if library A depends on
dnl         library B, library B must be specified here as a manual
dnl         check if library A is also specified as a manual check.
dnl  4 Link line entry for the library - i.e. -l<libname>
dnl  5 Usual location of library if different to the system default lib dirs
dnl  6 Name(s) of header files needed to use the library (only really need to
dnl    provide one for a specimin test unless it is know that sub-sets of the
dnl    header files are regularly seen in the wild). If specifying multiple
dnl    headers, provide a space seperated list.
dnl  7 Usual location of library's header files if different to sys defaults
dnl  8 Name(s) of the Ravl libraries that provide functionality dependant on
dnl    the third-party library. Only lists libraries that are omitted from the
dnl    Ravl build if the external library is not present. Any Ravl library is 
dnl    associated with the most appropriate external library, this does not
dnl    imply this is the only external library that can inhibit the Ravl libs
dnl    inclusion. If additional requirements may block the Ravl library, the
dnl    associated <lib>_SUPPORT variable will be suitably amended towards the
dnl    end of this script. This may include blockage by another external pkg
dnl    being missing or even a platfom dependancy 
dnl
dnl               Name,          Fn,                 Deps,   -l,           -L, .h,                -I  Ravl-lib
dnl 
AX_RAVL_HELPER_CC([LibJasper],   [jas_init],         [],     [-ljasper],   [], [jasper/jasper.h], [], [libRavlImgIOJasper])
AX_RAVL_HELPER_CC([JPEG],        [jpeg_read_header], [],     [-ljpeg],     [], [jpeglib.h],       [], [NA])
dnl# JPEG check will support either the standard or jpeg-turbo library as they
dnl# are a straight substitution for each other
AX_RAVL_HELPER_CC([posixThreads],[pthread_self],     [],     [ -lpthread], [], [pthread.h],       [], [NA])
AX_RAVL_HELPER_CC([LibTIFF],     [TIFFGetVersion],   [Zlib], [-ltiff],     [], [],                [], [NA])
AX_RAVL_HELPER_CC([Zlib],        [zlibCompileFlags], [],     [-lz],        [], [],                [], [libRavlZLib])

dnl#JFi:Develop code to prime these by ./configure
dnl#
dnl# Developer notes:
dnl#
dnl#   -l entries are from original .def files and may now not be wholly 
dnl#      accurate (reflect current release or the library).
dnl#   -l entry (and 8 params for _HELPER call) do not necessarily mean
dnl#      pkg-config isn't supported
dnl#   _HELPER_PC or _HELPER_CC rather than _HELPER_xC imply the method to use
dnl#   When adding support for these, remember to update the .def file to remove the $(error call
dnl#
dnl# Priorities
dnl#
dnl#   _HELPER_CC([Socket],      [socket],           [],     [-lsocket],   [], [sys/socket.h],    [], [NA])
dnl#   _HELPER_CC([NetDB],       [gethostbyname_r],  [],     [-lnsr],        [], [netdb.h],         [], [NA])
dnl#
dnl#
dnl# Less so
dnl#
dnl#   _HELPER_CC([dvdread],     [TBA],              [],     [-ldvdread],  [], [],                []. [libRavlDVDRead])
dnl#      libRavlDVDRead also inhibited by lack of libmpeg2
dnl#   _HELPER_xC([libClipStationPro], [],           [],     [-ldvsoem],   [], [],                [], [CSPDriver])
dnl#   _HELPER_xC([LibGd],       [],                 [],     [-lgd],       [], [],                [], [RavlImgGd])
dnl#   _HELPER_xC([LAPACK],      [],                 [],     [-llapack -lblas], [], [],           [], [RavlLapack RavlLapackWraps])
dnl#   _HELPER_xC([libmcrypt],   [],                 [],     [-lmcrypt],   [], [],                [], [RavlCrypto])
dnl#   _HELPER_xC([libmpeg2],    [],                 [],     [-lmpeg2],    [], [],                [], [RavlLibMPEG2])
dnl#      Also inhibits libRavlDVDRead on its absence
dnl#   _HELPER_xC([LibPython],   [],                 [],     [],           [], [],                [], [RavlPython])
dnl#     May have to use:
dnl#       PYTHON_VERSION = $(shell python -c "import distutils.sysconfig; print distutils.sysconfig.get_config_var('VERSION')")
dnl#       PYTHON_INCLUDE = $(shell python -c "import distutils.sysconfig; print distutils.sysconfig.get_python_inc()")
dnl#       PYTHON_LIB = python$(PYTHON_VERSION)
dnl#   _HELPER_xC([SwigPython],  [],                 [Python Swig], [],    [], [],                [], [RavlPythonSwig])
dnl#  
dnl#
dnl# More or less dead
dnl#
dnl#   _HELPER_xC([libehs],      [],                 [],     [-lpme -lehs], [], [],                [], [RavlEHS])
dnl#   _HELPER_xC([libGuppi],    [],                 [],     [-lguppitank `gnome-config libguppi --libs`], [], [], [], [RavlDPGraph RavlGuppi RavlPlot])
dnl#   _HELPER_xC([libmkl],      [],                 [posixThreads Math], [-lmkl_ia32 -lguide], [], [], [], [RavlIntelMKL])
dnl#   _HELPER_xC([Meteor1],     [],                 [],     [],           [], [],                [], [RavlImgIOMeteor1])
dnl#   _HELPER_xC([libmpeg],     [],                 [],     [-lmpeg],     [], [],                [], [RavlMPEG])
dnl#   _HELPER_xC([uEyeSDK],     [],                 [],     [-lueye_api], [], [],                [], [RavlImgIOuEye])
dnl#      Also need to set uEyeSDK_CFLAGS=-D__LINUX__
dnl#   _HELPER_xC([UUIdTheo],    [],                 [],     [-luuid],     [], [],                [], [RavlUUId])
dnl#  
dnl#
dnl# Needed?
dnl#
dnl#   _HELPER_CC([GLUT],        [glutInit],         [],     [-lglut|-lfreeglut], [], [GL/glut.h],[], [NA])
dnl#   _HELPER_xC([x11],         [],                 [],     [],           [], [],                [], [])
dnl#  
dnl#
dnl# OS Specific
dnl#
dnl#   _HELPER_xC([SGIVL],       [],                 [],     [-lvl -ldmedia],[], [],              [], [RavlDMedia])
dnl#   _HELPER_xC([LibDirectShow], [],               [],     [],           [], [],                [], [RavlDirectShow])
dnl#   Mac OSX Support:
dnl#     Currently global MacOSX REQUIRES flag. 
dnl#     RavlMacOSXVideoCapture depends on RavlMacOSXRunLoop and all libraries
dnl#     Already use MacOSXVideoCapture_SUPPORT later in this script, so this needs to be kept in mind
dnl#     _HELPER_xC([MacOSXFoundation], [],            [],     [-framework Foundation], [], [],     [], [RavlMacOSXRunLoop])
dnl#     _HELPER_xC([MacOSXQuartzCore], [],            [],     [-framework QuartzCore], [], [],     [], [])
dnl#     _HELPER_xC([MacOSXQTKit], [],                 [],     [-framework QTKit], [], [],          [], [])
dnl#     _HELPER_xC([ObjC],        [],                 [],     [-lobjc],     [], [],                [], [])
dnl#
dnl# Dropped
dnl#
dnl#    CPPUnit CUDA DMedia FFTW glib {GNU}ReadLine JavaScript libmpeg Math 
dnl#    Perl5 php libptp2 QtTestRunner SQLite3 Sys system Xaw Xerces Xt 
dnl#
dnl# In the meantime...
dnl#
dnl#    declare dummy <Lib>_SUPPORT substitution variables
dnl#    here to prevent the single lib makefile trying to load unsupported
dnl#    libraries
dnl#
AC_SUBST([avifile_SUPPORT])
avifile_SUPPORT=""
echo avifile_SUPPORT=@avifile_SUPPORT@ >> ./ExtPkgSupp.$$
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
AC_SUBST([LAPACK_SUPPORT])
LAPACK_SUPPORT=""
echo LAPACK_SUPPORT=@LAPACK_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([MacOSXRunLoop_SUPPORT])
MacOSXRunLoop_SUPPORT=""
echo MacOSXRunLoop_SUPPORT=@MacOSXRunLoop_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([MacOSXVideoCapture_SUPPORT])
MacOSXVideoCapture_SUPPORT=""
echo MacOSXVideoCapture_SUPPORT=@MacOSXVideoCapture_SUPPORT@ >> ./ExtPkgSupp.$$
AC_SUBST([Meteor1_SUPPORT])
Meteor1_SUPPORT=""
echo Meteor1_SUPPORT=@Meteor1_SUPPORT@ >> ./ExtPkgSupp.$$
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

# FFMPEG and OpenGL are special cases as Ravl regards these as single entities
# while the libraries are provided in smaller components. Initial definitions
# (above) work at the sub-component level, we now need to provide the switches
# and checking for the meta-level. As both library packages are checked for by
# using pkg-config, we'll only define the top level --with-<lib> and specific
# --with-<lib>-pkgconfig flags.
#
# First we consider FFMPEG; full package is named libffmpeg while existing
# sub-packages are FFMPEGCDEC, FFMPEGCONT, FFMPEGUTIL and FFMPEGSWS
AC_ARG_WITH([libffmpeg-pkgconf],
            [AS_HELP_STRING([--with-libffmpeg-pkgconf],
                            [Enable the FFMPEG libraries via pkg-conf. If declaring a pkg-config control file, this will be used for all of the individual FFMPEG sub-packages. You are more likely to want to use the individual command-line options for each of the sub-packages (--with-FFMPEGCDEC-pkgconf --with-FFMPEGCONT-pkgconf --with-FFMPESWS-pkgconf --with-FFMPEGUTIL-pkgconf) as the pkg-config control files are usually seperate for each sub-package.]
                           )
            ],
            [case $with_libffmepg_pkgconf in
             'n' | 'N' | 'no' | 'No' | 'NO' ) 
                              with_libffmpeg_pkgconf=no
                              use_libffmpeg=no
                              trace 4 --with-libffmpeg-pkgconf disabling FFMPEG libraries
                              AS_IF([test X$use_FFMPEGCDEC = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGCDEC has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCDEC = Xcheck],
                                           [use_FFMPEGCDEC=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGCONT = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGCONT has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCONT = Xcheck],
                                           [use_FFMPEGCONT=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGSWS = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGSWS has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGSWS = Xcheck],
                                           [use_FFMPEGSWS=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGUTIL = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGUTIL has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGUTIL = Xcheck],
                                           [use_FFMPEGUTIL=no]
                                          )
                                    ]
                                   )
                              ;;
             'y' | 'Y' | 'yes' | 'Yes' | 'YES' )
                              with_libffmpeg_pkgconf=yes
                              use_libffmpeg=yes
                              trace 4 --with-libffmpeg-pkgconf enabling FFMPEG libraries
                              AS_IF([test X$use_FFMPEGCDEC = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGCDEC has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCDEC = Xcheck],
                                           [use_FFMPEGCDEC=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGCONT = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGCONT has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCONT = Xcheck],
                                           [use_FFMPEGCONT=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGSWS = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGSWS has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGSWS = Xcheck],
                                           [use_FFMPEGSWS=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGUTIL = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGUTIL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGUTIL = Xcheck],
                                           [use_FFMPEGUTIL=yes]
                                          )
                                    ]
                                   )
                              ;;
             * )              use_libffmpeg=yes
                              trace 4 with-libffmpeg-pkgconf setting pkg-config file for FFMPEG sub-packages to "@<:@$with_libffmpeg_pkgconf@:>@"
                              AS_IF([test X$use_FFMPEGCDEC = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGCDEC has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCDEC = Xcheck],
                                           [with_FFMPEGCDEC_pkgconf=$with_libffmpeg_pkgconf],
                                           [AS_IF([test X$with_FFMPEGCDEC_pkgconf != X$with_libffmpeg_pkgconf],
                                                  [AC_MSG_WARN([--with-FFMPEGCDEC-pkgconf and --with-libffmepg-pkgconf disagree, --with-FFMPEGCDEC-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGCONT = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGCONT has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCONT = Xcheck],
                                           [with_FFMPEGCONT_pkgconf=$with_libffmpeg_pkgconf]
                                           [AS_IF([test X$with_FFMPEGCONT_pkgconf != X$with_libffmpeg_pkgconf],
                                                  [AC_MSG_WARN([--with-FFMPEGCONT-pkgconf and --with-libffmepg-pkgconf disagree, --with-FFMPEGCONT-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGSWS = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGSWS has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGSWS = Xcheck],
                                           [with_FFMPEGSWS_pkgconf=$with_libffmpeg_pkgconf]
                                           [AS_IF([test X$with_FFMPEGSWS_pkgconf != X$with_libffmpeg_pkgconf],
                                                  [AC_MSG_WARN([--with-FFMPEGSWS-pkgconf and --with-libffmepg-pkgconf disagree, --with-FFMPEGSWS-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGUTIL = Xno],
                                    [AC_MSG_FAILURE([Asking to configure libffmpeg via pkg-config but FFMPEGUTIL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGUTIL = Xcheck],
                                           [with_FFMPEGUTIL_pkgconf=$with_libffmpeg_pkgconf]
                                           [AS_IF([test X$with_FFMPEGUTIL_pkgconf != X$with_libffmpeg_pkgconf],
                                                  [AC_MSG_WARN([--with-FFMPEGUTIL-pkgconf and --with-libffmepg-pkgconf disagree, --with-FFMPEGUTIL-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              ;;
             esac
            ],
            [trace 4 No --with-libffmpeg-pkgconf specified so enabling library on implication from the FFMPEG sub-packages
             # Disabling one FFMPEG sub-package disables the superset package
             # enabling all the sub-packages implies the superset is enabled
             AS_IF([test X$use_FFMPEGCDEC = Xno -o X$use_FFMPEGCONT = Xno -o X$use_FFMPEGSWS = Xno -o X$use_FFMPEGUTIL = Xno],
                   [use_libffmpeg=no],
                   [AS_IF([test X$use_FFMPEGCDEC = Xyes -a X$use_FFMPEGCONT = Xyes -a X$use_FFMPEGSWS = Xyes -a X$use_FFMPEGUTIL = Xyes],
                          [use_libffmpeg=yes],
                          [use_libffmpeg=check]
                         )
                   ]
                  )
            ]
           )
AC_ARG_WITH([libffmpeg],
            [AS_HELP_STRING([--with-libffmpeg],
                            [Enable the functionality that relies on the FFMPEG libraries @<:@default=check@:>@.]
                           )
            ],
            [case $with_libffmepg in
             'n' | 'N' | 'no' | 'No' | 'NO' ) 
                              trace 4 --with-libffmpeg disabling FFMPEG libraries
                              # If use_libffmpeg_pkgconf is set to something
                              # other than no, we've already asked for this
                              # library
                              AS_IF([test X$use_libffmpeg_pkgconf != X -a X$use_libffmpeg_pkgconf != Xno ],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg by --with-libffmpeg but --with-libffmpeg-pkgconf has specifically enabled it])
                                    ],
                                    [use_libffmpeg=no]
                                   )
                              # Also we want to ensure we haven't asked for
                              # any of the sub-packages
                              AS_IF([test X$use_FFMPEGCDEC = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGCDEC has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCDEC = Xcheck],
                                           [use_FFMPEGCDEC=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGCONT = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGCONT has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCONT = Xcheck],
                                           [use_FFMPEGCONT=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGSWS = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGSWS has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGSWS = Xcheck],
                                           [use_FFMPEGSWS=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGUTIL = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable libffmpeg but FFMPEGUTIL has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGUTIL = Xcheck],
                                           [use_FFMPEGUTIL=no]
                                          )
                                    ]
                                   )
                              ;;
             'y' | 'Y' | 'yes' | 'Yes' | 'YES' ) 
                              trace 4 --with-libffmpeg enabling FFMPEG libraries
                              AS_IF([test X$use_libffmpeg_pkgconf = Xno],
                                    [AC_MSG_FAILURE([Asking to enable libffmpeg by --with-libffmpeg but --with-libffmpeg-pkgconf has specifically disabled it])
                                    ],
                                    [use_libffmpeg=yes]
                                   )
                              AS_IF([test X$use_FFMPEGCDEC = Xno],
                                    [AC_MSG_FAILURE([Asking to enable libffmpeg but FFMPEGCDEC has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCDEC = Xcheck],
                                           [use_FFMPEGCDEC=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGCONT = Xno],
                                    [AC_MSG_FAILURE([Asking to enable libffmpeg but FFMPEGCONT has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGCONT = Xcheck],
                                           [use_FFMPEGCONT=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGSWS = Xno],
                                    [AC_MSG_FAILURE([Asking to enable libffmpeg but FFMPEGSWS has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGSWS = Xcheck],
                                           [use_FFMPEGSWS=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_FFMPEGUTIL = Xno],
                                    [AC_MSG_FAILURE([Asking to enable libffmpeg but FFMPEGUTIL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_FFMPEGUTIL = Xcheck],
                                           [use_FFMPEGUTIL=yes]
                                          )
                                    ]
                                   )
                              ;;
             'c' | 'C' | 'ck' | 'Ck' | 'CK' | 'check' | 'Check' | 'CHECK' )
                              trace 4 --with-libffmpeg checking for FFMPEG libraries
                              AS_IF([test X$with_libffmpeg_pkgconf != X -o X$use_FFMPEGCDEC != Xcheck -o X$use_FFMPEGCONT != Xcheck -o X$use_FFMPEGSWS != Xcheck -o X$use_FFMPEGUTIL != Xcheck ],
                                    [AC_MSG_WARN([--with-libffmpeg asking to check for the libs but previous settings over-ride this])
                                    ],
                                    [use_libffmpeg=check]
                                   )
                              ;;
               * )            AC_MSG_WARN([--with-libffmpeg set to an unknown option, setting ignored])
                              ;;
             esac
            ],
            [trace 5 No --with-libffmpeg specified]
           )
libffmpeg_SUPPORT=""
libffmpeg_RavlLib="libRavlLibFFmpeg"
AC_SUBST([libffmpeg_SUPPORT])
echo libffmpeg_SUPPORT=@libffmpeg_SUPPORT@ >> ./ExtPkgSupp.$$

# Now we consider OpenGL; full package is named OpenGL while existing
# component libraries are libGL and libGLU
AC_ARG_WITH([OpenGL-pkgconf],
            [AS_HELP_STRING([--with-OpenGL-pkgconf],
                            [Enable the OpenGL libraries via pkg-conf. If declaring a pkg-config control file, this will be used for both libGL and libGLU. You are more likely to want to use the individual command-line options for these two libraries (--with-libGL-pkgconf and -with-libGLU-pkgconf) as the pkg-config control files are usually seperate.]
                           )
            ],
            [case $with_OpenGL_pkgconf in
             'n' | 'N' | 'no' | 'No' | 'NO' ) 
                              with_OpenGL_pkgconf=no
                              use_OpenGL=no
                              trace 4 --with-OpenGL-pkgconf disabling OpenGL libraries
                              AS_IF([test X$use_libGL = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable OpenGL but libGL has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_libGL = Xcheck],
                                           [use_libGL=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_libGLU = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable OpenGL but libGLU has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_libGLU = Xcheck],
                                           [use_libGLU=no]
                                          )
                                    ]
                                   )
                              ;;
             'y' | 'Y' | 'yes' | 'Yes' | 'YES' )
                              with_OpenGL_pkgconf=yes
                              use_OpenGL=yes
                              trace 4 --with-OpenGL-pkgconf enabling OpenGL libraries
                              AS_IF([test X$use_libGL = Xno],
                                    [AC_MSG_FAILURE([Asking to configure OpenGL via pkg-config but libGL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGL = Xcheck],
                                           [use_libGL=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_libGLU = Xno],
                                    [AC_MSG_FAILURE([Asking to configure OpenGL via pkg-config but libGLU has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGLU = Xcheck],
                                           [use_libGLU=yes]
                                          )
                                    ]
                                   )
                              ;;
             * )              use_OpenGL=yes
                              trace 4 with-OpenGL-pkgconf setting pkg-config file for OpenGL libraries to "@<:@$with_OpenGL_pkgconf@:>@"
                              AS_IF([test X$use_libGL = Xno],
                                    [AC_MSG_FAILURE([Asking to configure OpenGL via pkg-config but libGL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGL = Xcheck],
                                           [with_libGL_pkgconf=$with_OpenGL_pkgconf],
                                           [AS_IF([test X$with_libGL_pkgconf != X$with_OpenGL_pkgconf],
                                                  [AC_MSG_WARN([--with-libGL-pkgconf and --with-OpenGL-pkgconf disagree, --with-libGL-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_libGLU = Xno],
                                    [AC_MSG_FAILURE([Asking to configure OpenGL via pkg-config but libGLU has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGLU = Xcheck],
                                           [with_libGLU_pkgconf=$with_OpenGL_pkgconf],
                                           [AS_IF([test X$with_libGLU_pkgconf != X$with_OpenGL_pkgconf],
                                                  [AC_MSG_WARN([--with-libGLU-pkgconf and --with-OpenGL-pkgconf disagree, --with-libGLU-pkgconf takes precidence])
                                                  ]
                                                 )
                                           ]
                                          )
                                    ]
                                   )
                              ;;
             esac
            ],
            [trace 4 No --with-OpenGL-pkgconf specified using value implied from the constituent libraries
             # Disabling one OpenGL sub-library disables the superset, while 
             # enabling all the sub-libraries implies the superset is enabled
             AS_IF([test X$use_libGL = Xno -o X$use_libGLU = Xno],
                   [use_OpenGL=no],
                   [AS_IF([test X$use_libGL = Xyes -a X$use_libGLU = Xyes],
                          [use_OpenGL=yes],
                          [use_OpenGL=check]
                         )
                   ]
                  )
            ]
           )
AC_ARG_WITH([OpenGL],
            [AS_HELP_STRING([--with-OpenGL],
                            [Enable the functionality that relies on the OpenGL libraries @<:@default=check@:>@.]
                           )
            ],
            [case $with_OpenGL in
             'n' | 'N' | 'no' | 'No' | 'NO' ) 
                              trace 4 --with-OpenGL disabling OpenGL libraries
                              # If with_OpenGL_pkgconf is set to something
                              # other than no, we've already asked for this
                              # library
                              AS_IF([test X$with_OpenGL_pkgconf != X -a X$with_OpenGL_pkgconf != Xno ],
                                    [AC_MSG_FAILURE([Asking to disable OpenGL by --with-OpenGL but --with-OpenGL-pkgconf has specifically enabled it])
                                    ],
                                    [use_OpenGL=no]
                                   )
                              # Also we want to ensure we haven't asked for
                              # any of the sub-packages
                              AS_IF([test X$use_libGL = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable OpenGL but libGL has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_libGL = Xcheck],
                                           [use_libGL=no]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_libGLU = Xyes],
                                    [AC_MSG_FAILURE([Asking to disable OpenGL but libGLU has been specifically enabled])
                                    ],
                                    [AS_IF([test X$use_libGLU = Xcheck],
                                           [use_libGLU=no]
                                          )
                                    ]
                                   )
                              ;;
             'y' | 'Y' | 'yes' | 'Yes' | 'YES' ) 
                              trace 4 --with-OpenGL enabling OpenGL libraries
                              AS_IF([test X$with_OpenGL_pkgconf = Xno],
                                    [AC_MSG_FAILURE([Asking to enable OpenGL by --with-OpenGL but --with-OpenGL-pkgconf has specifically disabled it])
                                    ],
                                    [use_OpenGL=yes]
                                   )
                              AS_IF([test X$use_libGL = Xno],
                                    [AC_MSG_FAILURE([Asking to enable OpenGL but libGL has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGL = Xcheck],
                                           [use_libGL=yes]
                                          )
                                    ]
                                   )
                              AS_IF([test X$use_libGLU = Xno],
                                    [AC_MSG_FAILURE([Asking to enable OpenGL but libGLU has been specifically disabled])
                                    ],
                                    [AS_IF([test X$use_libGLU = Xcheck],
                                           [use_libGLU=yes]
                                          )
                                    ]
                                   )
                              ;;
             'c' | 'C' | 'ck' | 'Ck' | 'CK' | 'check' | 'Check' | 'CHECK' )
                              trace 4 --with-OpenGL checking for OpenGL libraries
                              AS_IF([test X$with_OpenGL_pkgconf != X -o X$use_libGL != Xcheck -o X$use_libGLU != Xcheck],
                                    [AC_MSG_WARN([--with-OpenGL asking to check for the libs but previous settings over-ride this])
                                    ],
                                    [use_OpenGL=check]
                                   )
                              ;;
             esac
            ],
            [trace 5 No --with-OpenGL specified]
           )
OpenGL_SUPPORT=""
OpenGL_RavlLib="libRavlDPDisplay3d libRavlGUI3D"
AC_SUBST([OpenGL_SUPPORT])
echo OpenGL_SUPPORT=@OpenGL_SUPPORT@ >> ./ExtPkgSupp.$$

# If debugging, output what we've defined from the cmd-line (with defaults)
if ( test X$DBG != X && test $DBG -ge 3 )
then
   trace 3 Declared libraries
   for lib in $HelperLibs
   do
      eval testwith=\$with_${lib}_fn
      eval prereq=\$with_${lib}_prq
      eval lnklibs=\$with_${lib}_lnklibs
      eval loc=\$with_${lib}_libpath
      eval hdr=\$with_${lib}_hdr
      eval inc=\$with_${lib}_inc
      eval pkgconf=\$with_${lib}_pkgconf
      eval setting=\$use_${lib}
      TRACE_INDENT="     "
      trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$setting
   done
   # Special cases not listed in $HelperLibs:
   TRACE_INDENT="     "
   trace 3 libffmpeg::FFMPEGCEDC FFMPEGCONT FFMPEGUTIL FFMPEGSWS:::::$with_libffmpeg_pkgconf:$use_libffmpeg
   TRACE_INDENT="     "
   trace 3 OpenGL::libGL libGLU:::::$with_OpenGL_pkgconf:$use_OpenGL
fi

# Expand the librarys'pre-requisites to include the pre-requisites for the 
# pre-requisites, etc.
#
# Method: For each library, add the pre-requisites of its direct pre-requisites.
# Repeat this process until we make no further changes.
change_made=true
trace 2 Propogate pre-requisites back through libraries
while $change_made
do
   TRACE_INDENT="     "
   trace 2 Iterating...
   change_made=false
   for lib in $HelperLibs
   do
      TRACE_INDENT="          "
      trace 4 Processing library $lib...
      eval prereq=\$with_${lib}_prq
      # First check for the special case of libffmpeg which is really a 
      # dependancy on all the sub-packages of FFMPEG
      converted=`echo " $prereq " | $SED 's/ libffmpeg / FFMPEGCDEC FFMPEGFMT FFMPEGUTIL FFMPEGSWS /'`
      if ( test " $prereq " != "$converted" )
      then
         TRACE_INDENT="               "
         trace 4 Converting libffmepg dependancy to FFMPEG sub-packages
         AS_VAR_SET(with_${lib}_prq, [$converted])
         prereq=$converted
      fi
      # And check for the case of OpenGL which is really libGL & libGLU
      converted=`echo " $prereq " | $SED 's/ OpenGL / libGL libGLU /'`
      if ( test " $prereq " != "$converted" )
      then
         TRACE_INDENT="               "
         trace 4 Converting OpenGL dependancy to libGL and libGLU
         AS_VAR_SET(with_${lib}_prq, [$converted])
         prereq=$converted
      fi
      for prql in $prereq
      do
         TRACE_INDENT="               "
         trace 4 Adding prerequisites of $prql
         eval prereqs_prql=\$with_${prql}_prq
         for to_add in $prereqs_prql
         do
            if ( test $to_add != $lib )
            then
               eval current_prereqs=\$with_${lib}_prq
               existing=`$EXPR " \$current_prereqs " : ".* \(\$to_add\) .*"`
               if ( test X$existing != X )
               then
                  TRACE_INDENT="                    "
                  trace 6 $to_add already listed
               else
                  current_prereqs=$current_prereqs" "$to_add
                  AS_VAR_SET(with_${lib}_prq, [$current_prereqs])
                  change_made=true
                  TRACE_INDENT="                    "
                  trace 4 $to_add
               fi
            else
               TRACE_INDENT="                    "
               trace 6 Not adding self-reference
            fi
         done
      done
      if ( test X$DBG != X && test $DBG -ge 6 )
      then
         TRACE_INDENT="               "
         trace 6 Pre-requisites now:
         eval current_prereqs=\$with_${lib}_prq
         TRACE_INDENT="                    "
         trace 6 $current_prereqs
      fi
   done
done

# Check that the CMD-line switches provided by the user don't disable libraries 
# that are pre-requisites of enabled libraries
#
# Method: Consider each library in turn and look at its direct pre-requisites.
# Repeat the process to propogate up any blockages. When we change nothing on
# a pass, we should be finished.
# Repeat pass logic is a hangover from an earlier implementation, it should now
# not be needed (as we've moved to full expansion of the pre-requisites where,
# before we did not). It doesn't cost much so code left in "just because..."
change_made=true
trace 2 Find blockages introduced from the cmd-line 
while $change_made
do
   TRACE_INDENT="     "
   trace 2 Iterating...
   change_made=false
   for lib in $HelperLibs
   do
      eval lib_enablement=\$use_${lib}
      case $lib_enablement in
      'blocked' | 'no' )
                TRACE_INDENT="          "
                trace 6 Library $lib disabled
                ;;
      'check')
                eval prereq=\$with_${lib}_prq
                for prql in $prereq
                do
                   TRACE_INDENT="          "
                   trace 4 Prereq of Check lib $lib - $prql
                   eval prql_enablement=\$use_${prql}
                   case $prql_enablement in
                   'blocked' | 'no' )   AS_VAR_SET(use_${lib}, [blocked])
                                        TRACE_INDENT="               "
                                        trace 5 Causes a block
                                        change_made=true
                                        ;;
                   esac
                done
                ;;
      'needed')
                eval prereq=\$with_${lib}_prq
                for prql in $prereq
                do
                   TRACE_INDENT="          "
                   trace 4 Prereq of Needed lib $lib - $prql
                   eval prql_enablement=\$use_${prql}
                   case $prql_enablement in
                   'blocked' )          AC_MSG_FAILURE([Library $lib is required by a specifically enabled library but a specifically disabled library prevents its prerequisite library $prql from being used])
                                        ;;
                   'check' )            AS_VAR_SET(use_${prql}, [needed])
                                        TRACE_INDENT="               "
                                        trace 5 Now needed
                                        change_made=true
                                        ;;
                   'no' )               AC_MSG_FAILURE([Specifically disabled library $prql is required by a specifically enabled library])
                                        ;;
                   esac
                done
                ;;
      'yes')
                eval prereq=\$with_${lib}_prq
                for prql in $prereq
                do
                   TRACE_INDENT="          "
                   trace 4 Prereq of Wanted lib $lib - $prql
                   eval prql_enablement=\$use_${prql}
                   case $prql_enablement in
                   'blocked' )          AC_MSG_FAILURE([Specifically enabled library $lib requires at least one specifically disabled library])
                                        ;;
                   'check' )            AS_VAR_SET(use_${prql}, [needed])
                                        TRACE_INDENT="               "
                                        trace 5 Now needed
                                        change_made=true
                                        ;;
                   'no' )               AC_MSG_FAILURE([Library $prql specifically disabled but required for specifically enabled library $lib])
                                        ;;
                   esac
                done
                ;;
      esac

      if ( test X$DBG != X && test $DBG -ge 3 )
      then
         eval testwith=\$with_${lib}_fn
         eval prereq=\$with_${lib}_prq
         eval lnklibs=\$with_${lib}_lnklibs
         eval loc=\$with_${lib}_libpath
         eval hdr=\$with_${lib}_hdr
         eval inc=\$with_${lib}_inc
         eval pkgconf=\$with_${lib}_pkgconf
         eval setting=\$use_${lib}
         TRACE_INDENT="          "
         trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$setting
      fi
   done
done
# Special cases not on list of helper libs (as set of sub pacakges):
TRACE_INDENT="          "
trace 3 libffmpeg::FFMPEGCEDC FFMPEGCONT FFMPEGUTIL FFMPEGSWS:::::$with_libffmpeg_pkgconf:$use_libffmpeg
TRACE_INDENT="          "
trace 3 OpenGL::libGL libGLU:::::$with_OpenGL_pkgconf:$use_OpenGL

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
      [ RESOURCES+="GnuPlot "
        GnuPlot_SUPPORT=libRavlPlot
      ])
AC_SUBST([GnuPlot])
AC_SUBST([GnuPlot_SUPPORT])
echo GnuPlot_SUPPORT=@GnuPlot_SUPPORT@ >> ./ExtPkgSupp.$$
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
if ( test $ac_cv_search_dlopen != 'none required' )
then
        DynLink_LIBS=$ac_cv_search_dlopen
else
        DynLink_LIBS=""
fi
AC_SUBST([DynLink_LIBS])



# Check for all helper libraries used by Ravl.
#
# First see if each of the libraries exist and error out if the missing libs
# were specified as needed (either from the cmd-line of from dependencies)
#
trace 2 Test availability of libraries
AS_FOR([LIB], [lib], [$HelperLibs],
       [eval lib_enablement=\$use_${lib}
        dnl# Should really be using the macro LIB here instead of ${lib} (as
        dnl# per instructions for AS_FOR). Problem is that it gets expanded
        dnl# only as $lib not ${lib} which causes problems on the next lines
        dnl# Ignoring defined convention currently works in this case; but...
        dnl# We have to use AS_FOR rather than a manual for here as the first 
        dnl# call to check for a library/header brings in additional code to
        dnl# ensure the pre-requisites of the check are available. If we do
        dnl# not use the AS_FOR, this code gets put in the for loop body and
        dnl# is thus multiply executed. Unfortunately, there is only an
        dnl# undocumented _AC_.... macro that can be used to bring in this 
        dnl# prerequisite checking code so we can't use that to bring these
        dnl# checks in at a more useful place
        eval pkgconf=\$with_${lib}_pkgconf
        AS_IF([test X$pkgconf != X],
              [AS_CASE([$lib_enablement],
                       [check],   [AC_MSG_CHECKING(pkg-config for LIB)
                                   PKG_CHECK_EXISTS([$pkgconf],
                                                    [AS_VAR_SET([use_${lib}],
                                                                [present]
                                                               )
                                                     AS_VAR_SET([${lib}_CFLAGS],
                                                                [`$PKG_CONFIG --cflags $pkgconf 2>/dev/null `]
                                                               )
                                                     AS_VAR_SET([${lib}_LIBS],
                                                                [`$PKG_CONFIG --libs $pkgconf 2>/dev/null`]
                                                               )
                                                     AC_MSG_RESULT([yes])
                                                    ],
                                                    [AS_VAR_SET([use_${lib}],
                                                                [missing]
                                                               )
                                                     AC_MSG_RESULT([no])
                                                    ]
                                                  )
                                  ],
                       [needed],  [AC_MSG_CHECKING(pkg-config for LIB)
                                   PKG_CHECK_EXISTS([$pkgconf],
                                                    [AS_VAR_SET([use_${lib}],
                                                                [present]
                                                               )
                                                     AS_VAR_SET([${lib}_CFLAGS],
                                                                [`$PKG_CONFIG --cflags $pkgconf 2>/dev/null `]
                                                               )
                                                     AS_VAR_SET([${lib}_LIBS],
                                                                [`$PKG_CONFIG --libs $pkgconf 2>/dev/null`]
                                                               )
                                                     AC_MSG_RESULT([yes])
                                                    ],
                                                    [AC_MSG_FAILURE([Library $lib is unavailable and a prerequisite of a selected library])
                                                    ]
                                                   )
                                  ],
                       [yes],     [AC_MSG_CHECKING(pkg-config for LIB)
                                   DOTpc_CFLAGS=""
                                   DOTpc_LIBS=""
                                   PKG_CHECK_MODULES(DOTpc, 
                                                     [$pkgconf],
                                                     [AS_VAR_SET([use_${lib}],
                                                                 [present]
                                                                )
                                                     AS_VAR_SET([${lib}_CFLAGS],
                                                                [$DOTpc_CFLAGS]
                                                               )
                                                     AS_VAR_SET([${lib}_LIBS],
                                                                [$DOTpc_LIBS]
                                                               )
dnl                                                  # We use the literal DOTpc
dnl                                                  # as PKG_CHECK_MODULES does
dnl                                                  # not like variables for 
dnl                                                  # the first parameter. 
dnl                                                  # Its name gets printed
dnl                                                  # as part of configure's
dnl                                                  # test for the .pc file
dnl                                                  # hence its name. The
dnl                                                  # downside to all this is
dnl                                                  # that the output from
dnl                                                  # --help will include both
dnl                                                  # DOTpc_CFLAGS and _LIBS
dnl                                                  # as usable variables.
                                                     ],
                                                     [AC_MSG_FAILURE([Required library $lib is unavailable])
                                                     ]
                                                    )
                                  ],
                       [no],      [:],
                       [blocked], [:],
                       [trace 1 "Assert - Should not fall through here"]
                      )
                      if ( test X$DBG != X && test $DBG -ge 3 )
                      then
                         eval testwith=\$with_${lib}_fn
                         eval prereq=\$with_${lib}_prq
                         eval lnklibs=\$with_${lib}_lnklibs
                         eval loc=\$with_${lib}_libpath
                         eval hdr=\$with_${lib}_hdr
                         eval inc=\$with_${lib}_inc
                         eval lib_enablement=\$use_${lib}
                         TRACE_INDENT="     "
                         trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$pkgconf:$inc:$lib_enablement
                      fi
              ],
              [eval testwith=\$with_${lib}_fn
               eval lnklibs=\$with_${lib}_lnklibs
               lnklibs=`echo $lnklibs | $SED 's/^-l//'`
               eval loc=\$with_${lib}_libpath
               eval hdr=\$with_${lib}_hdr
               eval inc=\$with_${lib}_inc
               eval prereq=\$with_${lib}_prq
               AS_CASE([$lib_enablement],
                       [check],  [AX_RAVL_CHECK_FOR_LIB([$lib],
                                                        [$lnklibs], 
                                                        [$testwith], 
                                                        [$loc],
                                                        [$hdr],
                                                        [$inc],
                                                        [AS_VAR_SET([use_${lib}],
                                                                    [present]
                                                                   )
                                                        ],
                                                        [AS_VAR_SET([use_${lib}],
                                                                    [missing]
                                                                   )
                                                        ],
                                                        [$prereq]dnl
                                                       )dnl
                                 ],
                       [needed], [AX_RAVL_CHECK_FOR_LIB([$lib],
                                                        [$lnklibs], 
                                                        [$testwith], 
                                                        [$loc],
                                                        [$hdr],
                                                        [$inc],
                                                        [AS_VAR_SET([use_${lib}],
                                                                    [present]
                                                                   )
                                                        ],
                                                        [AC_MSG_FAILURE([Library $lib is unavailable and a prerequisite of a selected library])],
                                                        [$prereq]dnl
                                                       )dnl
                                 ],
                       [yes],    [AX_RAVL_CHECK_FOR_LIB([$lib],
                                                        [$lnklibs], 
                                                        [$testwith], 
                                                        [$loc],
                                                        [$hdr],
                                                        [$inc],
                                                        [AS_VAR_SET([use_${lib}],
                                                                    [present]
                                                                   )
                                                        ],
                                                        [AC_MSG_FAILURE([Required library $lib is unavailable])],
                                                        [$prereq]dnl
                                                       )dnl
                                  ],
                       [no],      [:],
                       [blocked], [:],
                       [trace 1 "Assert - Should never fall through here"]
                      )
                      if ( test X$DBG != X && test $DBG -ge 3 )
                      then
                         eval lnklibs=\$with_${lib}_lnklibs
                         eval lib_enablement=\$use_${lib}
                         eval pkgconf=\$with_${lib}_pkgconf
                         TRACE_INDENT="     "
                         trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$lib_enablement
                      fi
              ]
             )
       ]
      )



# Once again, FFMPEG is a special case:
#
# At one point in its development; 25/2/8, FFMPEG changed the location of
# its header files. Unfortunately, this was not done with an accompanying
# revision bump. At best, revisioning seems to be:
#
# sub-package       Old layout        Could be either       New Layout
# libavcodec          51.50.0              51.50.1            51.51.0
# libavformat         52.6.0               52.7.0             52.8.0
# libavutil           49.5.0               49.6.0             49.7.0
# libswscale       no guarentee             0.5.0              0.5.1
#
# We therefore need to manually identify which header layout we have and
# ensure that it is consistent across all four of the FFMPEG sub-packages. 
# However, consistency across the sub-packages is only really important if
# we are considering them all as a single entity (as Ravl does). Therefore,
# we only need check for consistency if we have all four packages to group 
# together as the libffmpeg resource.
#
if ( test $use_FFMPEGCDEC = 'present' -a $use_FFMPEGCONT = 'present' -a $use_FFMPEGSWS = 'present' -a $use_FFMPEGUTIL = 'present' )
then
   # To check for the header file consistency, there is a need to determine
   # which flavour of ffmpeg header file we have - old versions were located
   # under a single /usr/include/ffmpeg directory, newer revisions put the
   # headers in component specific dirs. To achieve this, we need to locate a
   # specimin header for each component of ffmpeg in turn and see where it was.

   # First look for the codec header files...
   CDC_HDR=NotFound
   trace 5 Processing FFMPEGCDEC_CFLAGS @<:@$FFMPEGCDEC_CFLAGS@:>@ to determine header layout
   for h in $FFMPEGCDEC_CFLAGS 
   do
      # for each part of the CFLAGS, look for a -Ipath entry and extract the path
      path=`echo $h | $SED -n 's/-I\([[^ ]]*\).*/\1/p'`
      if ( test -n "$path" )
      then
         TRACE_INDENT="     "
         trace 5 Looking for headers under $path
         if ( test -f $path/libavcodec/avcodec.h )
         then
            # This is the newer location
            TRACE_INDENT="          "
            trace 5 New layout found
            if ( test $CDC_HDR = NotFound )
            then
               CDC_HDR=New
            else
               if ( test $CDC_HDR = Old ) 
               then
                  CDC_HDR=Either
               fi
            fi
         else 
            if ( test -f $path/avcodec.h )
            then
               TRACE_INDENT="          "
               trace 5 Old layout found
               if ( test $CDC_HDR = NotFound )
               then
                  CDC_HDR=Old
               else
                  if ( test $CDC_HDR = New )
                  then
                     CDC_HDR=Either
                  fi
               fi
            else
               TRACE_INDENT="          "
               trace 5 No match found
            fi
         fi
      else
         TRACE_INDENT="     "
         trace 6 Discarding $h component of CFLAGS
      fi
   done

   if ( test $CDC_HDR = NotFound )
   then
      # Headers are not in a directory specified by CFLAGS
      # check the default locations for, firstly, the new
      # layout; then the old.
      TRACE_INDENT="     "
      trace 5 Headers not found using CFLAGS now checking standard dirs
      AC_CHECK_HEADER([libavcodec/avcodec.h],
                      [AC_CHECK_HEADER([avcodec.h],
                                       [CDC_HDR=Either],
                                       [CDC_HDR=New],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_CHECK_HEADER([avcodec.h],
                                       [CDC_HDR=Old],
                                       [CDC_HDR=NotFound],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_INCLUDES_DEFAULT]
                     )
   fi 
   trace 4 FFMPEGCDEC headers are $CDC_HDR

   # Now look for the container...
   FMT_HDR=NotFound
   trace 5 Processing FFMPEGCONT_CFLAGS @<:@$FFMPEGCONT_CFLAGS@:>@ to determine header layout
   for h in $FFMPEGCONT_CFLAGS
   do
      # for each part of the CFLAGS, look for a -Ipath entry and extract the path
      path=`echo $h | $SED -n 's/-I\([[^ ]]*\).*/\1/p'`
      if ( test -n "$path" )
      then
         TRACE_INDENT="     "
         trace 5 Looking for headers under $path
         if ( test -f $path/libavformat/avformat.h )
         then
            # This is the newer location
            TRACE_INDENT="          "
            trace 5 New layout found
            if ( test $FMT_HDR = NotFound )
            then
               FMT_HDR=New
            else
               if ( test $FMT_HDR = Old ) 
               then
                  FMT_HDR=Either
               fi
            fi
         else 
            if ( test -f $path/avformat.h )
            then
               TRACE_INDENT="          "
               trace 5 Old layout found
               if ( test $FMT_HDR = NotFound )
               then
                  FMT_HDR=Old
               else
                  if ( test $FMT_HDR = New ) 
                  then
                     FMT_HDR=Either
                  fi
               fi
            else
               TRACE_INDENT="          "
               trace 5 No match found
            fi
         fi
      else
         TRACE_INDENT="     "
         trace 6 Discarding $h component of CFLAGS
      fi
   done

   if ( test $FMT_HDR = NotFound )
   then
      # Headers are not in a directory specified by CFLAGS
      # check the default locations for, firstly, the new
      # layout; then the old.
      TRACE_INDENT="     "
      trace 5 Headers not found using CFLAGS now checking standard dirs
      AC_CHECK_HEADER([libavformat/avformat.h],
                      [AC_CHECK_HEADER([avformat.h],
                                       [FMT_HDR=Either],
                                       [FMT_HDR=New],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_CHECK_HEADER([avformat.h],
                                       [FMT_HDR=Old],
                                       [FMT_HDR=NotFound],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_INCLUDES_DEFAULT]
                     )
   fi 
   trace 4 FFMPEGCONT headers are $FMT_HDR

   # Now ensure the first two are in step
   if ( test $CDC_HDR = Either )
   then
      FFMPEG_HDR_FMT=$FMT_HDR
   else
      if ( test $FMT_HDR = Either )
      then
         FFMPEG_HDR_FMT=$CDC_FMT
      else
         if ( test $CDC_HDR != $FMT_HDR )
         then
            FFMPEG_HDR_FMT=Broken
            trace 5 FFMPEG Headers are broken as the codec and container are different layouts
         else
            FFMPEG_HDR_FMT=$FMT_HDR
         fi
      fi
   fi  

   # Turn our attention to the utilities...
   UTIL_HDR=NotFound
   trace 5 Processing FFMPEGUTIL_CFLAGS @<:@$FFMPEGUTIL_CFLAGS@:>@ to determine header layout
   for h in $FFMPEGUTIL_CFLAGS
   do
      # for each part of the CFLAGS, look for a -Ipath entry and extract the path
      path=`echo $h | $SED -n 's/-I\([[^ ]]*\).*/\1/p'`
      if ( test -n "$path" )
      then
         TRACE_INDENT="     "
         trace 5 Looking for headers under $path
         if ( test -f $path/libavutil/avutil.h )
         then
            # This is the newer location
            TRACE_INDENT="          "
            trace 5 New layout found
            if ( test $UTIL_HDR = NotFound )
            then
               UTIL_HDR=New
            else
               if ( test $UTIL_HDR = Old ) 
               then
                  UTIL_HDR=Either
               fi
            fi
         else 
            if ( test -f $path/avutil.h )
            then
               TRACE_INDENT="          "
               trace 5 Old layout found
               if ( test $UTIL_HDR = NotFound )
               then
                  UTIL_HDR=Old
               else
                  if ( test $UTIL_HDR = New ) 
                  then
                     UTIL_HDR=Either
                  fi
               fi
            else
               TRACE_INDENT="          "
               trace 5 No match found
            fi
         fi
      else
         TRACE_INDENT="     "
         trace 6 Discarding $h component of CFLAGS
      fi
   done

   if ( test $UTIL_HDR = NotFound )
   then
      # Headers are not in a directory specified by CFLAGS
      # check the default locations for, firstly, the new
      # layout; then the old.
      TRACE_INDENT="     "
      trace 5 Headers not found using CFLAGS now checking standard dirs
      AC_CHECK_HEADER([libavutil/avutil.h],
                      [AC_CHECK_HEADER([avutil.h],
                                       [UTIL_HDR=Either],
                                       [UTIL_HDR=New],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_CHECK_HEADER([avutil.h],
                                       [UTIL_HDR=Old],
                                       [UTIL_HDR=NotFound],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_INCLUDES_DEFAULT]
                     )
   fi 
   trace 4 FFMPEGUTIL headers are $UTIL_HDR

   # Check UTIL matches the other headers
   if ( test $FFMPEG_HDR_FMT = Either )
   then
      FFMPEG_HDR_FMT = $UTIL_HDR
   else
      if ( test $FFMPEG_HDR_FMT != $UTIL_HDR )
      then
         if ( test $UTIL_HDR != Either )
         then
            trace 5 FFMPEG Headers are broken as the utils headers differ from the codec and container layout
            FFMPEG_HDR_FMT=Broken
         fi
      fi
   fi

   # Finally, continue on to the s/w Scaling...
   SWS_HDR=NotFound
   trace 5 Processing FFMPEGCDEC_CFLAGS @<:@$FFMPEGCDEC_CFLAGS@:>@ to determine header layout
   for h in $FFMPEGSWS_CFLAGS
   do
      # for each part of the CFLAGS, look for a -Ipath entry and extract the path
      path=`echo $h | $SED -n 's/-I\([[^ ]]*\).*/\1/p'`
      if ( test -n "$path" )
      then
         TRACE_INDENT="     "
         trace 5 Looking for headers under $path
         if ( test -f $path/libswscale/swscale.h )
         then
            # This is the newer location
            TRACE_INDENT="          "
            trace 5 New layout found
            if ( test $SWS_HDR = NotFound )
            then
               SWS_HDR=New
            else
               if ( test $SWS_HDR = Old ) 
               then
                  SWS_HDR=Either
               fi
            fi
         else 
            if ( test -f $path/swscale.h )
            then
               TRACE_INDENT="          "
               trace 5 Old layout found
               if ( test $SWS_HDR = NotFound )
               then
                  SWS_HDR=Old
               else
                  if ( test $SWS_HDR = New ) 
                  then
                     SWS_HDR=Either
                  fi
               fi
            else
               TRACE_INDENT="          "
               trace 5 No match found
            fi
         fi
      else
         TRACE_INDENT="     "
         trace 6 Discarding $h component of CFLAGS
      fi
   done

   if ( test $SWS_HDR = NotFound )
   then
      # Headers are not in a directory specified by CFLAGS
      # check the default locations for, firstly, the new
      # layout; then the old.
      TRACE_INDENT="     "
      trace 5 Headers not found using CFLAGS now checking standard dirs
      AC_CHECK_HEADER([libswscale/swscale.h],
                      [AC_CHECK_HEADER([swscale.h],
                                       [SWS_HDR=Either],
                                       [SWS_HDR=New],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_CHECK_HEADER([swscale.h],
                                       [SWS_HDR=Old],
                                       [SWS_HDR=NotFound],
                                       [AC_INCLUDES_DEFAULT]
                                      )
                      ],
                      [AC_INCLUDES_DEFAULT]
                     )
   fi 
   trace 4 FFMPEGSWS headers are $SWS_HDR

   # Check SWS matches the other headers
   if ( test $FFMPEG_HDR_FMT = Either )
   then
      FFMPEG_HDR_FMT = $SWS_HDR
   else
      if ( test $FFMPEG_HDR_FMT != $SWS_HDR )
      then
         if ( test $SWS_HDR != Either )
         then
            trace 5 FFMPEG Headers are broken as the swscaling headers differ the other components
            FFMPEG_HDR_FMT=Broken
         fi
      fi
   fi

   # If either format is acceptable, force use of the newer layout
   if ( test $FFMPEG_HDR_FMT = Either )
   then
      FFMPEG_HDR_FMT = New
      use_libffmpeg=present
   else
      if ( test $FFMPEG_HDR_FMT = Broken )
      then
         if ( test $use_libffmpeg = yes )
         then
            AC_MSG_FAILURE([Required library libffmpeg has inconsistent headers])
         else
            trace 4 Inconsistent FFMPEG headers block libffmpeg
            use_libffmpeg=blocked
         fi
      else
         if ( test $FFMPEG_HDR_FMT = NotFound )
         then
            # We only actually hit this code if none of the sub-packages'
            # header files are found. If one or more are found and one or
            # more are missing, $FFMPEG_HDR_FMT will be set to 'Broken'
            if ( test $use_libffmpeg = yes )
            then
               AC_MSG_FAILURE([Required library libffmpeg has missing headers])
            else
               trace 4 Missing FFMPEG headers block libffmpeg
               use_libffmpeg=blocked
            fi
         else
            use_libffmpeg=present
         fi
      fi
   fi
else
   if ( test $use_libffmpeg = yes )
   then
      AC_MSG_FAILURE([Required library libffmpeg is unavailable])
   else
      trace 4  We do not have all four FFMPEG sub-pacakges so libffmpeg is blocked
      use_libffmpeg=blocked
   fi
fi


# Special cases - trace those libs that are not on the main list of helper
# libraries and won't have been traced out in the last set of diagnostics
trace 3 libffmpeg::FFMPEGCEDC FFMPEGCONT FFMPEGUTIL FFMPEGSWS:::::$with_libffmpeg_pkgconf:$use_libffmpeg
trace 3 OpenGL::libGL libGLU:::::$with_OpenGL_pkgconf:$use_OpenGL

# Now propogate which libraries are blocked by their prerequisites not being
# present. Repeatedly pass until we get no more alterations. As per the code
# for finding blockages on the command line (see above), the repeated passes
# should no longer be required. However, "if it ain't broke...."
#
# Note: We do not need to generate any errors here when we find a library that
# is blocked by missing prerequisites. This is because any such errors will 
# have been generated (where necessary) in the code checking for the individual
# libraries. (The error is generated when the prerequisite library is found to
# be missing as it knows it is required for another library)
#
trace 2 Block libraries dependent on missing libraries
change_made=true
while $change_made
do
   TRACE_INDENT="     "
   trace 2 Iterating...
   change_made=false
   for lib in $HelperLibs
   do
      eval lib_enablement=\$use_${lib}
      case $lib_enablement in
      'blocked' | 'missing' | 'no' )
                   TRACE_INDENT="          "
                   trace 6 Library $lib not required
                   ;;
      'check' | 'yes' )
                   trace 1 Assertion: --with-${lib} should not equal ${lib_enablement} here!
                   ;;
      'present' )
                   eval prereq=\$with_${lib}_prq
                   for prql in $prereq
                   do
                      TRACE_INDENT="          "
                      trace 4 Testing availability of pre-req $prql of $lib
                      eval prql_enablement=\$use_${prql}
                      case $prql_enablement in
                      'blocked' | 'missing' | 'no' )
                                    AS_VAR_SET(use_${lib}, [blocked])
                                    TRACE_INDENT="               "
                                    trace 5 Causes a block
                                    change_made=true
                                    ;;
                      'check' | 'yes' )
                                    trace 1 Assertion: --with-${prql} should not equal ${prql_enablement} here!
                                    ;;
                      'present' )
                                    TRACE_INDENT="               "
                                    trace 6 Test OK
                                    ;;
                      esac
                   done
                   ;;
      esac
      if ( test X$DBG != X && test $DBG -ge 3 )
      then
         eval testwith=\$with_${lib}_fn
         eval prereq=\$with_${lib}_prq
         eval lnklibs=\$with_${lib}_lnklibs
         eval loc=\$with_${lib}_libpath
         eval hdr=\$with_${lib}_hdr
         eval inc=\$with_${lib}_inc
         eval pkgconf=\$with_${lib}_pkgconf
         eval lib_enablement=\$use_${lib}
         TRACE_INDENT="          "
         trace 3 $lib:$testwith:$prereq:$lnklibs:$loc:$hdr:$inc:$pkgconf:$lib_enablement
      fi
   done
done

# Special cases (sets of helper sub-packages):
TRACE_INDENT="          "
trace 3 libffmpeg::FFMPEGCEDC FFMPEGCONT FFMPEGUTIL FFMPEGSWS:::::$with_libffmpeg_pkgconf:$use_libffmpeg
TRACE_INDENT="          "
trace 3 OpenGL::libGL libGLU:::::$with_OpenGL_pkgconf:$use_OpenGL


#
# Create proforma Config.Project.in
#

# Create initial, empty file
echo > ./Config.Project.in

echo \# Ravl configuration file - regenerate using configure >> ./Config.Project.in
echo >> ./Config.Project.in

# Define location of compilers and helper programs for build system
echo >> ./Config.Project.in
echo \# List of programs required for the build process >> ./Config.Project.in
echo >> ./Config.Project.in
echo AR=@AR@ >> ./Config.Project.in
echo AWK=@AWK@ >> ./Config.Project.in
echo CC=@CC@ >> ./Config.Project.in
echo CP=@CP@ >> ./Config.Project.in
echo CPP=@CPP@ >> ./Config.Project.in
echo CHGRP=@CHGRP@ >> ./Config.Project.in
echo CHMOD=@CHMOD@ >> ./Config.Project.in
echo CHOWN=@CHOWN@ >> ./Config.Project.in
echo CXX=@CXX@ >> ./Config.Project.in
echo CXXCPP=@CXXCPP@ >> ./Config.Project.in
echo EXPR=@EXPR@ >> ./Config.Project.in
echo GET_CWD=@GET_CWD@ >> ./Config.Project.in
echo GREP=@GREP@ >> ./Config.Project.in
echo LD=@LD@ >> ./Config.Project.in
echo LEX=@LEX@ >> ./Config.Project.in
echo LEXLIB=@LEXLIB@ >> ./Config.Project.in
echo LN_S=@LN_S@ >> ./Config.Project.in
echo MAKE=@MAKE@ >> ./Config.Project.in
echo MKDIR_P=@MKDIR_P@ >> ./Config.Project.in
echo RANLIB=@RANLIB@ >> ./Config.Project.in
echo PKG_PAGER=@PKG_PAGER@ >> ./Config.Project.in
echo RM=@RM@ >> ./Config.Project.in
echo SED=@SED@ >> ./Config.Project.in
echo SORT=@SORT@ >> ./Config.Project.in
echo SYNC=@SYNC@ >> ./Config.Project.in
echo TOUCH=@TOUCH@ >> ./Config.Project.in
echo TR=@TR@ >> ./Config.Project.in
echo XARGS=@XARGS@ >> ./Config.Project.in
echo YACC=@YACC@ >> ./Config.Project.in
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
echo \# User specified at configure time >> ./Config.Project.in
echo CONFIGURE_LDFLAGS=@LDFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# General Ravl linker settings >> ./Config.Project.in
echo PKG_CHECK_LDFLAGS=@PKG_CHECK_LDFLAGS@ >> ./Config.Project.in
echo PKG_DEBUG_LDFLAGS=@PKG_DEBUG_LDFLAGS@ >> ./Config.Project.in
echo PKG_DEFAULT_LDFLAGS=@PKG_DEFAULT_LDFLAGS@ >> ./Config.Project.in
echo PKG_GLOBAL_LDFLAGS=@PKG_GLOBAL_LDFLAGS@ >> ./Config.Project.in
echo PKG_GPROF_LDFLAGS=@PKG_GPROF_LDFLAGS@ >> ./Config.Project.in
echo PKG_OPT_LDFLAGS=@PKG_OPT_LDFLAGS@ >> ./Config.Project.in
echo PKG_PROF_LDFLAGS=@PKG_PROF_LDFLAGS@ >> ./Config.Project.in
echo PKG_SHARED_LDFLAGS=@PKG_SHARED_LDFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# Linker settings for Ravl libraries >> ./Config.Project.in
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
echo LIBPATHSWITCH=@LIBPATHSWITCH@ >> ./Config.Project.in
echo USE_LIBPATHSWITCH=$with_rpath >> ./Config.Project.in
echo >> ./Config.Project.in
echo UNDEFSYMB=@UNDEFSYMB@ >> ./Config.Project.in
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
echo CONFIGURE_MAKEFLAGS=@MAKEFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in
echo \# General Ravl make settings >> ./Config.Project.in
echo PKG_MAKEFLAGS=@PKG_MAKEFLAGS@ >> ./Config.Project.in
echo >> ./Config.Project.in



# Allow definition of RESOURCES
echo >> ./Config.Project.in
echo \# List of resources available to Ravl >> ./Config.Project.in
echo >> ./Config.Project.in
echo "RESOURCES=@RESOURCES@" >> ./Config.Project.in
echo >> ./Config.Project.in

# Declare where external executables used by Ravl may be found
#
echo >> ./Config.Project.in
echo \# List of external programs available to Ravl >> ./Config.Project.in
echo >> ./Config.Project.in
# Definition of GnuPlot
echo "GnuPlot=@GnuPlot@" >> ./Config.Project.in
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
# First initialise ALL_EXTERNAL_LIBS here as we need to record these libraries
# as well as the external libs
ALL_EXTERNAL_LIBS=""
echo >> ./Config.Project.in
echo \# Location of system calls that can vary their parent library >> ./Config.Project.in
echo >> ./Config.Project.in
# Location of dlopen
echo "DynLink_LIBS=@DynLink_LIBS@" >> ./Config.Project.in
echo >> ./Config.Project.in
ALL_EXTERNAL_LIBS+=" "$DynLink_LIBS


#
# Now set RESOURCES & ALL_EXTERNAL_LIBS based on the libraries we can use
# Also output each external library's <Lib>_CFLAGS and <Lib>_LIBS definition
# to Config.Project
#
echo >> ./Config.Project.in
echo \# List of external libraries available to Ravl >> ./Config.Project.in
echo >> ./Config.Project.in
trace 2 Calculating RESOURCES settings from library availability
trace 3 \(Initial value of RESOURCES "@<:@$RESOURCES@:>@"\)
for lib in $HelperLibs
do
   # Ravl does not care about the individual components of FFMPEG, only the
   # complete set as indicated by the libffmpeg RESOURCES flag. So... 
   # Quietly drop the FFMPEGxxx RESOURCES from the list
   if ( test $lib != FFMPEGCDEC && test $lib != FFMPEGCONT && test $lib != FFMPEGSWS && test $lib != FFMPEGUTIL )
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
                   if ( test "$with_rpath" = 'Y' )
                   then
                      eval setting=\$${lib}_LIBS
                      setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LIBPATHSWITCH)\=\1 -L\1/g'`
dnl                   # The sed command is actually:
dnl                   #    's/-L *\([^ ][^ ]*\)\$(LIBPATHSWITCH)\=\1 -L\1/g'
dnl                   # the square brackets having been substituted to prevent
dnl                   # m4 removing them. 
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
                      eval ${lib}_LIBS=\$setting
                   fi
                   TRACE_INDENT="     "
                   trace 3 RESOURCES+=${lib}
                   RESOURCES+="${lib} "
                   # Allow the dependant Ravl libs to be listed out to the makefile
                   eval ${lib}_SUPPORT=\$${lib}_RavlLib
                   # Put the placeholders into Config.Project.in so we record the
                   # flags associated with the libraries that we can use
                   echo ${lib}_CFLAGS=@${lib}_CFLAGS@ >> Config.Project.in
                   echo ${lib}_LIBS=@${lib}_LIBS@ >> Config.Project.in
                   echo >> ./Config.Project.in
                   eval liblist=\$${lib}_LIBS
                   ALL_EXTERNAL_LIBS+=" "$liblist
                   ;;
      esac
   else
      eval lib_enablement=\$use_${lib}
      # Even though we don't record the FFMPEG libraries individually, we
      # still need to amend their _LIBS settings if we want to use RPATH (as 
      # we concatenate the individual _LIBS entries to form the single,
      # libffmeg_LIBS setting used in Config.Project)
      if ( test "$with_rpath" = 'Y' )
      then
         if ( test "$lib_enablement" = 'present' )
         then
            eval setting=\$${lib}_LIBS
            setting=`echo $setting | $SED 's/-L *\(@<:@^ @:>@@<:@^ @:>@*\)/\$(LIBPATHSWITCH)\=\1 -L\1/g'`
dnl         # The sed command is actually:
dnl         #    's/-L *\([^ ][^ ]*\)\$(LIBPATHSWITCH)\=\1 -L\1/g'
dnl         # the square brackets having been substituted to prevent m4 removing
dnl         # them. 
            # The sed command looks for:
            #
            #            -L<optional spaces><something><space> 
            # 
            # if this pattern is found, each and every occurrence is replaced by
            # the following (where <something> is the substring found in the
            # original fragment):
            #
            #          $(LIBPATHSWITCH)=<something> -L<something>
            #
            # The rest of the line is left unchanged
            eval ${lib}_LIBS=\$setting
         fi
      fi
      if ( test X$DBG != X && test $DBG -ge 3 )
      then
         TRACE_INDENT="     "
         trace 3 ${lib} $lib_enablement ignored
      fi
   fi
done
trace 3 RESOURCES now "@<:@$RESOURCES@:>@"
trace 3 ALL_EXTERNAL_LIBS now "@<:@$ALL_EXTERNAL_LIBS@:>@"

if ( test X$use_libffmpeg != 'Xno' )
then
   # Set up a single libffmpeg set of flags if all the FFMPEG libraries exist
   if ( test X$use_FFMPEGCDEC = 'Xpresent' -a X$use_FFMPEGCONT = 'Xpresent' -a X$use_FFMPEGSWS = 'Xpresent' -a X$use_FFMPEGUTIL = 'Xpresent' )
   then
      # If we have a consistent set of headers (Old or New) across the FFMPEG 
      # sub-componenets then we can safely employ FFMPEG as a whole
      if ( test X$FFMPEG_HDR_FMT = XNew -o X$FFMPEG_HDR_FMT = XOld )
      then
         if ( test X$FFMPEG_HDR_FMT = XOld )
         then
            echo libffmpeg_CFLAGS=@FFMPEGCDEC_CFLAGS@ @FFMPEGCONT_CFLAGS@ @FFMPEGSWS_CFLAGS@ @FFMPEGUTIL_CFLAGS@ -DFFMPEG_OLD_HDR >> Config.Project.in
         else
            echo libffmpeg_CFLAGS=@FFMPEGCDEC_CFLAGS@ @FFMPEGCONT_CFLAGS@ @FFMPEGSWS_CFLAGS@ @FFMPEGUTIL_CFLAGS@  >> Config.Project.in
         fi
         echo libffmpeg_LIBS=@FFMPEGCDEC_LIBS@ @FFMPEGCONT_LIBS@ @FFMPEGSWS_LIBS@ @FFMPEGUTIL_LIBS@ >> Config.Project.in
         echo >> ./Config.Project.in
         RESOURCES+="libffmpeg "
         libffmpeg_SUPPORT=$libffmpeg_RavlLib
         ALL_EXTERNAL_LIBS+=" ${FFMPEGCDEC_LIBS} ${FFMPEGCONT_LIBS} ${FFMPEGSWS_LIBS} ${FFMPEGUTIL_LIBS}"
         trace 3 Added libffmpeg to RESOURCES 
         trace 3 ALL_EXTERNAL_LIBS now "@<:@$ALL_EXTERNAL_LIBS@:>@"
      else
         trace 3 libffmpeg dropped as FFMPEGxxx have inconsistent header layouts
      fi
   else
      trace 3 libffmpeg disabled as we do not have all the FFMPEGxxx libraries
   fi
else
   trace 4 libffmpeg not added to Config.Project as we did not want it
fi

if ( test X$use_OpenGL != 'Xno' )
then
   # Set up a single OpenGL set of flags if both the GL and GLU libraries exist
   if ( test X$use_libGL = 'Xpresent' -a X$use_libGLU = 'Xpresent' )
   then
      echo OpenGL_CFLAGS=@libGL_CFLAGS@ @libGLU_CFLAGS@ >> Config.Project.in
      echo OpenGL_LIBS=@libGL_LIBS@ @libGLU_LIBS@ >> Config.Project.in
      echo >> ./Config.Project.in
      RESOURCES+="OpenGL "
      OpenGL_SUPPORT=$OpenGL_RavlLib
      trace 3 Added OpenGL to RESOURCES 
      # No need to add to ALL_EXTERNAL_LIBS as libGL and libGLU are already there
   else
      trace 3 OpenGL disabled as we do not have bot libGL and libGLU
   fi
else
   trace 4 OpenGL not added to Config.Project as we did not want it
fi

# finish Config.Project.in external libraries section with ALL_EXTERNAL_LIBS def
echo ALL_EXTERNAL_LIBS=@ALL_EXTERNAL_LIBS@ >> ./Config.Project.in
echo >> ./Config.Project.in


echo >> ./Config.Project.in
echo >> ./Config.Project.in

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
RavlAAM=libRavlAAM
RavlAutoPort=""
# libRavlAutoPort not currently implemented
RavlCameraCal=libRavlCameraCal
RavlChartDetector=libRavlChartDetector
RavlClassWizard=libRavlClassWizard
RavlDataServer=libRavlDataServer
RavlDevAudio=libRavlDevAudio
RavlDPDisplay=libRavlDPDisplay
RavlExtImgIO=libRavlExtImgIO
RavlGUI=libRavlGUI
RavlImageProc=libRavlImageProc
RavlLogic_LIBS="libRavlLogicAgent libRavlLogicNLP libRavlLogic"
RavlNet=libRavlNet
RavlRawVidIO=libRavlRawVidIO
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
   OpenSceneGraph_SUPPORT=""
   LibGnome_SUPPORT=""
   Meteor1_SUPPORT=""
   RavlGUI=""
      # Lack of RavlGUI blocks the following:
         libGuppi_SUPPORT=""
         RavlVDF=""
         RavlVPlay=""
         # Also indirectly blocks LibGnome, OpenGL, OpwnSceneGraph, Ravl2dGUI and
         # libglade2 via dependancy but they are already disabled for their direct
         # dependancy on GTK
   # 2d-GUI support blocked both directly by lack of GTK2 and by lack of RavlGUI
   Ravl2dGUI_LIBS=""
      # Lack of 2d-GUI support blocks the following:
         RavlAAM=""
         #   Blocked by RavlGUIUtil alone (although RavlGUI2D blocks RavlGUIUtil
         #   anyway)
      # Lack of 2d-GUI would also block RavlVDF (blocked by RavlGUI2D missing) and
      # RavlVPlay (blocked by the absense of either of the 2d libs) if they hadn't
      # already been blocked by RavlGUI being omitted
   RavlDPDisplay=""
   # DPDisplay Blocked by lack of GTK2 directly or indirectly by the lack of 
   # either RavlGUI2D or RavlGUI
      # DPDisplay blocks the following:
         MacOSXVideoCapture_SUPPORT=""
         # Also blocks RavlVDF which is already blocked by missing 2d-GUI
      
   trace 3 Lack of libGTK2 blocking Glade Gnome Meteor1 OpenGL 2dGUI etc.
fi

if ( ( test X$use_JPEG != 'Xpresent' ) || ( test X$use_LibTIFF != 'Xpresent' ) || ( test X$use_libPNG != 'Xpresent' ) )
then
   # Lack of Image support blocks libRavlExtImgIO
   RavlExtImgIO=""
   trace 3 Lack of image support blocking RavlExtImgIO
fi

if ( test X$use_libGL != 'Xpresent' )
then
   # Lack of libGL blocks OpenSceneGraph functionality in Ravl
   OpenSceneGraph_SUPPORT=""
   trace 3 Lack of libGL blocking OpenSceneGraph
fi

if ( test X$use_GTKGLExt != 'Xpresent' )
then
   # Lack of GTKGLExt blocks OpenSceneGraph functionality in Ravl and will
   # also alter, but not remove, libRavlGUI3D in its absence
   OpenSceneGraph_SUPPORT=""
   trace 3 Lack of GTKGLExt blocking OpenSceneGraph
fi

if ( test X$use_libmpeg2 != 'Xpresent' )
then
   # Lack of libmpeg2 blocks dvdread functionality in Ravl
   dvdread_SUPPORT=""
   trace 3 Lack of libmpeg2 blocking dvdread
fi


if ( test X$use_RLog != 'Xpresent' )
then
   # Lack of RLog blocks XMPP functionality (which blocks iksemel support)
   RavlXMPP=""
   iksemel_SUPPORT=""
   trace 3 Lack of RLog blocking XMPP and iksemel
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
                      # absence of libRavl3D blocks the OpenGL and OpenSceneGraph support functionality
                         OpenGL_SUPPORT=""
                         OpenSceneGraph_SUPPORT=""

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
                      # Absence of libRavlNet blocks libRavlPythonSwig by dependancy
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
Xsolaris* ) VER_STRING=`$EXPR X$host_os : 'Xsolaris2\.\([[0-9]]*\).*'`
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

# Add, to Config.Project.in, the initialisation of the make variables used by
# SingleLib.mk. These have been recorded in ExtPkgSupp.$$ as this script has
# been running; so, sort this file and append it with a suitable heading.
echo >> ./Config.Project.in
echo \# Optional Ravl sub-libraries that may or may not be included in the >> ./Config.Project.in
echo \# single library>> ./Config.Project.in
$SORT ./ExtPkgSupp.$$ >> ./Config.Project.in
rm ./ExtPkgSupp.$$
echo >> ./Config.Project.in
echo >> ./Config.Project.in
echo >> ./Config.Project.in


dnl Declare RESOURCES and ALL_EXTERNAL_LIBS as outputs
AC_SUBST([RESOURCES])
AC_SUBST([ALL_EXTERNAL_LIBS])

dnl Create Makefiles and *.Project files
AC_CONFIG_FILES([Config.Project])
AC_CONFIG_FILES([Targets.Project:src/AutoTools/Targets.Project.in])
AC_CONFIG_FILES([Makefile:src/AutoTools/Makefile.in])
AC_OUTPUT
