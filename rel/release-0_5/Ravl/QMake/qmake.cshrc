
# Where is QMake installed ?
setenv QMAKE_INSTALL /vol/vssp/local

#define project area if not already set (for binaries, libs, *.hh etc.)
if (! $?PROJECT_OUT) setenv PROJECT_OUT $HOME/build

if (-x $QMAKE_INSTALL/config.arc) then ;
  setenv ARC `$QMAKE_INSTALL/config.arc`
  # add project binaries dir to path
  set path = ( $path $PROJECT_OUT/$ARC/bin )
else
  echo "Could not set up QMAKE environment correctly"
endif

# The USERBUILD=1 in the following causes USERLIBS = Auto to be used by default.

# can use this alias for making
alias qm "gmake -f $QMAKE_INSTALL/QMake.mk USERBUILD=1 \!* && rehash"
