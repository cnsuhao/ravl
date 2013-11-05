# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2006-2012, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

PACKAGE = Ravl/fann

LICENSE=own

HEADERS=fann.h compat_time.h fann_activation.h fann_error.h fann_io.h floatfann.h \
 fann_cascade.h fann_train.h doublefann.h fann_data.h fann_internal.h fixedfann.h \
 fann_common.h

LOCALHEADERS=config.h fann.c fann_io.c fann_train.c fann_train_data.c \
             fann_error.c fann_cascade.c

SOURCES = floatfann.c

PLIB = fann

SUMMARY_LIB=Ravl

USESLIBS=RavlCore 
