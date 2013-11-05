# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2002, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
# file-header-ends-here

PACKAGE=Ravl

HEADERS=KalmanFilter.hh LinearKalmanFilter.hh ExtendedKalmanFilter.hh \
 KalmanTwoWheelDifferentialMotionModel.hh KalmanNullMeasurementModel.hh 

SOURCES=KalmanFilter.cc LinearKalmanFilter.cc ExtendedKalmanFilter.cc \
 KalmanTwoWheelDifferentialMotionModel.cc KalmanNullMeasurementModel.cc 

TESTEXES = test_kalman.cc 

PLIB=RavlKalmanFilter

SUMMARY_LIB=Ravl

USESLIBS=RavlCore  RavlPatternRec
