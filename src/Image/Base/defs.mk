# This file is part of RAVL, Recognition And Vision Library 
# Copyright (C) 2001-11, University of Surrey
# This code may be redistributed under the terms of the GNU Lesser
# General Public License (LGPL). See the lgpl.licence file for details or
# see http://www.gnu.org/copyleft/lesser.html
#
# file-header-ends-here

PACKAGE=Ravl/Image

HEADERS=ImageRectangle.hh Image.hh \
 RGBValue.hh ByteRGBValue.hh ByteRGBColour.hh UInt16RGBValue.hh RealRGBValue.hh \
 RGBAValue.hh ByteRGBAValue.hh RealRGBAValue.hh \
 BGRValue.hh ByteBGRValue.hh RealBGRValue.hh BGRAValue.hh ByteBGRAValue.hh \
 YUVValue.hh ByteYUVValue.hh RealYUVValue.hh \
 YUVAValue.hh ByteYUVAValue.hh \
 YUV422Value.hh ByteYUV422Value.hh \
 VYUValue.hh ByteVYUValue.hh \
 RealHSVValue.hh FixedPointHSVValue.hh \
 IAValue.hh ByteIAValue.hh \
 ByteRGBMedian.hh RealRGBAverage.hh \
 ScaleValues.hh Reflect.hh \
 RGBcYUV.hh ImageConv.hh \
 Font.hh PSFFont.h \
 DrawSymbol.hh DrawFrame.hh DrawCross.hh DrawMask.hh \
 DrawLine.hh DrawCircle.hh DrawPolygon.hh  DrawEllipse.hh\
 BilinearInterpolation.hh \
 YCbCrBT601Value.hh ByteYCbCrBT601Value.hh UInt16YCbCrBT601Value.hh \
 YCbCr422BT601Value.hh ByteYCbCr422BT601Value.hh UInt16YCbCr422BT601Value.hh \
 YCbCrBT709Value.hh ByteYCbCrBT709Value.hh UInt16YCbCrBT709Value.hh \
 YCbCr422BT709Value.hh ByteYCbCr422BT709Value.hh UInt16YCbCr422BT709Value.hh \
 YPbPrBT601Value.hh FloatYPbPrBT601Value.hh \
 YPbPrBT709Value.hh FloatYPbPrBT709Value.hh \
 YPbPr422BT601Value.hh FloatYPbPr422BT601Value.hh \
 YPbPr422BT709Value.hh FloatYPbPr422BT709Value.hh

SOURCES=ImageRectangle.cc Image.cc ByteRGBValue.cc ByteYUVValue.cc RealRGBValue.cc \
 RealYUVValue.cc RGBcYUV.cc Font.cc ImageConv.cc ImageConv2.cc ImageConv3.cc ImageConv4.cc \
 ImageConv5.cc ByteRGBAValue.cc ByteVYUValue.cc ByteYUV422Value.cc ByteYUVAValue.cc \
 ImgCnvRGB.cc \
 YCbCrBT601Value.cc ByteYCbCrBT601Value.cc UInt16YCbCrBT601Value.cc \
 ByteYCbCr422BT601Value.cc UInt16YCbCr422BT601Value.cc \
 YCbCrBT709Value.cc ByteYCbCrBT709Value.cc UInt16YCbCrBT709Value.cc \
 ByteYCbCr422BT709Value.cc UInt16YCbCr422BT709Value.cc \
 YPbPrBT601Value.cc FloatYPbPrBT601Value.cc \
 YPbPrBT709Value.cc FloatYPbPrBT709Value.cc \
 FloatYPbPr422BT709Value.cc FloatYPbPr422BT601Value.cc

PLIB=RavlImage

SUMMARY_LIB=Ravl

USESLIBS=RavlCore RavlIO RavlMath

PROGLIBS=RavlImageIO RavlOS RavlMath RavlDPDisplay.opt

MAINS=imgdiff.cc

TESTEXES=testImage.cc testHSVValue.cc test_imgdiff.cc

EXAMPLES= exImage.cc imgdiff.cc exDraw.cc exFont.cc

EHT=Ravl.API.Images.Pixel_Types.html Drawing.html YUV_Pixel_Types.html YCbCr_Pixel_Types.html YPbPr_Pixel_Types.html Converters.html

AUXDIR=share/RAVL/Fonts

AUXFILES=default8x16.psf
