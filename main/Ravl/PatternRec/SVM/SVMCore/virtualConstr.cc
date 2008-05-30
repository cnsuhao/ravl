// This file is used in conjunction with RAVL, Recognition And Vision Library
// Copyright (C) 2004, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU
// General Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html
// file-header-ends-here
// $Id: FaceGraph.cc 8674 2006-05-24 12:19:29Z alex $

#include "SvmClassifier.hh"
#include "CommonKernels.hh"
#include "DesignSvm.hh"
#include "DesignClassifierSvmSmo.hh"
#include "SvmLinearClassifier.hh"
#include "SvmQuadraticClassifier.hh"
#include "DesignOneClass.hh"
#include "OneClassLinear.hh"

//#include "Ravl/BinStream.hh"
#include "Ravl/VirtualConstructor.hh"

//need for typename definition
#include "Ravl/DP/FileFormatStream.hh"
#include "Ravl/DP/FileFormatBinStream.hh"
#include "Ravl/DP/Converter.hh"
#include "Ravl/Types.hh"

namespace RavlN
{
  using namespace RavlN;

  void InitRavlSVMIO() {
  }

  // -- DesignSvmC -------------------------------------------------------
  DesignClassifierSupervisedC DesignSvm2DesignClassifierSupervised(const DesignSvmC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(DesignSvm2DesignClassifierSupervised, 1,
                               "RavlN::DesignClassifierSupervisedC RavlN::"
                               "Convert(const RavlN::DesignSvm2DesignCla"
                               "ssifierSupervised &)");

  static TypeNameC TypeDesignSvm(typeid(DesignSvmC), "RavlN::DesignSvmC");

  FileFormatStreamC<DesignSvmC> FileFormatStream_DesignSvm;
  FileFormatBinStreamC<DesignSvmC> FileFormatBinStream_DesignSvm;

  // -- DesignSvmSmoC -------------------------------------------------------
  DesignSvmC DesignSvmSmo2DesignSvm(const DesignSvmSmoC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(DesignSvmSmo2DesignSvm, 1,
                               "RavlN::DesignSvmC RavlN::"
                               "Convert(const RavlN::DesignSvmSmo2DesignSvm &)");

  static TypeNameC TypeDesignSvmSmo(typeid(DesignSvmSmoC),
                                    "RavlN::DesignSvmSmoC");

  FileFormatStreamC<DesignSvmSmoC> FileFormatStream_DesignSvmSmo;
  FileFormatBinStreamC<DesignSvmSmoC> FileFormatBinStream_DesignSvmSmo;

  // -- DesignOneClassC -------------------------------------------------------
  DesignSvmC DesignOneClass2DesignSvm(const DesignOneClassC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(DesignOneClass2DesignSvm, 1,
                               "RavlN::DesignSvmC RavlN::"
                               "Convert(const RavlN::DesignOneClass2DesignSvm &)");

  static TypeNameC TypeDesignOneClass(typeid(DesignOneClassC),
                                      "RavlN::DesignOneClassC");

  FileFormatStreamC<DesignOneClassC> FileFormatStream_DesignOneClass;
  FileFormatBinStreamC<DesignOneClassC> FileFormatBinStream_DesignOneClass;

  // -- Classifier2C ---------------------------------------------------------
  ClassifierC Classifier2ToClassifier(const Classifier2C &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(Classifier2ToClassifier ,1,
                               "RavlN::ClassifierC RavlN::Convert(const "
                               "RavlN::Classifier2ToClassifier &)");

  static TypeNameC TypeClassifier2(typeid(Classifier2C),
                                   "RavlN::Classifier2C");

  FileFormatStreamC<Classifier2C> FileFormatStream_Classifier2;
  FileFormatBinStreamC<Classifier2C> FileFormatBinStream_Classifier2;

  // -- SvmClassifierC -------------------------------------------------------
  Classifier2C SvmClassifier2Classifier2(const SvmClassifierC &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(SvmClassifier2Classifier2 ,1,
                               "RavlN::Classifier2C RavlN::Convert(const "
                               "RavlN::SvmClassifier2Classifier2 &)");

  static TypeNameC TypeSvmClassifier(typeid(SvmClassifierC),
                                     "RavlN::SvmClassifierC");

  FileFormatStreamC<SvmClassifierC> FileFormatStream_SvmClassifier;
  FileFormatBinStreamC<SvmClassifierC> FileFormatBinStream_SvmClassifier;

  // -- SvmLinearClassifierC -------------------------------------------------
  Classifier2C SvmLinearClassifier2Classifier2(const SvmLinearClassifierC &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(SvmLinearClassifier2Classifier2 ,1,
                               "RavlN::Classifier2C RavlN::Convert(const "
                               "RavlN::SvmLinearClassifier2Classifier2 &)");

  static TypeNameC TypeSvmLinearClassifier(typeid(SvmLinearClassifierC),
                                           "RavlN::SvmLinearClassifierC");

  FileFormatStreamC<SvmLinearClassifierC> FileFormatStream_SvmLinearClassifier;
  FileFormatBinStreamC<SvmLinearClassifierC> FileFormatBinStream_SvmLinearClassifier;

  // -- SvmQuadraticClassifierC -------------------------------------------------
  Classifier2C SvmQuadraticClassifier2Classifier2(const SvmQuadraticClassifierC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(SvmQuadraticClassifier2Classifier2 ,1,
                               "RavlN::Classifier2C RavlN::Convert(const "
                               "RavlN::SvmQuadraticClassifier2Classifier2 &)");

  static TypeNameC TypeSvmQuadraticClassifier(typeid(SvmQuadraticClassifierC),
                                              "RavlN::SvmQuadraticClassifierC");

  FileFormatStreamC<SvmQuadraticClassifierC> FileFormatStream_SvmQuadraticClassifier;
  FileFormatBinStreamC<SvmQuadraticClassifierC> FileFormatBinStream_SvmQuadraticClassifier;

  // -- OneClassC -------------------------------------------------------
  Classifier2C OneClass2Classifier2(const OneClassC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(OneClass2Classifier2 ,1,
                               "RavlN::Classifier2C RavlN::Convert(const "
                               "RavlN::OneClass2Classifier2 &)");

  static TypeNameC TypeOneClass(typeid(OneClassC), "RavlN::OneClassC");

  FileFormatStreamC<OneClassC> FileFormatStream_OneClass;
  FileFormatBinStreamC<OneClassC> FileFormatBinStream_OneClass;

  // -- OneClassLinearC -------------------------------------------------------
  Classifier2C OneClassLinear2Classifier2(const OneClassLinearC &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(OneClassLinear2Classifier2 ,1,
                               "RavlN::Classifier2C RavlN::Convert(const "
                               "RavlN::OneClassLinear2Classifier2 &)");

  static TypeNameC TypeOneClassLinear(typeid(OneClassLinearC), "RavlN::OneClassLinearC");

  FileFormatStreamC<OneClassLinearC> FileFormatStream_OneClassLinear;
  FileFormatBinStreamC<OneClassLinearC> FileFormatBinStream_OneClassLinear;

  // -- Linear kernel -------------------------------------------------------
  KernelFunctionC LinearKernel2KernelFunction(const LinearKernelC &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(LinearKernel2KernelFunction, 1,
                               "RavlN::KernelFunctionC RavlN::Convert(const "
                               "RavlN::LinearKernel2KernelFunction &)");

  static TypeNameC TypeLinearKernel(typeid(LinearKernelC),
                                    "RavlN::LinearKernelC");

  FileFormatStreamC<LinearKernelC> FileFormatStream_LinearKernel;
  FileFormatBinStreamC<LinearKernelC> FileFormatBinStream_LinearKernel;

  // -- Quadratic kernel -------------------------------------------------------
  KernelFunctionC QuadraticKernel2KernelFunction(const QuadraticKernelC &func)
    { return func; }

  DP_REGISTER_CONVERSION_NAMED(QuadraticKernel2KernelFunction, 1,
                               "RavlN::KernelFunctionC RavlN::Convert(const "
                               "RavlN::QuadraticKernel2KernelFunction &)");

  static TypeNameC TypeQuadraticKernel(typeid(QuadraticKernelC),
                                       "RavlN::QuadraticKernelC");

  FileFormatStreamC<QuadraticKernelC> FileFormatStream_QuadraticKernel;
  FileFormatBinStreamC<QuadraticKernelC> FileFormatBinStream_QuadraticKernel;

  // -- Polynomial kernel ---------------------------------------------------
  KernelFunctionC PolynomialKernel2KernelFunction(const PolynomialKernelC &f)
  { return f; }

  DP_REGISTER_CONVERSION_NAMED(PolynomialKernel2KernelFunction, 1,
                               "RavlN::KernelFunctionC RavlN::Convert(const "
                               "RavlN::PolynomialKernel2KernelFunction &)");

  static TypeNameC TypePolynomialKernel(typeid(PolynomialKernelC),
                                        "RavlN::PolynomialKernelC");

  FileFormatStreamC<PolynomialKernelC> FileFormatStream_PolynomialKernel;
  FileFormatBinStreamC<PolynomialKernelC> FileFormatBinStream_PolynomialKernel;

  // -- RBF kernel ---------------------------------------------------
  KernelFunctionC RBFKernel2KernelFunction(const RBFKernelC &func)
  { return func; }

  DP_REGISTER_CONVERSION_NAMED(RBFKernel2KernelFunction, 1,
                               "RavlN::KernelFunctionC RavlN::Convert(const "
                               "RavlN::RBFKernel2KernelFunction &)");

  static TypeNameC TypeRBFKernel(typeid(RBFKernelC), "RavlN::RBFKernelC");

  FileFormatStreamC<RBFKernelC> FileFormatStream_RBFKernel;
  FileFormatBinStreamC<RBFKernelC> FileFormatBinStream_RBFKernel;
}

using namespace RavlN;
using namespace RavlN;
//---------------------------------------------------------------------------
// Stream load operators defined in RAVL_INITVIRTUALCONSTRUCTOR_FULL macro
// Implementation of 'load from stream' constructors defined there as well
RAVL_INITVIRTUALCONSTRUCTOR_FULL(LinearKernelBodyC,        LinearKernelC,        KernelFunctionC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(QuadraticKernelBodyC,     QuadraticKernelC,     KernelFunctionC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(PolynomialKernelBodyC,    PolynomialKernelC,    KernelFunctionC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(RBFKernelBodyC,           RBFKernelC,           KernelFunctionC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignSvmBodyC,           DesignSvmC,           DesignClassifierSupervisedC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignSvmSmoBodyC,        DesignSvmSmoC,        DesignSvmC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(Classifier2BodyC,         Classifier2C,         ClassifierC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(SvmClassifierBodyC,       SvmClassifierC,       Classifier2C);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(SvmLinearClassifierBodyC, SvmLinearClassifierC, Classifier2C);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(SvmQuadraticClassifierBodyC, SvmQuadraticClassifierC, Classifier2C);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(DesignOneClassBodyC,      DesignOneClassC,      DesignSvmC);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(OneClassBodyC,            OneClassC,            Classifier2C);
RAVL_INITVIRTUALCONSTRUCTOR_FULL(OneClassLinearBodyC,      OneClassLinearC,      Classifier2C);

