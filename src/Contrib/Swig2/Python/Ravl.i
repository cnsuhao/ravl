// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2006, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html

// Set the Python module 'docstring'
%define RAVL_DOCSTRING
"Recognition and Vision Library
RAVL provides a base C++ class library together with a range of
computer vision, pattern recognition and supporting tools."
%enddef

%module(docstring=RAVL_DOCSTRING) Ravl

// Enable basic Python automatic 'docstring' entries for  
// function arguments and return values
%feature("autodoc", "0");
        
%{
namespace RavlGUIN
{
  extern void InitDPWindowFormat();
  extern void InitDPDisplayImage();
  void XInitRavlGUIDisplay()
  {
    InitDPWindowFormat();
    InitDPDisplayImage();
  }
}
%}

%include "Ravl/Swig2/Types.i"
%include "Ravl/Swig2/Size.i"
%include "Ravl/Swig2/Point2d.i"
%include "Ravl/Swig2/IndexRange.i"
%include "Ravl/Swig2/Image.i"
%include "Ravl/Swig2/Polygon2d.i"
%include "Ravl/Swig2/String.i"
%include "Ravl/Swig2/Font.i"
%include "Ravl/Swig2/Array2d.i"
%include "Ravl/Swig2/SArray1d.i"
%include "Ravl/Swig2/SArray2d.i"
%include "Ravl/Swig2/TVector.i"
%include "Ravl/Swig2/Vector.i"
%include "Ravl/Swig2/TMatrix.i"
%include "Ravl/Swig2/Matrix.i"
%include "Ravl/Swig2/MatrixRUTC.i"
%include "Ravl/Swig2/VectorMatrix.i"
%include "Ravl/Swig2/MeanCovariance.i"
%include "Ravl/Swig2/Affine2d.i"
%include "Ravl/Swig2/DList.i"
%include "Ravl/Swig2/Date.i"
%include "Ravl/Swig2/PointSet2d.i"
%include "Ravl/Swig2/RealRange.i"
%include "Ravl/Swig2/RealRange2d.i"
%include "Ravl/Swig2/Hash.i"
%include "Ravl/Swig2/DPIPort.i"
%include "Ravl/Swig2/SmartPtr.i"
%include "Ravl/Swig2/SysLog.i"


%include "Ravl/Swig2/Function.i"
%include "Ravl/Swig2/Sample.i"
%include "Ravl/Swig2/SampleVector.i"
%include "Ravl/Swig2/SampleLabel.i"
%include "Ravl/Swig2/DataSetBase.i"
%include "Ravl/Swig2/DataSet1.i"
%include "Ravl/Swig2/DataSet2.i"
%include "Ravl/Swig2/DataSetVectorLabel.i"
%include "Ravl/Swig2/DataSetIO.i"
%include "Ravl/Swig2/Classifier.i"
%include "Ravl/Swig2/ClassifierDiscriminantFunction.i"
%include "Ravl/Swig2/DesignFunctionSupervised.i"
%include "Ravl/Swig2/DesignFuncLSQ.i"
%include "Ravl/Swig2/DesignClassifierSupervised.i"
%include "Ravl/Swig2/DesignDiscriminantFunction.i"
%include "Ravl/Swig2/DesignClassifierNeuralNetwork2.i"
%include "Ravl/Swig2/DesignClassifierLogisticRegression.i"
%include "Ravl/Swig2/Error.i"
%include "Ravl/Swig2/GnuPlot2d.i"


%include "Ravl/Swig2/LoadSave.i"

