   
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

%include "Ravl/Swig2/Macros.i"
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
%include "Ravl/Swig2/RCHash.i"
%include "Ravl/Swig2/DPIPort.i"
%include "Ravl/Swig2/SmartPtr.i"
%include "Ravl/Swig2/SysLog.i"
%include "Ravl/Swig2/XMLFactory.i"
%include "Ravl/Swig2/Collection.i"
%include "Ravl/Swig2/LoadSave.i"

