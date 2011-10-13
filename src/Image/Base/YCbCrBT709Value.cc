
#include "Ravl/Image/YCbCrBT709Value.hh"
#include "Ravl/Image/RGBValue.hh"

namespace RavlImageN {
  
  // Convert a pixel type from RGB to YCbCrBT709
  
  void RGBFloat2YCbCrBT709Float(const RGBValueC<float> &value,YCbCrBT709ValueC<float> &outValue) {
    outValue.Set( + 0.21259f * value.Red() + 0.71521f * value.Green() + 0.07220f * value.Blue(),
                  - 0.11719f * value.Red() - 0.39423f * value.Green() + 0.51141f * value.Blue(),
                  + 0.51141f * value.Red() - 0.46454f * value.Green() - 0.04688f * value.Blue());
  }
  
  // Convert a pixel type from YCbCrBT709 to RGB
  
  void YCbCrBT709Float2RGBFloat(const YCbCrBT709ValueC<float> &value,RGBValueC<float> &outValue) {
    outValue.Set(value.Y() + 0.00000f * value.Cb() + 1.53967f * value.Cr(),
                 value.Y() - 0.18317f * value.Cb() - 0.45764f * value.Cr(),
                 value.Y() + 1.81421f * value.Cb() + 0.00000f * value.Cr());
  }
  
     
}
