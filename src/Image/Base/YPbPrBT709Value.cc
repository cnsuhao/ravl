
#include "Ravl/Image/YPbPrBT709Value.hh"
#include "Ravl/Image/RGBValue.hh"

namespace RavlImageN {
  
  // Convert a pixel type from RGB to YPbPrBT709
  
  void RGBFloat2YPbPrBT709Float(const RGBValueC<float> &value,YPbPrBT709ValueC<float> &outValue) {
    outValue.Set( + 0.21259f    * value.Red() + 0.71521f * value.Green() + 0.07220f * value.Blue(),
                  - 0.11719f    * value.Red() - 0.39423f * value.Green() + 0.51141f * value.Blue(),
                  + 0.51141f    * value.Red() - 0.46454f * value.Green() - 0.04688f * value.Blue());
  }
  
  // Convert a pixel type from YPbPrBT709 to RGB
  
  void YPbPrBT709Float2RGBFloat(const YPbPrBT709ValueC<float> &value,RGBValueC<float> &outValue) {
    outValue.Set(value.Y() + 0.00000f * value.Pb() + 1.53967f * value.Pr(),
                 value.Y() - 0.18317f * value.Pb() - 0.45764f * value.Pr(),
                 value.Y() + 1.81421f * value.Pb() + 0.00000f * value.Pr());
  }
  
     
}
