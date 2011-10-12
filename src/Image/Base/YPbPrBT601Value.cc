
#include "Ravl/Image/YPbPrBT601Value.hh"
#include "Ravl/Image/RGBValue.hh"

namespace RavlImageN {
  
  // Convert a pixel type from RGB to YPbPrBT601
  
  void RGBFloat2YPbPrBT601Float(const RGBValueC<float> &value,YPbPrBT601ValueC<float> &outValue) {
    outValue.Set(+ 0.299000f * value.Red()  + 0.587000f * value.Green()  + 0.114000f * value.Blue(),
                 - 0.168736f * value.Red()  - 0.331264f * value.Green()  + 0.500000f * value.Blue(),
                 + 0.500000f * value.Red()  - 0.418688f * value.Green()  - 0.081312f * value.Blue());
  }
  
  // Convert a pixel type from YPbPrBT601 to RGB
  
  void YPbPrBT601Float2RGBFloat(const YPbPrBT601ValueC<float> &value,RGBValueC<float> &outValue) {
    outValue.Set(value.Y()  + 0.000000f * value.Pb()  + 1.402000f * value.Pr(),
                 value.Y()  - 0.344136f * value.Pb()  - 0.714136f * value.Pr(),
                 value.Y()  + 1.814210f * value.Pb()  + 0.000000f * value.Pr());
  }
  
     
}
