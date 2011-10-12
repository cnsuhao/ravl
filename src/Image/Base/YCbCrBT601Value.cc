
#include "Ravl/Image/YCbCrBT601Value.hh"
#include "Ravl/Image/RGBValue.hh"

namespace RavlImageN {
  
  // Convert a pixel type from RGB to YCbCrBT601
  
  void RGBFloat2YCbCrBT601Float(const RGBValueC<float> &value,YCbCrBT601ValueC<float> &outValue) {
    outValue.Set( + 0.299f    * value.Red() + 0.587f    * value.Green() + 0.114f    * value.Blue(),
                  - 0.168736f * value.Red() - 0.331264f * value.Green() + 0.5f      * value.Blue(),
                  + 0.5f      * value.Red() - 0.418688f * value.Green() - 0.081312f * value.Blue());
  }
  
  // Convert a pixel type from YCbCrBT601 to RGB
  
  void YCbCrBT601Float2RGBFloat(const YCbCrBT601ValueC<float> &value,RGBValueC<float> &outValue) {
    outValue.Set(value.Y() - 0.1218894199e-5f * value.Cb() + 1.4019995886f    * value.Cr(),
                 value.Y() - 0.3441356781f    * value.Cb() - 0.7141361555f    * value.Cr(),
                 value.Y() + 1.772000066f     * value.Cb() + 0.4062980664e-6f * value.Cr());
  }
  
  
}
