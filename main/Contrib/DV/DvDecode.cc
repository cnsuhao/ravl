#include"Ravl/Image/DvDecode.hh"
#include"Ravl/Image/Deinterlace.hh"

namespace RavlImageN {

DvDecodeBodyC::DvDecodeBodyC (bool deint)
  : init(false), deinterlace(deint)
{
  decoded = (uint8_t*)malloc(576*720*3);
  decoder = dv_decoder_new(false, true, true);
  dv_init(true, true);
  decoder->quality = DV_QUALITY_BEST; // for know we will make it highest
}


ImageC<ByteRGBValueC>
DvDecodeBodyC::Apply(ByteT *data)
{
  
  uint8_t *pixels[3];
  int pitches[3];
  
  pixels[0] = decoded;
  pixels[1] = NULL;
  pixels[2] = NULL;

  pitches[0] = 720 * 3;
  pitches[1] = 0;
  pitches[2] = 0;

  if(!init) {
    dv_parse_header(decoder, (uint8_t*)data);
    init=true;
    cout << "Audio is: " <<  decoder->audio->frequency / 1000.0 << "kHz" << endl;
    cout << decoder->audio->quantization << " bits quantization" << endl;
    cout << decoder->audio->num_channels << " channels" << endl;
  }
  
  dv_decode_full_frame(decoder, data, e_dv_color_rgb, pixels, pitches);
  ImageC<ByteRGBValueC>im(576, 720, (ByteRGBValueC*)&pixels[0][0], false);

  if(deinterlace) im = DeinterlaceSubsample(im);

  return im;
}


} // end namespace RavlImageN

