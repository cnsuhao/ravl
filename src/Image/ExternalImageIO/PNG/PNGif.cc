// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//

#define RAVLIMAGE_PNGIF_CCFILE   1
// Prevents extern definition of Png in PNGif.hh

#include <dlfcn.h>
#include <png.h>

#ifndef png_jmpbuf
 /* The png_jmpbuf() macro, used in error handling, became available in
  * libpng version 1.0.6.  If you want to be able to run your code with older
  * versions of libpng, you must define the macro yourself (but only if it
  * is not already defined by libpng!).
  */
# define png_jmpbuf(png_ptr) ((png_ptr)->jmpbuf)
#endif

#include "Ravl/DP/DynamicLink.hh"
#include "Ravl/Image/PNGif.hh"
#include "Ravl/Image/PNGFormat.hh"


namespace RavlImageN {

   using namespace RavlN; 

   PNGif::PNGif ()
   {
      void * handle;

      if ( (handle = DynamicLinkLoad ("libpng.so", true)) )
      {   /* Assumes loading libpng.so automatically brings in libz.so */
         png_create_info_structFP.AsVoidPtr = dlsym (handle, "png_create_info_struct");
         png_create_read_structFP.AsVoidPtr = dlsym (handle, "png_create_read_struct");
         png_create_write_structFP.AsVoidPtr = dlsym (handle, "png_create_write_struct");
         png_destroy_read_structFP.AsVoidPtr = dlsym (handle, "png_destroy_read_struct");
         png_destroy_write_structFP.AsVoidPtr = dlsym (handle, "png_destroy_write_struct");
         png_errorFP.AsVoidPtr = dlsym (handle, "png_error");
         png_get_IHDRFP.AsVoidPtr = dlsym (handle, "png_get_IHDR");
         png_get_io_ptrFP.AsVoidPtr = dlsym (handle, "png_get_io_ptr");
         png_get_sBITFP.AsVoidPtr = dlsym (handle, "png_get_sBIT");
         png_get_validFP.AsVoidPtr = dlsym (handle, "png_get_valid");
         png_read_endFP.AsVoidPtr = dlsym (handle, "png_read_end");
         png_read_infoFP.AsVoidPtr = dlsym (handle, "png_read_info");
         png_read_rowsFP.AsVoidPtr = dlsym (handle, "png_read_rows");
         png_read_update_infoFP.AsVoidPtr = dlsym (handle, "png_read_update_info");
         png_set_expandFP.AsVoidPtr = dlsym (handle, "png_set_expand");
         png_set_fillerFP.AsVoidPtr = dlsym (handle, "png_set_filler");
         png_set_gray_to_rgbFP.AsVoidPtr = dlsym (handle, "png_set_gray_to_rgb");
         png_set_interlace_handlingFP.AsVoidPtr = dlsym (handle, "png_set_interlace_handling");
         png_set_IHDRFP.AsVoidPtr = dlsym (handle, "png_set_IHDR");
         png_set_packingFP.AsVoidPtr = dlsym (handle, "png_set_packing");
         png_set_read_fnFP.AsVoidPtr = dlsym (handle, "png_set_read_fn");
         png_set_rgb_to_gray_fixedFP.AsVoidPtr = dlsym (handle, "png_set_rgb_to_gray_fixed");
         png_set_shiftFP.AsVoidPtr = dlsym (handle, "png_set_shift");
         png_set_strip_16FP.AsVoidPtr = dlsym (handle, "png_set_strip_16");
         png_set_strip_alphaFP.AsVoidPtr = dlsym (handle, "png_set_strip_alpha");
         png_set_textFP.AsVoidPtr = dlsym (handle, "png_set_text");
         png_set_write_fnFP.AsVoidPtr = dlsym (handle, "png_set_write_fn");
         png_set_swapFP.AsVoidPtr = dlsym (handle, "png_set_swap");
         png_write_endFP.AsVoidPtr = dlsym (handle, "png_write_end");
         png_write_infoFP.AsVoidPtr = dlsym (handle, "png_write_info");
         png_write_rowsFP.AsVoidPtr = dlsym (handle, "png_write_rows");

         RegisterFileFormatPNG = new FileFormatPNGC (false,"png","General png file IO. ");
         RegisterFileFormatPNG16 = new FileFormatPNGC (true,"png16","16 bit only png file IO. ");
      }
   }
         // Default constructor

   PNGif::~PNGif ()
   { }
         // Standard destructor

   png_infop PNGif::create_info_struct (png_structp png_ptr)
   {
      if (png_create_info_structFP.Call != NULL)
         return (*png_create_info_structFP.Call) (png_ptr);
      else
         return NULL;
   }
   // If possible call the PNG png_create_info_struct function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error code is based on PNG
   // version 1.2.50).

   png_structp PNGif::create_read_struct (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn)
   {
      if (png_create_read_structFP.Call != NULL)
         return (*png_create_read_structFP.Call) (user_png_ver, error_ptr, error_fn, warn_fn);
      else
         return NULL;
   }
   // If possible call the PNG png_create_read_struct function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error return is based on PNG
   // version 1.2.50).

   png_structp PNGif::create_write_struct (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn)
   {
      if (png_create_write_structFP.Call != NULL)
         return (*png_create_write_structFP.Call) (user_png_ver, error_ptr, error_fn, warn_fn);
      else
         return NULL;
   }
   // If possible call the PNG png_create_write_struct function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error return is based on PNG
   // version 1.2.50).

   void PNGif::destroy_read_struct (png_structpp png_ptr_ptr, png_infopp info_ptr_ptr, png_infopp end_info_ptr_ptr)
   {
      if (png_destroy_read_structFP.Call != NULL)
         (*png_destroy_read_structFP.Call) (png_ptr_ptr, info_ptr_ptr, end_info_ptr_ptr);
   }
   // If possible call the PNG png_destroy_read_struct function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::destroy_write_struct (png_structpp png_ptr_ptr, png_infopp info_ptr_ptr)
   {
      if (png_destroy_write_structFP.Call != NULL)
         (*png_destroy_write_structFP.Call) (png_ptr_ptr, info_ptr_ptr);
   }
   // If possible call the PNG png_destroy_write_struct function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::error (png_structp png_ptr, png_const_charp error)
   {
      if (png_errorFP.Call != NULL)
         (*png_errorFP.Call) (png_ptr, error);
      else
      {  // PNG error mechanism not accessible, try any user-defined
         // error handler that may be set up..
         if (png_ptr != NULL && png_ptr->error_fn != NULL)
            (*(png_ptr->error_fn))(png_ptr, error);
         // Control passes to here if there is no user-defined handler
         // or that handler has (mistakingly) returned.
         // If png_ptr is valid, longjmp back to the point the user
         // should have initialised into the structure.
         if (png_ptr)
            longjmp(png_jmpbuf(png_ptr), 1);
         // We have no user error trapping options, so our only way
         // forward is to silently error
      }
   }
   // If possible call the PNG png_error function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) attempt to call any user
   // defined error handling that is specified in the png_ptr structure.

   png_uint_32 PNGif::get_IHDR (png_structp png_ptr, png_infop info_ptr, png_uint_32 *width,
                                            png_uint_32 *height, int *bit_depth, int *color_type,
                                            int *interlace_type, int *compression_type, int *filter_type)
   {
      if (png_get_IHDRFP.Call != NULL)
         return (*png_get_IHDRFP.Call) (png_ptr, info_ptr, width, height, 
                                                                     bit_depth, color_type, interlace_type,
                                                                     compression_type, filter_type);
      else
         return 0;
   }
   // If possible call the PNG png_get_IHDR function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error code is based on PNG
   // version 1.2.50).

   png_voidp PNGif::get_io_ptr (png_structp png_ptr)
   {
      if (png_get_io_ptrFP.Call != NULL)
         return (*png_get_io_ptrFP.Call) (png_ptr);
      else
         return NULL;
   }
   // If possible call the PNG png_get_io_ptr function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error code is based on PNG
   // version 1.2.50).

   png_uint_32 PNGif::get_sBIT (png_structp png_ptr, png_infop info_ptr, png_color_8p *sig_bit)
   {
      if (png_get_sBITFP.Call != NULL)
         return (*png_get_sBITFP.Call) (png_ptr, info_ptr, sig_bit);
      else
         return 0;
   }
   // If possible call the PNG png_get_sBIT function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error code is based on PNG
   // version 1.2.50).

   png_uint_32 PNGif::get_valid   (png_structp png_ptr, png_infop info_ptr, png_uint_32 flag)
   {
      if (png_get_validFP.Call != NULL)
         return (*png_get_validFP.Call) (png_ptr, info_ptr, flag);
      else
         return 0;
   }
   // If possible call PNG png_get_valid function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error code is based on PNG
   // version 1.2.50).

   void PNGif::read_end (png_structp png_ptr, png_infop info_ptr)
   {
      if (png_read_endFP.Call != NULL)
         (*png_read_endFP.Call) (png_ptr, info_ptr);
      else
         error (png_ptr, "Unable to call libpng function png_read_end");
   }
   // If possible call the PNG png_read_end function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::read_info (png_structp png_ptr, png_infop info_ptr)
   {
      if (png_read_infoFP.Call != NULL)
         (*png_read_infoFP.Call) (png_ptr, info_ptr);
      else
         error (png_ptr, "Unable to call libpng function png_read_info");
   }
   // If possible call the PNG png_read_info function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::read_rows (png_structp png_ptr, png_bytepp row, png_bytepp display_row, png_uint_32 num_rows)
   {
      if (png_read_rowsFP.Call != NULL)
         (*png_read_rowsFP.Call) (png_ptr, row, display_row, num_rows);
      else
         error (png_ptr, "Unable to call libpng function png_read_rows");
   }
   // If possible call the PNG png_read_rows function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::read_update_info (png_structp png_ptr, png_infop info_ptr)
   {
      if (png_read_update_infoFP.Call != NULL)
         (*png_read_update_infoFP.Call) (png_ptr, info_ptr);
      else
         error (png_ptr, "Unable to call libpng function png_read_update_info");
   }
   // If possible call the PNG png_read_update_info function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::set_expand (png_structp png_ptr)
   {
      if (png_set_expandFP.Call != NULL)
         (*png_set_expandFP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_expand function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_filler (png_structp png_ptr, png_uint_32 filler, int flags)
   {
      if (png_set_fillerFP.Call != NULL)
         (*png_set_fillerFP.Call) (png_ptr, filler, flags);
   }
   // If possible call the PNG png_set_filler function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_gray_to_rgb (png_structp png_ptr)
   {
      if (png_set_gray_to_rgbFP.Call != NULL)
         (*png_set_gray_to_rgbFP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_gray_to_rgb function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   int PNGif::set_interlace_handling (png_structp png_ptr)
   {
      if (png_set_interlace_handlingFP.Call != NULL)
         return (*png_set_interlace_handlingFP.Call) (png_ptr);
      else
         return 1; /* report as non-interlaced */
   }
   // If possible call the PNG png_set_interlace_handling function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) return an error typical of a
   // failure within the PNG function itself (error return is based on PNG
   // version 1.2.50).

   void PNGif::set_IHDR (png_structp png_ptr, png_infop info_ptr, png_uint_32 width,
                                     png_uint_32 height, int bit_depth, int color_type,
                                     int interlace_type, int compression_type, int filter_type)
   {
      if (png_set_IHDRFP.Call != NULL)
         (*png_set_IHDRFP.Call) (png_ptr, info_ptr, width, height,
                                                              bit_depth, color_type, interlace_type,
                                                              compression_type, filter_type);
   }
   // If possible call the PNG png_set_IHDR function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_packing (png_structp png_ptr)
   {
      if (png_set_packingFP.Call != NULL)
         (*png_set_packingFP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_packing function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_read_fn (png_structp png_ptr, png_voidp io_ptr, png_rw_ptr read_data_fn)
   {
      if (png_set_read_fnFP.Call != NULL)
         (*png_set_read_fnFP.Call) (png_ptr, io_ptr, read_data_fn);
   }
   // If possible call the PNG png_set_read_fn function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_rgb_to_gray_fixed (png_structp png_ptr, int error_action, png_fixed_point red, png_fixed_point green)
   {
      if (png_set_rgb_to_gray_fixedFP.Call != NULL)
         (*png_set_rgb_to_gray_fixedFP.Call) (png_ptr, error_action, red, green);
   }
   // If possible call the PNG png_set_rgb_to_gray_fixed function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_shift (png_structp png_ptr, png_color_8p true_bits)
   {
      if (png_set_shiftFP.Call != NULL)
         (*png_set_shiftFP.Call) (png_ptr, true_bits);
   }
   // If possible call the PNG png_set_shift function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_strip_16 (png_structp png_ptr)
   {
      if (png_set_strip_16FP.Call != NULL)
         (*png_set_strip_16FP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_strip_16 function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_strip_alpha (png_structp png_ptr)
   {
      if (png_set_strip_alphaFP.Call != NULL)
         (*png_set_strip_alphaFP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_strip_alpha function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_swap (png_structp png_ptr)
   {
      if (png_set_swapFP.Call != NULL)
         (*png_set_swapFP.Call) (png_ptr);
   }
   // If possible call the PNG png_set_swap function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::set_text (png_structp png_ptr, png_infop info_ptr, png_textp text_ptr, int num_text)
   {
      if (png_set_textFP.Call != NULL)
         (*png_set_textFP.Call) (png_ptr, info_ptr, text_ptr, num_text);
      else
         error (png_ptr, "Unable to call libpng function png_set_text");
   }
   // If possible call the PNG png_set_text function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::set_write_fn (png_structp png_ptr, png_voidp io_ptr, png_rw_ptr write_data_fn, png_flush_ptr output_flush_fn)
   {
      if (png_set_write_fnFP.Call != NULL)
         (*png_set_write_fnFP.Call) (png_ptr, io_ptr, write_data_fn, output_flush_fn);
   }
   // If possible call the PNG png_set_write_fn function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) silently fail as per the
   // PNG function if png_ptr_ptr is NULL (behaviour based on PNG version
   // 1.2.50).

   void PNGif::write_end (png_structp png_ptr, png_infop info_ptr)
   {
      if (png_write_endFP.Call != NULL)
         (*png_write_endFP.Call) (png_ptr, info_ptr);
      else
         error (png_ptr, "Unable to call libpng function write_end");
   }
   // If possible call the PNG png_write_end function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::write_info (png_structp png_ptr, png_infop info_ptr)
   {
      if (png_write_infoFP.Call != NULL)
         (*png_write_infoFP.Call) (png_ptr, info_ptr);
      else
         error (png_ptr, "Unable to call libpng function write_info");
   }
   // If possible call the PNG png_write_info function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   void PNGif::write_rows (png_structp png_ptr, png_bytepp row, png_uint_32 num_rows)
   {
      if (png_write_rowsFP.Call != NULL)
         (*png_write_rowsFP.Call) (png_ptr, row, num_rows);
      else
         error (png_ptr, "Unable to call libpng function write_rows");
   }
   // If possible call the PNG png_write_rows function.
   // If it is not possible to make the function call (for example if the 
   // PNG library did not dynamically load) call the error handler to try
   // and use any user-defined error handling declared in png_ptr. Using
   // the error handler as original function makes use of it (based on
   // PNG version 1.2.50) so any calling code should have set up either
   // a custom catcher routine or the longjmp.

   PNGif Png;

} // End of RavlImageN
