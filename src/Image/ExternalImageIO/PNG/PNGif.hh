// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//

#ifndef RAVLIMAGE_PNGIF_HEADER
#  define RAVLIMAGE_PNGIF_HEADER	1

#  include <png.h>

#  include "Ravl/Image/PNGFormat.hh"


   namespace RavlImageN {

	class PNGif {
		public:
			PNGif ();
			~PNGif ();

			/* This structure does not atttempt to provide the complete libPNG
			   interface (feel free to extend it..) but the following functions do
			   implement those parts that are used or mentioned by the Ravl code.
			*/
			png_infop create_info_struct (png_structp png_ptr);
			png_structp create_read_struct (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn);
			png_structp create_write_struct (png_const_charp user_png_ver, png_voidp error_ptr, png_error_ptr error_fn, png_error_ptr warn_fn);
			void destroy_read_struct (png_structpp png_ptr_ptr, png_infopp info_ptr_ptr, png_infopp end_info_ptr_ptr);
			void destroy_write_struct (png_structpp png_ptr_ptr, png_infopp info_ptr_ptr);
			void error (png_structp png_ptr, png_const_charp error);
			png_uint_32 get_IHDR (png_structp png_ptr, png_infop info_ptr, png_uint_32 *width,
                                              png_uint_32 *height, int *bit_depth, int *color_type,
                                              int *interlace_type, int *compression_type, int *filter_type);
			png_voidp get_io_ptr (png_structp png_ptr);
			png_uint_32 get_sBIT (png_structp png_ptr, png_infop info_ptr, png_color_8p *sig_bit);
			png_uint_32 get_valid   (png_structp png_ptr, png_infop info_ptr, png_uint_32 flag);
			void read_end (png_structp png_ptr, png_infop info_ptr);
			void read_info (png_structp png_ptr, png_infop info_ptr);
			void read_rows (png_structp png_ptr, png_bytepp row, png_bytepp display_row, png_uint_32 num_rows);
			void read_update_info (png_structp png_ptr, png_infop info_ptr);
			void set_expand (png_structp png_ptr);
			void set_filler (png_structp png_ptr, png_uint_32 filler, int flags);
			void set_gray_to_rgb (png_structp png_ptr);
			int set_interlace_handling (png_structp png_ptr);
			void set_IHDR (png_structp png_ptr, png_infop info_ptr, png_uint_32 width,
                                       png_uint_32 height, int bit_depth, int color_type,
                                       int interlace_type, int compression_type, int filter_type);
			void set_packing (png_structp png_ptr);
			void set_read_fn (png_structp png_ptr, png_voidp io_ptr, png_rw_ptr read_data_fn);
			void set_rgb_to_gray_fixed (png_structp png_ptr, int error_action, png_fixed_point red, png_fixed_point green);
			void set_shift (png_structp png_ptr, png_color_8p true_bits);
			void set_strip_16 (png_structp png_ptr);
			void set_strip_alpha (png_structp png_ptr);
			void set_swap (png_structp png_ptr);
			void set_text (png_structp png_ptr, png_infop info_ptr, png_textp text_ptr, int num_text);
			void set_write_fn (png_structp png_ptr, png_voidp io_ptr, png_rw_ptr write_data_fn, png_flush_ptr output_flush_fn);
			void write_end (png_structp png_ptr, png_infop info_ptr);
			void write_info (png_structp png_ptr, png_infop info_ptr);
			void write_rows (png_structp png_ptr, png_bytepp row, png_uint_32 num_rows);

		protected:
			union { void * AsVoidPtr; png_infop (* Call) (png_structp); } png_create_info_structFP;
			union { void * AsVoidPtr; png_structp (* Call) (png_const_charp, png_voidp, png_error_ptr, png_error_ptr); } png_create_read_structFP;
			union { void * AsVoidPtr; png_structp (* Call) (png_const_charp, png_voidp, png_error_ptr, png_error_ptr); } png_create_write_structFP;
			union { void * AsVoidPtr; void (* Call) (png_structpp, png_infopp, png_infopp); } png_destroy_read_structFP;
			union { void * AsVoidPtr; void (* Call) (png_structpp, png_infopp); } png_destroy_write_structFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_const_charp); } png_errorFP;
			union { void * AsVoidPtr; png_uint_32 (* Call) (png_structp, png_infop, png_uint_32 *, png_uint_32 *, int *, int *, int *, int *, int *); } png_get_IHDRFP;
			union { void * AsVoidPtr; png_voidp (* Call) (  png_structp); } png_get_io_ptrFP;
			union { void * AsVoidPtr; png_uint_32 (* Call) (png_structp, png_infop, png_color_8p *); } png_get_sBITFP;
			union { void * AsVoidPtr; png_uint_32 (* Call) (png_structp, png_infop, png_uint_32); } png_get_validFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop); } png_read_endFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop); } png_read_infoFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_bytepp, png_bytepp, png_uint_32); } png_read_rowsFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop); } png_read_update_infoFP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_expandFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_uint_32, int); } png_set_fillerFP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_gray_to_rgbFP;
			union { void * AsVoidPtr; int (* Call) (png_structp); } png_set_interlace_handlingFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop, png_uint_32, png_uint_32, int, int, int, int, int); } png_set_IHDRFP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_packingFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_voidp, png_rw_ptr); } png_set_read_fnFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, int, png_fixed_point, png_fixed_point); } png_set_rgb_to_gray_fixedFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_color_8p); } png_set_shiftFP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_strip_16FP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_strip_alphaFP;
			union { void * AsVoidPtr; void (* Call) (png_structp); } png_set_swapFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop, png_textp, int); } png_set_textFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_voidp, png_rw_ptr, png_flush_ptr); } png_set_write_fnFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop); } png_write_endFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_infop); } png_write_infoFP;
			union { void * AsVoidPtr; void (* Call) (png_structp, png_bytepp, png_uint_32); } png_write_rowsFP;

			FileFormatPNGC * RegisterFileFormatPNG;
			FileFormatPNGC * RegisterFileFormatPNG16;

	}; // End of PNGif

#  ifndef RAVLIMAGE_PNGIF_CCFILE
	extern PNGif Png;
#  endif

   } // End of RavlImageN

#endif // End of duplicate inclusion protection
