// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////


#define RAVLIMAGE_JPEGIF_CCFILE  1
// Prevents extern definition of Jpeg, etc. in JPEGif.hh

#include <dlfcn.h>
#include <stdio.h>
#include <jpeglib.h>
#include <jerror.h>

#include "Ravl/DP/DynamicLink.hh"
#include "Ravl/TypeName.hh"
#include "Ravl/DP/Converter.hh"
#include "Ravl/DP/FileFormatStream.hh"
#include "Ravl/DP/FileFormatBinStream.hh"
#include "Ravl/Image/JPEGif.hh"


namespace RavlImageN {
   
   JPEGif Jpeg;
   // Global class instance used to access the jpeg library

   const char * const JPEGif_stub_message_table[] = { "Bogus message code %d",
                                                "Failed to load JPEG library"
                                              };
   // Fallback error messages should jpeg library be unavilable


   // Global helper routines follow. These are needed outside of the interface
   // class to make them available for use with function pointers as used with
   // the standard jpeg interfacing methods.


   boolean JPEGif_call_resync_to_restart (j_decompress_ptr cinfo, int desired)
   {
      return Jpeg.resync_to_restart (cinfo, desired);
   }
   // Helper routine to allow provision of the function pointer to the resync
   // function that DPImageIOJPegIBaseC requires.

   static void JPEGif_stub_emit_message (j_common_ptr cinfo, int msg_level)
   {
      struct jpeg_error_mgr * err = cinfo->err;

      if (msg_level < 0) 
      {  /* Warning - show only first warning unless trace level at least 3 */
         if (err->num_warnings++ == 0 || err->trace_level >= 3)
            (*err->output_message) (cinfo);
      }
      else
      {  /* Trace - show it if trace_level >= msg_level. */
        if (err->trace_level >= msg_level) (*err->output_message) (cinfo);
      }
   }
   // Stub emit_message function provided in case the jpeg_error_mgr function
   // is unavailable. Without jpeg_error_mgr, we cannot access the jpeg library
   // version of emit_message and need to provide an alternative in case the
   // user attempts to use the relevant function pointer in the jpeg structure.

   static void JPEGif_stub_error_exit (j_common_ptr cinfo)
   {
      (*cinfo->err->output_message) (cinfo);

      Jpeg.destroy (cinfo);
      // If we have access to it, call jpeg_destroy to clean up any files the
      // jpeg library memory manager may have left.

      RavlIssueError("Fatal JPEG error\n\r");
   }
   // Stub error_exit function provided in case the jpeg_error_mgr function
   // is unavailable. Without jpeg_error_mgr, we cannot access the jpeg library
   // version of error_exit and need to provide an alternative in case the
   // user attempts to use the relevant function pointer in the jpeg structure.

   static void JPEGif_stub_format_message (j_common_ptr cinfo, char * buffer)
   {
      struct jpeg_error_mgr * err = cinfo->err;
      int error = err->msg_code;
      const char * msg = NULL;

      /* Look up message string in proper table */
      if (error <= err->last_jpeg_message)
      {  if (error > 0 )
            msg = err->jpeg_message_table[error];
      }
      else
      {  if (err->addon_message_table != NULL
	      && error >= err->first_addon_message
	          && error <= err->last_addon_message)
            msg = err->addon_message_table[error - err->first_addon_message];
      }

      if (msg == NULL)
      {   err->msg_parm.i[0] = error;
          msg = err->jpeg_message_table[0];
      }

      if (strstr(msg,"%s"))
         sprintf(buffer, msg, err->msg_parm.s);
      else
         sprintf(buffer, msg, err->msg_parm.i[0], err->msg_parm.i[1],
                  err->msg_parm.i[2], err->msg_parm.i[3], err->msg_parm.i[4],
                    err->msg_parm.i[5], err->msg_parm.i[6], err->msg_parm.i[7]);
   }
   // Stub format_message function provided in case the jpeg_error_mgr function
   // is unavailable. Without jpeg_error_mgr, we cannot access the jpeg library
   // version of format_message and need to provide an alternative in case the
   // user attempts to use the relevant function pointers in the jpeg structure.

   static void JPEGif_stub_output_message (j_common_ptr cinfo)
   {
      char buffer[JMSG_LENGTH_MAX];

      (*cinfo->err->format_message) (cinfo, buffer);
      // Most likely to call JPEGif_stub_format_message unless the user has defined 
      // their own handler (which would be called via the pointer anyway).

      RavlIssueWarning(buffer);
   }
   // Stub output_message function provided in case the jpeg_error_mgr function
   // is unavailable. Without jpeg_error_mgr, we cannot access the jpeg library
   // version of output_message and need to provide an alternative in case the
   // user attempts to use the relevant function pointers in the jpeg structure.

   static void JPEGif_stub_reset_error_mgr (j_common_ptr cinfo)
   {
      cinfo->err->msg_code = 0;
      cinfo->err->num_warnings = 0;
   }
   // Stub reset_error_mgr function provided in case the jpeg_error_mgr function
   // is unavailable. Without jpeg_error_mgr, we cannot access the jpeg library
   // version of reset_error_mgr and need to provide an alternative in case the
   // user attempts to use the relevant function pointer in the jpeg structure.


   // end of stub functions, calls member definitions follow...


   JPEGif::JPEGif ()
   {
      void * handle;

      if ( (handle = DynamicLinkLoad ("libjpeg.so", true)) )
      {
         jpeg_CreateCompressFP.AsVoidPtr = dlsym (handle, "jpeg_CreateCompress");
         jpeg_create_compressFP.AsVoidPtr = dlsym (handle, "jpeg_create_compress");
         jpeg_CreateDecompressFP.AsVoidPtr = dlsym (handle, "jpeg_CreateDecompress");
         jpeg_create_decompressFP.AsVoidPtr = dlsym (handle, "jpeg_create_decompress");
         jpeg_destroyFP.AsVoidPtr = dlsym (handle, "jpeg_destroy");
         jpeg_destroy_compressFP.AsVoidPtr = dlsym (handle, "jpeg_destroy_compress");
         jpeg_destroy_decompressFP.AsVoidPtr = dlsym (handle, "jpeg_destroy_decompress");
         jpeg_finish_compressFP.AsVoidPtr = dlsym (handle, "jpeg_finish_compress");
         jpeg_finish_decompressFP.AsVoidPtr = dlsym (handle, "jpeg_finish_decompress");
         jpeg_read_headerFP.AsVoidPtr = dlsym (handle, "jpeg_read_header");
         jpeg_read_scanlinesFP.AsVoidPtr = dlsym (handle, "jpeg_read_scanlines");
         jpeg_resync_to_restartFP.AsVoidPtr = dlsym (handle, "jpeg_resync_to_restart");
         jpeg_set_defaultsFP.AsVoidPtr = dlsym (handle, "jpeg_set_defaults");
         jpeg_set_qualityFP.AsVoidPtr = dlsym (handle, "jpeg_set_quality");
         jpeg_start_compressFP.AsVoidPtr = dlsym (handle, "jpeg_start_compress");
         jpeg_start_decompressFP.AsVoidPtr = dlsym (handle, "jpeg_start_decompress");
         jpeg_std_errorFP.AsVoidPtr = dlsym (handle, "jpeg_std_error");
         jpeg_write_scanlinesFP.AsVoidPtr = dlsym (handle, "jpeg_write_scanlines");

         jpeg_std_message_tableDP  = (const char * const *) dlsym (handle, "jpeg_std_message_table");


         type1  = new TypeNameC (typeid(CompressedImageJPEGC), "RavlImageN::CompressedImageJPEGC");
  

         DPConv_RGBImage2CompressedImageJPEG = new RavlN::DPConverterBaseC(RavlN::RegisterConversion(RGBImage2CompressedImageJPEG ,0.9,
                                                                                                     "CompressedImageJPEGC RavlImageN::Convert(const ImageC<ByteRGBValueC> &)"
                                                                                                    )
                                                                          );

         DPConv_CompressedImageJPEG2RGBImage = new RavlN::DPConverterBaseC(RavlN::RegisterConversion(CompressedImageJPEG2RGBImage, 1,
                                                                                                     "ImageC<ByteRGBValueC> RavlImageN::Convert(const CompressedImageJPEGC &)"
                                                                                                    )
                                                                          );


         FileFormatStream_CompressedImageJPEGC = new FileFormatStreamC<CompressedImageJPEGC> ();

         FileFormatBinStream_CompressedImageJPEGC  = new FileFormatBinStreamC<CompressedImageJPEGC> ();

  
         RegisterFileFormatJPEG = new FileFormatJPEGC (100,0,false,"jpeg","JPEG image file format. 'Lossless' compression.");
  
         RegisterFileFormatJPEGL = new FileFormatJPEGC (75,-25,false,"jpegl","JPEG image file format. 75% lossy compression.");
  
         RegisterFileFormatMJPEG = new FileFormatJPEGC (100,-1,true,"mjpeg","Sequence of JPEG images in a single file. 'lossless' compression.");
  
         RegisterFileFormatMJPEGL = new FileFormatJPEGC (75,-26,true,"mjpegl","Sequence of JPEG images in a single file.  75% lossy compression.");

      }
      else
      {
         jpeg_CreateCompressFP.AsVoidPtr = NULL;
         jpeg_create_compressFP.AsVoidPtr = NULL;
         jpeg_CreateDecompressFP.AsVoidPtr = NULL;
         jpeg_create_decompressFP.AsVoidPtr = NULL;
         jpeg_destroyFP.AsVoidPtr = NULL;
         jpeg_destroy_compressFP.AsVoidPtr = NULL;
         jpeg_destroy_decompressFP.AsVoidPtr = NULL;
         jpeg_finish_compressFP.AsVoidPtr = NULL;
         jpeg_finish_decompressFP.AsVoidPtr = NULL;
         jpeg_read_headerFP.AsVoidPtr = NULL;
         jpeg_read_scanlinesFP.AsVoidPtr = NULL;
         jpeg_resync_to_restartFP.AsVoidPtr = NULL;
         jpeg_set_defaultsFP.AsVoidPtr = NULL;
         jpeg_set_qualityFP.AsVoidPtr = NULL;
         jpeg_start_compressFP.AsVoidPtr = NULL;
         jpeg_start_decompressFP.AsVoidPtr = NULL;
         jpeg_std_errorFP.AsVoidPtr = NULL;
         jpeg_write_scanlinesFP.AsVoidPtr = NULL;

         jpeg_std_message_tableDP = NULL;
      }
   }
   // Default constructor

   JPEGif::~JPEGif ()
   { }
   // Standard destructor function

   void JPEGif::create_compress (j_compress_ptr cinfo)
   {
      if (jpeg_CreateCompressFP.Call != NULL)
         (*jpeg_CreateCompressFP.Call) (cinfo, JPEG_LIB_VERSION, (size_t) sizeof(struct jpeg_compress_struct));
      else
         if (jpeg_create_compressFP.Call != NULL)
            (*jpeg_create_compressFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_create_compress function or, preferably,
   // the jpeg_CreateCompress function. Lib JPEG post version 6 replaces any
   // jpeg_create_compress call via a hash-define to jpeg_CreateCompress; we
   // therefore need to cater for this here and use the later function if it
   // exists..
   // If it is not possible to make either function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::create_decompress (j_decompress_ptr cinfo)
   {
      if (jpeg_CreateDecompressFP.Call != NULL)
         (*jpeg_CreateDecompressFP.Call) (cinfo, JPEG_LIB_VERSION, (size_t) sizeof(struct jpeg_decompress_struct));
      else
         if (jpeg_create_decompressFP.Call != NULL)
            (*jpeg_create_decompressFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_create_decompress function or, preferably,
   // the jpeg_CreateDecompress function. Lib JPEG post version 6 replaces any
   // jpeg_create_decompress call via a hash-define to jpeg_CreateDecompress; we
   // therefore need to cater for this here and use the later function if it
   // exists..
   // If it is not possible to make either function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::destroy (j_common_ptr cinfo)
   {
      if (jpeg_destroyFP.Call != NULL)
         (*jpeg_destroyFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_destroy function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::destroy_compress (j_compress_ptr cinfo)
   {
      if (jpeg_destroy_compressFP.Call != NULL)
         (*jpeg_destroy_compressFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_destroy_compress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::destroy_decompress (j_decompress_ptr cinfo)
   {
      if (jpeg_destroy_decompressFP.Call != NULL)
         (*jpeg_destroy_decompressFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_destroy_decompress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::finish_compress (j_compress_ptr cinfo)
   {
      if (jpeg_finish_compressFP.Call != NULL)
         (*jpeg_finish_compressFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_finish_compress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   boolean JPEGif::finish_decompress (j_decompress_ptr cinfo)
   {
      if (jpeg_finish_decompressFP.Call != NULL)
         return (*jpeg_finish_decompressFP.Call) (cinfo);
      else
         return false;
   }
   // If possible call the JPEG jpeg_finish_decompress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   int JPEGif::read_header (j_decompress_ptr cinfo, boolean require_image)
   {
      if (jpeg_read_headerFP.Call != NULL)
         return (*jpeg_read_headerFP.Call) (cinfo, require_image);
      else
         return JPEG_SUSPENDED;
   }
   // If possible call the JPEG jpeg_read_header function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   JDIMENSION JPEGif::read_scanlines (j_decompress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION max_lines)
   {
      if (jpeg_read_scanlinesFP.Call != NULL)
         return (*jpeg_read_scanlinesFP.Call) (cinfo, scanlines, max_lines);
      else
         return 0;
   }
   // If possible call the JPEG jpeg_read_scanlines function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   boolean JPEGif::resync_to_restart (j_decompress_ptr cinfo, int desired)
   {
      if (jpeg_resync_to_restartFP.Call != NULL)
         return (*jpeg_resync_to_restartFP.Call) (cinfo, desired);
      else
         return false;
   }
   // If possible call the JPEG jpeg_resync_to_restart function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::set_defaults (j_compress_ptr cinfo)
   {
      if (jpeg_set_defaultsFP.Call != NULL)
         (*jpeg_set_defaultsFP.Call) (cinfo);
   }
   // If possible call the JPEG jpeg_set_defaults function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::set_quality (j_compress_ptr cinfo, int quality, boolean force_baseline)
   {
      if (jpeg_set_qualityFP.Call != NULL)
         (*jpeg_set_qualityFP.Call) (cinfo, quality, force_baseline);
   }
   // If possible call the JPEG jpeg_set_quality function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   void JPEGif::start_compress (j_compress_ptr cinfo, boolean write_all_tables)
   {
      if (jpeg_start_compressFP.Call != NULL)
         (*jpeg_start_compressFP.Call) (cinfo, write_all_tables);
   }
   // If possible call the JPEG jpeg_start_compress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   boolean JPEGif::start_decompress (j_decompress_ptr cinfo)
   {
      if (jpeg_start_decompressFP.Call != NULL)
         return (*jpeg_start_decompressFP.Call) (cinfo);
      else
         return false;
   }
   // If possible call the JPEG jpeg_start_decompress function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

   struct jpeg_error_mgr * JPEGif::std_error (struct jpeg_error_mgr * err)
   {
      if (jpeg_std_errorFP.Call != NULL)
         return (*jpeg_std_errorFP.Call) (err);
      else
      {
          // This code is needed if the user makes direct use of these pointers
          // in their own code and because this interface layer uses them if
          // other jpeg library calls are unavailable (the likely scenario if
          // we have found the std_error function unavailable).

          err->emit_message = (void (*)(jpeg_common_struct*, int)) &JPEGif_stub_emit_message;
          err->error_exit = (void (*)(jpeg_common_struct*)) &JPEGif_stub_error_exit;
          err->format_message = (void (*)(jpeg_common_struct*, char *)) &JPEGif_stub_format_message;
          err->output_message = (void (*)(jpeg_common_struct*)) &JPEGif_stub_output_message;
          err->reset_error_mgr = (void (*)(jpeg_common_struct*)) &JPEGif_stub_reset_error_mgr;
          // Effectivly, the above code does the same initialisation that the
          // libjpeg function would have done (based on version 62 / 6b of the
          // library) but points to local stub handlers instead of the (static
          // to the library) jpeg error routines.

          // Set up the pointer to the jpeg error messages if we can, otherwise
          // use our own stub table
          if (jpeg_std_message_tableDP != NULL) 
          {  err->jpeg_message_table = jpeg_std_message_tableDP;
             err->last_jpeg_message = (int) JMSG_LASTMSGCODE - 1;
          }
          else
          {  err->jpeg_message_table = JPEGif_stub_message_table;
             err->last_jpeg_message = (int) 0;
          }

          // The remaining code emulates the rest of the initialisation made by
          // the (revision 62 / 6b) jpeg library version of this function.
          err->addon_message_table = NULL;
          err->first_addon_message = 0;
          err->last_addon_message = 0;
          err->msg_code = 0;
          err->num_warnings = 0;
          err->trace_level = 0;

          return err;
      }
   }
   // If possible call the JPEG jpeg_std_error function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) initialise the error structure
   // as the JPEG function itself (based on JPEG version 62 / 6b) would and 
   // return successfully

   JDIMENSION JPEGif::write_scanlines (j_compress_ptr cinfo, JSAMPARRAY scanlines, JDIMENSION num_lines)
   {
      if (jpeg_write_scanlinesFP.Call != NULL)
         return (*jpeg_write_scanlinesFP.Call) (cinfo, scanlines, num_lines);
      else
         return 0;
   }
   // If possible call the JPEG jpeg_write_scanlines function
   // If it is not possible to make the function call (for example if the 
   // JPEG library did not dynamically load) return an error typical of a
   // failure within the JPEG function itself (error code is based on JPEG
   // version 62 / 6b).

} // End of RavlImageN
