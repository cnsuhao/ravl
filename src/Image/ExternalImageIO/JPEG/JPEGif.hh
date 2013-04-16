// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//////////////////////////////////////////////////

#ifndef RAVLIMAGE_JPEGIF_HEADER
#  define RAVLIMAGE_JPEGIF_HEADER        1

#  include <stdio.h>
#  include <string.h>
#  include <dlfcn.h>
#  include <jpeglib.h>

#  include "Ravl/TypeName.hh"
#  include "Ravl/DP/Converter.hh"
#  include "Ravl/DP/FileFormatStream.hh"
#  include "Ravl/DP/FileFormatBinStream.hh"
#  include "Ravl/Image/JPEGFormat.hh"
#  include "Ravl/Image/CompressedImageJPEG.hh"


   namespace RavlImageN {

      class JPEGif {
         public:
            JPEGif ();
            // Default constructor

            ~JPEGif ();
            // Standard destructor function

            void create_compress (j_compress_ptr);
            // If possible call the JPEG jpeg_create_compress function or,
            // preferably, the jpeg_CreateCompress function. Lib JPEG post
            // version 6 replaces any jpeg_create_compress call (via a
            // hash-define) with jpeg_CreateCompress; this function therefore
            // caters for this and calls the later function in preference if it
            // exists.

            void create_decompress (j_decompress_ptr);
            // If possible call the JPEG jpeg_create_decompress function or,
            // preferably, the jpeg_CreateDecompress function. Lib JPEG post
            // version 6 replaces any jpeg_create_decompress call (via a
            // hash-define) with jpeg_CreateDecompress; this function therefore
            // caters for this and calls the later function in preference if it
            // exists.

            void destroy (j_common_ptr);
            // If possible call the JPEG jpeg_destroy function

            void destroy_compress (j_compress_ptr);
            // If possible call the JPEG jpeg_destroy_compress function

            void destroy_decompress (j_decompress_ptr);
            // If possible call the JPEG jpeg_destroy_decompress function

            void finish_compress (j_compress_ptr);
            // If possible call the JPEG jpeg_finish_compress function

            boolean finish_decompress (j_decompress_ptr);
            // If possible call the JPEG jpeg_finish_decompress function

            int read_header (j_decompress_ptr, boolean);
            // If possible call the JPEG jpeg_read_header function

            JDIMENSION read_scanlines (j_decompress_ptr, JSAMPARRAY, JDIMENSION);
            // If possible call the JPEG jpeg_read_scanlines function

            boolean resync_to_restart (j_decompress_ptr, int);
            // If possible call the JPEG jpeg_resync_to_restart function

            void set_defaults (j_compress_ptr);
            // If possible call the JPEG jpeg_set_defaults function

            void set_quality (j_compress_ptr, int, boolean);
            // If possible call the JPEG jpeg_set_quality function

            void start_compress (j_compress_ptr, boolean);
            // If possible call the JPEG jpeg_start_compress function

            boolean start_decompress (j_decompress_ptr);
            // If possible call the JPEG jpeg_start_decompress function

            struct jpeg_error_mgr * std_error (struct jpeg_error_mgr *);
            // If possible call the JPEG jpeg_std_error function

            JDIMENSION write_scanlines (j_compress_ptr, JSAMPARRAY, JDIMENSION);
            // If possible call the JPEG jpeg_write_scanlines function



         protected:

            // Exported JPEG functions 
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr, int, size_t); } jpeg_CreateCompressFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr); } jpeg_create_compressFP;
            union { void * AsVoidPtr; void (* Call) (j_decompress_ptr, int, size_t); } jpeg_CreateDecompressFP;
            union { void * AsVoidPtr; void (* Call) (j_decompress_ptr); } jpeg_create_decompressFP;
            union { void * AsVoidPtr; void (* Call) (j_common_ptr); } jpeg_destroyFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr); } jpeg_destroy_compressFP;
            union { void * AsVoidPtr; void (* Call) (j_decompress_ptr); } jpeg_destroy_decompressFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr); } jpeg_finish_compressFP;
            union { void * AsVoidPtr; boolean (* Call) (j_decompress_ptr); } jpeg_finish_decompressFP;
            union { void * AsVoidPtr; int (* Call) (j_decompress_ptr, boolean); } jpeg_read_headerFP;
            union { void * AsVoidPtr; JDIMENSION (* Call) (j_decompress_ptr, JSAMPARRAY, JDIMENSION); } jpeg_read_scanlinesFP;
            union { void * AsVoidPtr; boolean (* Call) (j_decompress_ptr, int); } jpeg_resync_to_restartFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr); } jpeg_set_defaultsFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr, int, boolean); } jpeg_set_qualityFP;
            union { void * AsVoidPtr; void (* Call) (j_compress_ptr cinfo, boolean); } jpeg_start_compressFP;
            union { void * AsVoidPtr; boolean (* Call) (j_decompress_ptr); } jpeg_start_decompressFP;
            union { void * AsVoidPtr; struct jpeg_error_mgr * (* Call) (struct jpeg_error_mgr *); } jpeg_std_errorFP;
            union { void * AsVoidPtr; JDIMENSION (* Call) (j_compress_ptr, JSAMPARRAY, JDIMENSION); } jpeg_write_scanlinesFP;

            const char * const * jpeg_std_message_tableDP;

            TypeNameC * type1;

            RavlN::DPConverterBaseC * DPConv_RGBImage2CompressedImageJPEG;
            RavlN::DPConverterBaseC * DPConv_CompressedImageJPEG2RGBImage;

            FileFormatStreamC<CompressedImageJPEGC> * FileFormatStream_CompressedImageJPEGC;
            FileFormatBinStreamC<CompressedImageJPEGC> * FileFormatBinStream_CompressedImageJPEGC;

            FileFormatJPEGC * RegisterFileFormatJPEG;
            FileFormatJPEGC * RegisterFileFormatJPEGL;
            FileFormatJPEGC * RegisterFileFormatMJPEG;
            FileFormatJPEGC * RegisterFileFormatMJPEGL;


      }; // End of JPEGif


#     ifndef RAVLIMAGE_JPEG_CCFILE
         extern JPEGif Jpeg;

         extern boolean JPEGif_call_resync_to_restart (j_decompress_ptr, int);
         // Helper routine that calls Jpeg.resync_to_restart ().
         // This is needed as DPImageIOJPegIBaseC requires a function pointer
         // for the resync function.

         // All other helper functions are only directly referenced within
         // JPEGif.cc hence they are not exported here. (While not directly
         // referenced elsewhere, the other helpers may be called fomr other
         // modules (even in user code) but this will be done via the function
         // pointers initialised in JPEGif.cc).

#     endif

   } // End of RavlImageN

#endif // End of duplicate inclusion protection
