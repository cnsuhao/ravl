// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//

#ifndef RAVLIMAGE_TIFFIF_HEADER
#  define RAVLIMAGE_TIFFIF_HEADER        1

#  include <dlfcn.h>
#  include <tiffio.h>

#  include "Ravl/DP/DynamicLink.hh"
#  include "Ravl/Image/TiffFormat.hh"


   namespace RavlImageN {

      class   TIFFif {
         public:
            TIFFif ()
            {
               void * handle;

               if ( (handle = DynamicLinkLoad ("libtiff.so", true)) )
               {
                  TIFFClientOpenFP.AsVoidPtr = dlsym (handle, "TIFFClientOpen");
                  TIFFCloseFP.AsVoidPtr = dlsym (handle, "TIFFClose");
                  TIFFDefaultStripSizeFP.AsVoidPtr = dlsym (handle, "TIFFDefaultStripSize");
                  TIFFFlushFP.AsVoidPtr = dlsym (handle, "TIFFFlush");
                  TIFFRGBAImageBeginFP.AsVoidPtr = dlsym (handle, "TIFFRGBAImageBegin");
                  TIFFRGBAImageGetFP.AsVoidPtr = dlsym (handle, "TIFFRGBAImageGet");
                  TIFFRGBAImageEndFP.AsVoidPtr = dlsym (handle, "TIFFRGBAImageEnd");
                  TIFFOpenFP.AsVoidPtr = dlsym (handle, "TIFFOpen");
                  TIFFSetFieldFP.AsVoidPtr = dlsym (handle, "TIFFSetField");
                  TIFFWriteScanlineFP.AsVoidPtr = dlsym (handle, "WriteScanline");
                  _TIFFfreeFP.AsVoidPtr = dlsym (handle, "_TIFFfree");
                  _TIFFmallocFP.AsVoidPtr = dlsym (handle, "_TIFFmalloc");
                  _TIFFmemcpyFP.AsVoidPtr = dlsym (handle, "_TIFFmemcpy");

#if USE_TIFFWRITE
                  RegisterFileFormatTIFF = new FileFormatTIFFC ("tiff","Tiff file IO. ");
#else
                  RegisterFileFormatTIFF = new FileFormatTIFFC ("tiff","Tiff file IO. (only reading supported)");
#endif

               }

            }
            // Default constructor

            ~TIFFif ()
            { }
            // Standard destructor function

            TIFF* ClientOpen(const char *filename, const char *mode, thandle_t clientdata,
                             TIFFReadWriteProc readproc, TIFFReadWriteProc writeproc,
                             TIFFSeekProc seekproc, TIFFCloseProc closeproc, TIFFSizeProc sizeproc,
                             TIFFMapFileProc mapproc, TIFFUnmapFileProc unmapproc
                            )
            {
                if (TIFFClientOpenFP.Call != NULL)
                   return (*TIFFClientOpenFP.Call) (filename, mode, clientdata, readproc,
                                                    writeproc, seekproc, closeproc,
                                                    sizeproc, mapproc, unmapproc
                                                   );
                else
                   return NULL;
            }
            // If possible call TIFF ClientOpen function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            void Close(TIFF *tif)
            {
               if (TIFFCloseFP.Call != NULL)
                  (*TIFFCloseFP.Call) (tif);
            }
            // If possible call TIFF TIFFClose function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            uint32 DefaultStripSize(TIFF *tif, uint32 estimate)
            {
               if (TIFFDefaultStripSizeFP.Call != NULL)
                  return (*TIFFDefaultStripSizeFP.Call) (tif, estimate);
               else
                  return 0;
            }
            // If possible call TIFF TIFFDefaultStripSize function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) we try to return an error
            // typical of a failure within the TIFF function itself. However, no
            // specific error is documented for this call, so we have employed an
            // error code that is based on what the majority of the other TIFF
            // calls return on error (based on TIFF version 20091104).

            int Flush(TIFF *tif)
            {
               if (TIFFFlushFP.Call != NULL)
                  return (*TIFFFlushFP.Call) (tif);
               else
                  return 0;
            }
            // If possible call TIFF TIFFFlush function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            int RGBAImageBegin(TIFFRGBAImage *img, TIFF* tif, int stopOnError, char emsg[1024])
            {
               if (TIFFRGBAImageBeginFP.Call != NULL)
                  return (*TIFFRGBAImageBeginFP.Call) (img, tif, stopOnError, emsg);
               else
                  return 0;
            }
            // If possible call TIFFRGBAImageBegin function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            int RGBAImageGet(TIFFRGBAImage *img, uint32* raster, uint32 width , uint32 height)
            {
               if (TIFFRGBAImageGetFP.Call != NULL)
                  return (*TIFFRGBAImageGetFP.Call) (img, raster, width, height);
               else
                  return 0;
            }
            // If possible call TIFF TIFFRGBAImageGet function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            void RGBAImageEnd(TIFFRGBAImage *img)
            {
               if (TIFFRGBAImageEndFP.Call != NULL)
                  (*TIFFRGBAImageEndFP.Call) (img);
            }
            // If possible call TIFF TIFFRGBAImageEnd function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            TIFF* Open(const char *filename, const char *mode)
            {
               if (TIFFOpenFP.Call != NULL)
                  return (*TIFFOpenFP.Call) (filename, mode);
               else
                  return NULL;
            }
            // If possible call TIFF TIFFOpen function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            int SetField(TIFF *tif, ttag_t tag, ...) 
            {
               if (TIFFSetFieldFP.Call != NULL)
                  return (*TIFFSetFieldFP.Call) (tif, tag);
               else
                  return 0;
            }
            // If possible call TIFF TIFFSetField function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            int WriteScanline(TIFF *tif, tdata_t buf, uint32 row, tsample_t sample)
            {
               if (TIFFWriteScanlineFP.Call != NULL)
                  return (*TIFFWriteScanlineFP.Call) (tif, buf, row, sample);
               else
                  return -1;
            }
            // If possible call TIFF TIFFWriteScanline function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            void free(tdata_t buffer)
            {
               if (_TIFFfreeFP.Call != NULL)
                  (*_TIFFfreeFP.Call) (buffer);
            }
            // If possible call TIFF _TIFFfree function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            tdata_t malloc(tsize_t size)
            {
               if (_TIFFmallocFP.Call != NULL)
                  return (*_TIFFmallocFP.Call) (size);
               else
                  return NULL;
            }
            // If possible call TIFF _TIFFmalloc function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

            void memcpy(tdata_t dest, const tdata_t src, tsize_t n)
            {
               if (_TIFFmemcpyFP.Call != NULL)
                  (*_TIFFmemcpyFP.Call) (dest, src, n);
            }
            // If possible call TIFF _TIFFmemcpy function.
            // If it is not possible to make the function call (for example if the 
            // TIFF library did not dynamically load) return an error typical of a
            // failure within the TIFF function itself (error code is based on TIFF
            // version 20091104).

         protected:

            union { void * AsVoidPtr;
                    TIFF* (* Call) ( const char *, const char *, thandle_t,
                                     TIFFReadWriteProc, TIFFReadWriteProc,
                                     TIFFSeekProc, TIFFCloseProc, TIFFSizeProc,
                                     TIFFMapFileProc, TIFFUnmapFileProc
                                   );
                  } TIFFClientOpenFP;

            union { void * AsVoidPtr;
                    void (* Call) ( TIFF * );
                  } TIFFCloseFP;

            union { void * AsVoidPtr;
                    uint32 (* Call) ( TIFF *, uint32 );
                  } TIFFDefaultStripSizeFP;

            union { void * AsVoidPtr;
                    int (* Call) ( TIFF * );
                  } TIFFFlushFP;

            union { void * AsVoidPtr;
                    int (* Call) ( TIFFRGBAImage *, TIFF*, int, char emsg[1024] );
                  } TIFFRGBAImageBeginFP;

            union { void * AsVoidPtr;
                    int (* Call) ( TIFFRGBAImage *, uint32*, uint32, uint32 );
                  } TIFFRGBAImageGetFP;

            union { void * AsVoidPtr;
                    void (* Call) ( TIFFRGBAImage *) ;
                  } TIFFRGBAImageEndFP;

            union { void * AsVoidPtr;
                    TIFF* (* Call) ( const char *, const char * );
                  } TIFFOpenFP;

            union { void * AsVoidPtr;
                    int (* Call) ( TIFF *, ttag_t, ...);
                  } TIFFSetFieldFP;

            union { void * AsVoidPtr;
                    int (* Call) (TIFF *, tdata_t, uint32, tsample_t );
                  } TIFFWriteScanlineFP;

            union { void * AsVoidPtr;
                    void (* Call) ( tdata_t );
                  } _TIFFfreeFP;

            union { void * AsVoidPtr;
                    tdata_t (* Call) ( tsize_t );
                  } _TIFFmallocFP;

            union { void * AsVoidPtr;
                    void (* Call) ( tdata_t, const tdata_t, tsize_t );
                  } _TIFFmemcpyFP;

    
            FileFormatTIFFC * RegisterFileFormatTIFF;
    
      }; // End of TIFFif

#ifdef RAVLIMAGE_IMGIOTIFF_CCFILE
      TIFFif TIFF;
#else
      extern TIFFif TIFF;
#endif

   } // End of RavlImageN

#endif // End of ifdef RAVLIMAGE_TIFFIF_HEADER
