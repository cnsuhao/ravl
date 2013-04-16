// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2012, University of Surrey.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here

#ifndef RAVLIMAGE_JASPERIF_HEADER
#  define RAVLIMAGE_JASPERIF_HEADER        1

#  include <dlfcn.h>
#  include <jasper/jasper.h>

#  include "Ravl/DP/DynamicLink.hh"
#  include "Ravl/TypeName.hh"
#  include "Ravl/DP/Converter.hh"
#  include "Ravl/DP/FileFormatStream.hh"
#  include "Ravl/DP/FileFormatBinStream.hh"
#  include "Ravl/Image/CompressedImageJ2k.hh"
#  include "Ravl/Image/JasperFormat.hh"


   namespace RavlImageN {

      class JasperIF {
         public:
            JasperIF ()
            {
               void * handle;

               if ( (handle = DynamicLinkLoad ("libjasper.so", true)) )
               {
                  jas_cmprof_createfromclrspcFP.AsVoidPtr = dlsym (handle, "jas_cmprof_createfromclrspc");
                  jas_cmprof_destroyFP.AsVoidPtr = dlsym (handle, "jas_cmprof_destroy");
                  jas_image_chclrspcFP.AsVoidPtr = dlsym (handle, "jas_image_chclrspc");
                  jas_image_createFP.AsVoidPtr = dlsym (handle, "jas_image_create");
                  jas_image_decodeFP.AsVoidPtr = dlsym (handle, "jas_image_decode");
                  jas_image_destroyFP.AsVoidPtr = dlsym (handle, "jas_image_destroy");
                  jas_image_encodeFP.AsVoidPtr = dlsym (handle, "jas_image_encode");
                  jas_image_fmtfromnameFP.AsVoidPtr = dlsym (handle, "jas_image_fmtfromname");
                  jas_image_getfmtFP.AsVoidPtr = dlsym (handle, "jas_image_getfmt");
                  jas_image_lookupfmtbyidFP.AsVoidPtr = dlsym (handle, "jas_image_lookupfmtbyid");
                  jas_image_lookupfmtbynameFP.AsVoidPtr = dlsym (handle, "jas_image_lookupfmtbyname");
                  jas_image_readcmptFP.AsVoidPtr = dlsym (handle, "jas_image_readcmpt");
                  jas_image_writecmptFP.AsVoidPtr = dlsym (handle, "jas_image_writecmpt");
                  jas_initFP.AsVoidPtr = dlsym (handle, "jas_init");
                  jas_mallocFP.AsVoidPtr = dlsym (handle, "jas_malloc");
                  jas_matrix_createFP.AsVoidPtr = dlsym (handle, "jas_matrix_create");
                  jas_matrix_destroyFP.AsVoidPtr = dlsym (handle, "jas_matrix_destroy");
                  jas_stream_closeFP.AsVoidPtr = dlsym (handle, "jas_stream_close");
                  jas_stream_flushFP.AsVoidPtr = dlsym (handle, "jas_stream_flush");
                  jas_stream_fopenFP.AsVoidPtr = dlsym (handle, "jas_stream_fopen");
                  jas_stream_lengthFP.AsVoidPtr = dlsym (handle, "jas_stream_length");
                  jas_stream_memopenFP.AsVoidPtr = dlsym (handle, "jas_stream_memopen");
                  jas_stream_readFP.AsVoidPtr = dlsym (handle, "jas_stream_read");
                  jas_stream_seekFP.AsVoidPtr = dlsym (handle, "jas_stream_seek");

                  type1 = new TypeNameC (typeid(CompressedImageJ2kC),"RavlImageN::CompressedImageJ2kC");
                  DPConv_CompressedImageJ2K2RGBImage =
                          new RavlN::DPConverterBaseC ( RavlN::RegisterConversion(CompressedImageJ2K2RGBImage,
                                                                                  0.9,
                                                                                  "CompressedImageJ2kC RavlImageN::Convert(const ImageC<ByteRGBValueC> &)"
                                                                                 )
                                                      );
                  DPConv_CompressedImageJ2K2RGBImage =
                          new DPConverterBaseC (RegisterConversion (CompressedImageJ2K2RGBImage,
                                                                    1,
                                                                    "ImageC<ByteRGBValueC> RavlImageN::Convert(const CompressedImageJ2kC &)"
                                                                   )
                                               );
                  FileFormatStream_CompressedImageJ2kC =
                          new FileFormatStreamC<CompressedImageJ2kC> ();
                  FileFormatBinStream_CompressedImageJ2kC =
                          new FileFormatBinStreamC<CompressedImageJ2kC> ();

#if USE_JASPERWRITE
                  RegisterFileFormatJasper =
                          new FileFormatJasperC (1.0,"Jasper","Jasper file IO. ");
                  RegisterFileFormatJasperLossy1 =
                          new FileFormatJasperC (1.0/7.0,"JasperLossy1","Jasper file IO. (Compression rate = 1:7)");
                  RegisterFileFormatJasperLossy2 =
                          new FileFormatJasperC (0.08,"JasperLossy2","Jasper file IO. (Compression rate = 2:25)");
#else
                  RegisterFileFormatJasper =
                          new FileFormatJasperC (1.0,"Jasper","Jasper file IO. (only reading supported)");
                  RegisterFileFormatJasperLossy1 =
                          new FileFormatJasperC (1.0/7.0,"JasperLossy1","Jasper file IO. (only reading supported).");
                  RegisterFileFormatJasperLossy2 =
                          new FileFormatJasperC (0.08,"JasperLossy2","Jasper file IO. (only reading supported).");
#endif
               }

            }
            // Default constructor

            ~JasperIF ()
            { }
            // Standard destructor function

            jas_cmprof_t * cmprof_createfromclrspc (int clrspc)
            {
               if (jas_cmprof_createfromclrspcFP.Call != NULL)
                  return (*jas_cmprof_createfromclrspcFP.Call) (clrspc);
               else
                  return 0;
            }
            // If possible call Jasper jas_cmprof_createfromclrspc function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            void cmprof_destroy (jas_cmprof_t *prof)
            {
               if (jas_cmprof_destroyFP.Call != NULL)
                  (*jas_cmprof_destroyFP.Call) (prof);
            }
            // If possible call Jasper jas_cmprof_destroy function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) do nothing (as there is no
            // return value).

            jas_image_t * image_chclrspc (jas_image_t *image, jas_cmprof_t *outprof,
                                          int intent
                                         )
            {
               if (jas_image_chclrspcFP.Call != NULL)
                  return (*jas_image_chclrspcFP.Call) (image, outprof, intent);
               else
                  return 0;
            }
            // If possible call Jasper jas_image_chclrspc function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_image_t * image_create (int numcmpts, jas_image_cmptparm_t *cmptparms,
                                        jas_clrspc_t clrspc
                                       )
            {
               if (jas_image_createFP.Call != NULL)
                  return (*jas_image_createFP.Call) (numcmpts, cmptparms, clrspc);
               else
                  return 0;
            }
            // If possible call Jasper jas_image_create function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_image_t * image_decode (jas_stream_t *in, int fmt, char *optstr)
            {
               if (jas_image_decodeFP.Call != NULL)
                  return (*jas_image_decodeFP.Call) (in, fmt, optstr);
               else
                  return 0;
            }
            // If possible call Jasper jas_image_decode function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            void image_destroy (jas_image_t *image)
            {
               if (jas_image_destroyFP.Call != NULL)
                  (*jas_image_destroyFP.Call) (image);
            }
            // If possible call Jasper jas_image_destroy function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) do nothing (as there is no
            // return value).

            int image_encode (jas_image_t *image, jas_stream_t *out, int fmt, char *optstr)
            {
               if (jas_image_encodeFP.Call != NULL)
                  return (*jas_image_encodeFP.Call) (image, out, fmt, optstr);
               else
                  return -1;
            }
            // If possible call Jasper jas_image_encode function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int image_fmtfromname (char *filename)
            {
               if (jas_image_fmtfromnameFP.Call != NULL)
                  return (*jas_image_fmtfromnameFP.Call) (filename);
               else
                  return -1;
            }
            // If possible call Jasper jas_image_fmtfromname function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int image_getfmt (jas_stream_t *in)
            {
               if (jas_image_getfmtFP.Call != NULL)
                  return (*jas_image_getfmtFP.Call) (in);
               else
                  return -1;
            }
            // If possible call Jasper jas_image_getfmt function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_image_fmtinfo_t * image_lookupfmtbyid (int id)
            {
               if (jas_image_lookupfmtbyidFP.Call != NULL)
                  return (*jas_image_lookupfmtbyidFP.Call) (id);
               else
                  return 0;
            }
            // If possible call Jasper jas_image_lookupfmtbyid function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_image_fmtinfo_t * image_lookupfmtbyname (const char *name)
            {
               if (jas_image_lookupfmtbynameFP.Call != NULL)
                  return (*jas_image_lookupfmtbynameFP.Call) (name);
               else
                  return 0;
            }
            // If possible call Jasper jas_image_lookupfmtbyname function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int image_readcmpt (jas_image_t *image, int cmptno, jas_image_coord_t x,
                                jas_image_coord_t y, jas_image_coord_t width,
                                jas_image_coord_t height, jas_matrix_t *data
                               )
            {
               if (jas_image_readcmptFP.Call != NULL)
                  return (*jas_image_readcmptFP.Call) (image, cmptno, x, y, width, height, data);
               else
                  return -1;
            }
            // If possible call Jasper jas_image_readcmpt function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int image_writecmpt (jas_image_t *image, int cmptno, jas_image_coord_t x,
                                 jas_image_coord_t y, jas_image_coord_t width,
                                 jas_image_coord_t height, jas_matrix_t *data
                                )
            {
               if (jas_image_writecmptFP.Call != NULL)
                  return (*jas_image_writecmptFP.Call) (image, cmptno, x, y, width, height, data);
               else
                  return -1;
            }
            // If possible call Jasper jas_image_writecmpt function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int init ()
            {
               if (jas_initFP.Call != NULL)
                  return (*jas_initFP.Call) ();
               else
                  return 0;
                  /* jas_init always returns 0 - no concept of failure */
            }
            // If possible call Jasper jas_init function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).
    
            void * malloc (size_t size)
            {
               if (jas_mallocFP.Call != NULL)
                  return (*jas_mallocFP.Call) (size);
               else
                  return 0;
            }
            // If possible call Jasper jas_malloc function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper

            jas_matrix_t * matrix_create (int numrows, int numcols)
            {
               if (jas_matrix_createFP.Call != NULL)
                  return (*jas_matrix_createFP.Call) (numrows, numcols);
               else
                  return 0;
            }
            // If possible call Jasper jas_matrix_create function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            void matrix_destroy (jas_matrix_t *matrix)
            {
               if (jas_matrix_destroyFP.Call != NULL)
                  (*jas_matrix_destroyFP.Call) (matrix);
            }
            // If possible call Jasper jas_matrix_destroy function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) do nothing (as there is no
            // return value).

            int stream_close (jas_stream_t *stream)
            {
               if (jas_stream_closeFP.Call != NULL)
                  return (*jas_stream_closeFP.Call) (stream);
               else
                  return 0;
                  /* jas_stream_close always returns 0 - no concept of failure */
            }
            // If possible call Jasper jas_stream_close function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int stream_flush (jas_stream_t *stream)
            {
               if (jas_stream_flushFP.Call != NULL)
                  return (*jas_stream_flushFP.Call) (stream);
               else
                  return EOF;
            }
            // If possible call Jasper jas_stream_flush function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_stream_t * stream_fopen (const char *filename, const char *mode)
            {
               if (jas_stream_fopenFP.Call != NULL)
                  return (*jas_stream_fopenFP.Call) (filename, mode);
               else
                  return 0;
            }
            // If possible call Jasper jas_stream_fopen function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            long stream_length (jas_stream_t *stream)
            {
               if (jas_stream_lengthFP.Call != NULL)
                  return (*jas_stream_lengthFP.Call) (stream);
               else
                  return -1;
            }
            // If possible call Jasper jas_stream_length function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            jas_stream_t * stream_memopen (char *buf, int bufsize)
            {
               if (jas_stream_memopenFP.Call != NULL)
                  return (*jas_stream_memopenFP.Call) (buf, bufsize);
               else
                  return 0;
            }
            // If possible call Jasper jas_stream_memopen function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            int  stream_read (jas_stream_t *stream, void *buf, int cnt)
            {
               if (jas_stream_readFP.Call != NULL)
                  return (*jas_stream_readFP.Call) (stream, buf, cnt);
               else
                  return 0;
                  /* Act as no characters available */
            }
            // If possible call Jasper jas_stream_read function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

            long  stream_seek (jas_stream_t *stream, long offset, int origin)
            {
               if (jas_stream_seekFP.Call != NULL)
                  return (*jas_stream_seekFP.Call) (stream, offset, origin);
               else
                  return -1;
            }
            // If possible call Jasper jas_stream_seek function
            // If it is not possible to make the function call (for example if the 
            // Jasper library did not dynamically load) return an error typical of a
            // failure within the Jasper function itself (error code is based on Jasper
            // version 1.900.1).

         protected:
            union { void * AsVoidPtr;
                    jas_cmprof_t * (* Call)(int);
                  } jas_cmprof_createfromclrspcFP;

            union { void * AsVoidPtr;
                    void (* Call)(jas_cmprof_t *);
                  } jas_cmprof_destroyFP;

            union { void * AsVoidPtr;
                    jas_image_t * (* Call)(jas_image_t *, jas_cmprof_t *, int);
                  } jas_image_chclrspcFP;

            union { void * AsVoidPtr;
                    jas_image_t * (* Call)(int, jas_image_cmptparm_t *, jas_clrspc_t);
                  } jas_image_createFP;

            union { void * AsVoidPtr;
                    jas_image_t * (* Call)(jas_stream_t *, int, char *);
                  } jas_image_decodeFP;

            union { void * AsVoidPtr;
                    void (* Call)(jas_image_t *);
                  } jas_image_destroyFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_image_t *, jas_stream_t *, int, char *);
                  } jas_image_encodeFP;

            union { void * AsVoidPtr;
                    int (* Call)(char *);
                  } jas_image_fmtfromnameFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_stream_t *);
                  } jas_image_getfmtFP;

            union { void * AsVoidPtr;
                    jas_image_fmtinfo_t * (* Call)(int);
                  } jas_image_lookupfmtbyidFP;

            union { void * AsVoidPtr;
                    jas_image_fmtinfo_t * (* Call)(const char *);
                  } jas_image_lookupfmtbynameFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_image_t *, int, jas_image_coord_t, jas_image_coord_t,
                                 jas_image_coord_t, jas_image_coord_t, jas_matrix_t *
                                );
                  } jas_image_readcmptFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_image_t *, int, jas_image_coord_t, jas_image_coord_t,
                                 jas_image_coord_t, jas_image_coord_t, jas_matrix_t *
                                );
                  } jas_image_writecmptFP;

            union { void * AsVoidPtr; 
                   int (* Call)(void);
                  } jas_initFP;

            union { void * AsVoidPtr;
                    void * (* Call)(size_t);
                  } jas_mallocFP;

            union { void * AsVoidPtr;
                    jas_matrix_t * (* Call)(int, int);
                  } jas_matrix_createFP;

            union { void * AsVoidPtr;
                    void (* Call)(jas_matrix_t *);
                  } jas_matrix_destroyFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_stream_t *);
                  } jas_stream_closeFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_stream_t *);
                  } jas_stream_flushFP;

            union { void * AsVoidPtr;
                    jas_stream_t * (* Call)(const char *, const char *);
                  } jas_stream_fopenFP;

            union { void * AsVoidPtr;
                    long (* Call)(jas_stream_t *);
                  } jas_stream_lengthFP;
    
            union { void * AsVoidPtr;
                    jas_stream_t * (* Call)(char *, int);
                  } jas_stream_memopenFP;

            union { void * AsVoidPtr;
                    int (* Call)(jas_stream_t *, void *, int);
                  } jas_stream_readFP;

            union { void * AsVoidPtr;
                    long (* Call)(jas_stream_t *, long, int);
                  } jas_stream_seekFP;

            TypeNameC * type1;
            RavlN::DPConverterBaseC * DPConv_RGBImage2CompressedImageJ2K;
            RavlN::DPConverterBaseC * DPConv_CompressedImageJ2K2RGBImage;
            FileFormatStreamC<CompressedImageJ2kC> * FileFormatStream_CompressedImageJ2kC;
            FileFormatBinStreamC<CompressedImageJ2kC> * FileFormatBinStream_CompressedImageJ2kC;
            FileFormatJasperC * RegisterFileFormatJasper;
            FileFormatJasperC * RegisterFileFormatJasperLossy1;
            FileFormatJasperC * RegisterFileFormatJasperLossy2;

      }; // End of JasperIF


#ifdef RAVLIMAGE_IMGIOJASPER_CCFILE
     JasperIF Jas;
#else
     extern JasperIF Jas;
#endif

   } // End of RavlImageN

#endif // End of double inclusion protection
