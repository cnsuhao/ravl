// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2005, OmniPerception Ltd.
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImgIOJasper

#include "Ravl/Image/ImgIOJasper.hh"
#include "Ravl/Array2dIter.hh"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlImageN {
  
  static bool jasInitDone = false;
  //: Jasper IO base class.
  
  DPImageIOJasperBaseC::DPImageIOJasperBaseC()
    : iostream(0),
      defaultFmt(-1)
  {
    if(!jasInitDone) {
      if(jas_init()) {
        // Init failed ?
      }
      jasInitDone = true;
    }
  }
  
  //: Destructor.
  
  DPImageIOJasperBaseC::~DPImageIOJasperBaseC()
  { Close(); }
  
  //: Close the stream.
  
  void DPImageIOJasperBaseC::Close() {
    if(iostream != 0 && jas_stream_close(iostream)) {
      cerr << "DPImageIOJasperBaseC::~DPImageIOJasperBaseC, Warning: Failed to close stream. ";
    }
    iostream = 0;
  }
  
  //: Open stream for reading
  
  bool DPImageIOJasperBaseC::OpenRead(const StringC &filename) {
    ONDEBUG(cerr << "DPImageIOJasperBaseC::OpenRead, Opening " << filename << " \n");
    iostream = jas_stream_fopen(filename,"rb");
    return iostream != 0;
  }
  
  //: Open stream for write
  
  bool DPImageIOJasperBaseC::OpenWrite(const StringC &filename) {
    iostream = jas_stream_fopen(filename,"wb");    
    return iostream != 0;    
  }

  //: Test if the current stream can be read.
  
  bool DPImageIOJasperBaseC::CanReadImage() { 
    if(iostream == 0)
      return false;
    int fmt = jas_image_getfmt(iostream);
    ONDEBUG(cerr << "DPImageIOJasperBaseC::CanReadImage, Fmt=" << fmt << "\n");
    if(fmt < 0) return false;
    jas_image_fmtinfo_t *fmtinfo = jas_image_lookupfmtbyid(fmt);
    if(fmtinfo == 0) return false;
    bool canRead = (fmtinfo->ops.decode != 0);
    ONDEBUG(cerr << "DPImageIOJasperBaseC::CanReadImage, FmtInfo='" << fmtinfo->desc << "' Read=" << canRead << " Write=" << (fmtinfo->ops.encode != 0) << "\n");
    return canRead;
  }
  
  //: Load image from current stream.
  
  jas_image_t *DPImageIOJasperBaseC::LoadImage() {
    ONDEBUG(cerr << "DPImageIOJasperBaseC::LoadImage(), Called. \n");
    RavlAssert(iostream != 0);
    jas_image_t *img = jas_image_decode(iostream, -1, 0);
    if(img == 0) {
      cerr << "DPImageIOJasperBaseC::LoadImage(), Failed. \n";
    }
    return img;
  }

  //: Convert image to RGB colour space.
  
  jas_image_t *DPImageIOJasperBaseC::ConvertToRGB(jas_image_t *image) {   
    jas_image_t *newimage;
    jas_cmprof_t *outprof;
    if (!(outprof = jas_cmprof_createfromclrspc(JAS_CLRSPC_SRGB))) {
      jas_image_destroy(image);
      cerr << "DPImageIOJasperBaseC::ConvertToRGB, Failed to create colour space conversion. \n";
      return 0;
    }
    if (!(newimage = jas_image_chclrspc(image, outprof, JAS_CMXFORM_INTENT_PER))) {
      jas_image_destroy(image);
      jas_cmprof_destroy(outprof);
      cerr << "DPImageIOJasperBaseC::ConvertToRGB, Colour space conversion failed. \n";
      return 0;
    }
    jas_image_destroy(image);
    jas_cmprof_destroy(outprof);
    return newimage;
  }
  
  //: Free an old image.
  
  bool DPImageIOJasperBaseC::FreeImage(jas_image_t *image) {
    jas_image_destroy(image);
    return true;
  }
  
  //: Convert an image into RAVL form.
  // img is free'd by operation.
  // Returns false if failes.
  
  bool DPImageIOJasperBaseC::Jas2Ravl(jas_image_t *img,ImageC<ByteRGBValueC> &rimg) {
    ONDEBUG(cerr << "DPImageIOJasperBaseC::Jas2Ravl, Called. \n");
    // Check the colour space is correct.
    if(jas_image_clrspc(img) != JAS_CLRSPC_SRGB)
      img = ConvertToRGB(img);
    
    //: Sort out the bounding box.
    Index2dC topLeft(jas_image_tly(img),jas_image_tlx(img));
    Index2dC bottomRight(jas_image_bry(img)-1,jas_image_brx(img)-1);
    
    IndexRange2dC imageRec(topLeft,bottomRight);
    if(imageRec.Area() <= 0) { // Empty image ?
      rimg = ImageC<ByteRGBValueC>();
      return true;
    }
    Index2dC size = imageRec.Size();
    
    ONDEBUG(cerr << "ImageSize=" << size << " Origin=" << topLeft  << " End=" << bottomRight << "\n");
    
    //: Create an image.
    ImageC<ByteRGBValueC> anImg(imageRec);
    
    int ncmpts = jas_image_numcmpts(img);
    
    jas_matrix_t *matrix[3];
    matrix[0] = 0;
    matrix[1] = 0;
    matrix[2] = 0;
    
    for(int i = 0;i < ncmpts;i++) {
      
      int ctype = jas_image_cmpttype(img,i);
      int lwidth = jas_image_cmptwidth(img, i);
      int lheight = jas_image_cmptheight(img,i);
      int ldepth = jas_image_cmptprec(img, i);
      
      ONDEBUG(cerr << "Layer=" << i << " Type=" << ctype << " Depth=" << ldepth << " Width=" << lwidth << " Height=" << lheight << " \n");
      int comp = -1;
      switch(ctype) {
      case JAS_IMAGE_CT_RGB_R: comp = 0; break;
      case JAS_IMAGE_CT_RGB_G: comp = 1; break;
      case JAS_IMAGE_CT_RGB_B: comp = 2; break;
      default:
        continue;
      }
      // Check its one of the colour components
      
      if(lwidth != size.Col() && lheight != size.Row()) {
        cerr << "DPImageIOJasperBaseC::Jas2Ravl, Component size mis-match. \n";
        return false;
      }
      if(ldepth > 8) {
        cerr << "DPImageIOJasperBaseC::Jas2Ravl, Component depth mis-match. \n";
        return false;
      }
      ONDEBUG(cerr << "DPImageIOJasperBaseC::Jas2Ravl, Component=" << comp << "\n");
      
      matrix[comp] = jas_matrix_create(size.Row().V(),size.Col().V());
      jas_image_readcmpt(img,i,0,0,size.Col().V(),size.Row().V(),matrix[comp]);
    }
    
    if(matrix[0] == 0 || matrix[1] == 0 || matrix[2] == 0) {
      cerr << "DPImageIOJasperBaseC::Jas2Ravl, Component missing from image. \n";
      // Clean up.
      
      for(int i = 0;i < 3;i++) {
        if(matrix[i] != 0)
          jas_matrix_destroy(matrix[i]);
      }
      return false;
    }
    
    // Copy data from matrix's into a RAVL image.
    
    Array2dIterC<ByteRGBValueC> it(anImg);
    int r = 0;
    for(;it;) {
      int c = 0;
      do {
        *it = ByteRGBValueC(jas_matrix_get(matrix[0],r,c),
                            jas_matrix_get(matrix[1],r,c),
                            jas_matrix_get(matrix[2],r,c)
                            );
        c++;
      } while(it.Next()) ;
      r++;
    }
    
    // Clean up.
    
    for(int i = 0;i < 3;i++)
      jas_matrix_destroy(matrix[i]);
    
    rimg = anImg;
    
    ONDEBUG(cerr << "DPImageIOJasperBaseC::Jas2Ravl, Done. \n");
    return true;
  }
  
  
  //: save an image to the current stream.
  
  bool DPImageIOJasperBaseC::SaveImage(jas_image_t *img) {
    ONDEBUG(cerr << "DPImageIOJasperBaseC::SaveImage, Writting image. \n");
    if(jas_image_encode(img,iostream, defaultFmt, 0)) {
      cerr << "DPImageIOJasperBaseC::SaveImage, Failed to encode image. \n";
      return false;
    }
    jas_stream_flush(iostream);
    return true;
  }
  
  //: Convert a Ravl image into a Jasper one.
  
  jas_image_t *DPImageIOJasperBaseC::Ravl2Jas(const ImageC<ByteRGBValueC> &img) {
    jas_image_cmptparm_t cmptparms[3];
    IndexRange2dC frame = img.Frame();
    
    jas_matrix_t *matrix[3];

    jas_image_coord_t x = img.Frame().LCol().V();
    jas_image_coord_t y = img.Frame().TRow().V();
    jas_image_coord_t width = img.Frame().Cols();
    jas_image_coord_t height = img.Frame().Rows();
    
    cerr << " x=" << x << " y=" << y << " width=" << width << " height=" << height << "\n";
    
    for(int i = 0;i < 3;i++) {
      cmptparms[i].tlx = x;
      cmptparms[i].tly = y;
      cmptparms[i].vstep = 1;
      cmptparms[i].hstep = 1;
      cmptparms[i].width = width;
      cmptparms[i].height = height;
      cmptparms[i].prec = 8;           // 8 bits per channel
      cmptparms[i].sgnd = false;        // Not signed
      
      matrix[i] = jas_matrix_create(height,width);
    }
    jas_clrspc_t clrspc = JAS_CLRSPC_SRGB;
    jas_image_t *jimg = jas_image_create(3,cmptparms,clrspc);
    
    Array2dIterC<ByteRGBValueC> it(img);
    int r = 0;
    for(;it;) {
      int c = 0;
      do {
        jas_matrix_set(matrix[0],r,c,it->Red());
        jas_matrix_set(matrix[1],r,c,it->Green());
        jas_matrix_set(matrix[2],r,c,it->Blue());
        c++;
      } while(it.Next()) ;
      r++;
    }
    
    
    // Write planes into image.
    
    for(int i = 0;i < 3;i++) {
      jas_image_writecmpt(jimg, i,
                          x,y,width,height,
                          matrix[i]);
    }
    
    jas_image_setcmpttype(jimg,0,JAS_IMAGE_CT_RGB_R);
    jas_image_setcmpttype(jimg,1,JAS_IMAGE_CT_RGB_G);
    jas_image_setcmpttype(jimg,2,JAS_IMAGE_CT_RGB_B);
    
    for(int i = 0;i < 3;i++)
      jas_matrix_destroy(matrix[i]);
    
#if 0
    int cspc = jas_image_clrspc(jimg);
    int fam = jas_clrspc_fam(cspc);
    cerr << "Cspc=" << cspc << " Fam=" << fam << "\n";
#endif
    return jimg;
  }
  
  //: Look for format by extention name.
  // returns -1 if not found.
  
  IntT DPImageIOJasperBaseC::FindFormatByFilename(const StringC &filename){
    int fmt = jas_image_fmtfromname(const_cast<char *>(filename.chars()));
    ONDEBUG(cerr << "DPImageIOJasperBaseC::FindFormatByFilename, File='" <<filename << "' Fmt=" << fmt << "\n");
    return fmt;
  }
  
}
