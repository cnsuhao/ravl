// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2003, University of Surrey
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! rcsid="$Id$"
//! lib=RavlImgIO1394dc
//! author="Charles Galambos"

#include "ImgIO1394dc.hh"
//#include "Ravl/Image/ImgIO1394dc.hh"

namespace RavlImageN {

const FrameRate1394dcC frameRates[] =
                                      {
                                        { 1.875  ,FRAMERATE_1_875 },
                                        { 3.75   ,FRAMERATE_3_75  },
                                        { 7.5    ,FRAMERATE_7_5   },
                                        { 15     ,FRAMERATE_15    },
                                        { 30     ,FRAMERATE_30    },
                                        { 60     ,FRAMERATE_60    },
                                        { -1     ,FRAMERATE_60    }
                                      };

}

