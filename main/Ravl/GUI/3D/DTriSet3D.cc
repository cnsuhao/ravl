//////////////////////////////////////////
//! rcsid="$Id$"

#include "Ravl/GUI/DTriSet3D.hh"
#include "Ravl/GUI/Canvas3D.hh"
#include "Ravl/SArr1Iter2.hh"
#include "Ravl/SArr1Iter.hh"
#include "GL/gl.h"

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

namespace RavlGUIN {
  
  //: Render object.
  
  bool DTriSet3DBodyC::Render(Canvas3DC &canvas) {
    if (!model.IsValid())
      return true; // Don't do anything.
    
    // Setup materials and colours as appropriate
    if (canvas.GetLightingMode()) {
      GLfloat ambient[]  = {0.2,0.2,0.2,1.0};
      GLfloat diffuse[]  = {0.9,0.9,0.9,1.0};
      glMaterialfv(GL_FRONT_AND_BACK,GL_AMBIENT,ambient);
      glMaterialfv(GL_FRONT_AND_BACK,GL_DIFFUSE,diffuse);
    } else 
      glColor3f(1.0,1.0,1.0);
    // Render
    Canvas3DRenderMode eMode = canvas.GetRenderMode();
    SArray1dC<VertexC> verts = model.Vertices();


    switch(eMode) {
    case C3D_SMOOTH:
    case C3D_POINT:
    case C3D_WIRE:
      glEnableClientState(GL_NORMAL_ARRAY);
      glNormalPointer(GL_DOUBLE,sizeof(VertexC),(void *)&(verts[0].Normal()));
    case C3D_FLAT:      
      glEnableClientState(GL_VERTEX_ARRAY);
      glVertexPointer(3,GL_DOUBLE,sizeof(VertexC),(void *)&(verts[0].Position()));
      break;
    }
    
    switch(eMode) 
      {
      case C3D_POINT: {
	// Draw individual points
	glDrawArrays(GL_POINTS,0,verts.Size());
      } break;
      case C3D_WIRE: {
	for(SArray1dIterC<TriC> it(model.Faces());it;it++) {
	  glBegin(GL_LINE);
	  glArrayElement(model.Index(*it,0));
	  glArrayElement(model.Index(*it,1));
	  glArrayElement(model.Index(*it,2));
	  glEnd();
	}
      } break;
      case C3D_FLAT: {
	ONDEBUG(cerr << "flat render. \n");
	IntT eGLShadeModel;
	glGetIntegerv(GL_SHADE_MODEL,&eGLShadeModel);
	glShadeModel(GL_FLAT); // Flat shading
	// Draw filled polygon
	for(SArray1dIterC<TriC> it(model.Faces());it;it++) {
	  GLNormal(it->FaceNormal());
	  glBegin(GL_POLYGON);
	  glArrayElement(model.Index(*it,0));
	  glArrayElement(model.Index(*it,1));
	  glArrayElement(model.Index(*it,2));
	  glEnd();
	}
	glShadeModel((GLenum)eGLShadeModel); // Restore old shade model
      } break;
      case C3D_SMOOTH: {	
	ONDEBUG(cerr << "Smooth render. \n");
	IntT eGLShadeModel;
	glGetIntegerv(GL_SHADE_MODEL,&eGLShadeModel);
	glShadeModel(GL_SMOOTH); // Flat shading
	// Draw filled polygon
	for(SArray1dIterC<TriC> it(model.Faces());it;it++) {
	  glBegin(GL_POLYGON);
	  glArrayElement(model.Index(*it,0));
	  glArrayElement(model.Index(*it,1));
	  glArrayElement(model.Index(*it,2));
	  glEnd();
	}
	glShadeModel((GLenum)eGLShadeModel); // Restore old shade model
      } break;
      };

    switch(eMode) {
    case C3D_SMOOTH:
    case C3D_POINT:
    case C3D_WIRE:
      glDisableClientState(GL_NORMAL_ARRAY);
    case C3D_FLAT:      
      glDisableClientState(GL_VERTEX_ARRAY);
      break;
    }
    return true;
  }
  
  ostream &operator<<(ostream &strm,const DTriSet3DC &) {
    RavlAssert(0);
    return strm;
  }
  
  istream &operator>>(istream &strm,DTriSet3DC &) {
    RavlAssert(0);
    return strm;
  }
}

