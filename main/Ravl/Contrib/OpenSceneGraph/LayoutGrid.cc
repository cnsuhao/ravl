
#include "Ravl/OpenSceneGraph/LayoutGrid.hh"
#include "Ravl/OpenSceneGraph/TypeConvert.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/RLog.hh"

namespace RavlOSGN {

  //: Constructor
  
  LayoutGridC::LayoutGridC(bool create)
   : LayoutC(create),
     m_stackAxis1(0),
     m_stackAxis2(1),
     m_gap(0.1),
     m_minSize(1.0)
  {
  }
  
  //: XML factory constructor
  
  LayoutGridC::LayoutGridC(const XMLFactoryContextC &factory)
   : LayoutC(true),
     m_stackAxis1(0),
     m_stackAxis2(1),
     m_gap(0.1),
     m_minSize(1.0)
  {
    Setup(factory);
  }
  
  //: Update the layout
  
  bool LayoutGridC::UpdateLayout()
  {
    rDebug("Updating layout. ");
    
    unsigned colMax = 1;
    const unsigned viewSize = m_nodes.size();
    unsigned rowMax = RavlN::Floor(RavlN::Sqrt(viewSize));
    colMax = rowMax;
    if(RavlN::Sqr(rowMax) < viewSize)
      colMax++;
    
    // Sort out grid displays
    Vector3dC rowStart(0,0,0);
    unsigned at = 0;
    
    std::vector<float> colSizes(colMax);
    
    for(unsigned c = 0;c < colMax;c++) {
      float width = m_minSize;
      for(unsigned r = 0;r < rowMax;r++) {
        unsigned at = r * colMax + c;
        if(at >= m_nodes.size())
          break;
        float csize = m_nodes[at]->Bounds()._max[m_stackAxis2] - m_nodes[at]->Bounds()._min[m_stackAxis2];
        if(csize > width)
          width = csize;
      }
      colSizes[c] = width;
    }
    
    
    for(unsigned r = 0;r < rowMax;r++) {
      float hight = m_minSize;
      Vector3dC vat = rowStart;
      
      // Work out the row hight.
      for(unsigned c = 0;c < colMax && at < m_nodes.size();c++,at++) {
        float rsize = m_nodes[at]->Bounds()._max[m_stackAxis1] - m_nodes[at]->Bounds()._min[m_stackAxis1];
        if(rsize > hight)
          hight = rsize;
        Vector3dC correctedPosition = vat - MakeVector3d(m_nodes[at]->Bounds()._min);
        m_nodes[at]->SetPosition(correctedPosition);
        rDebug("Start %s ",RavlN::StringOf(correctedPosition).data());
        vat[m_stackAxis2] += colSizes[c] + m_gap;
      }
      rowStart[m_stackAxis1] += hight + m_gap;
    }
    
    return true;
  }
  
  //: Process a callback.
  
  void LayoutGridC::DoCallback()
  {
    LayoutC::DoCallback();
  }
  
  //: Do setup from factory
  
  bool LayoutGridC::Setup(const XMLFactoryContextC &factory)
  {
    m_stackAxis1 = factory.AttributeInt("stackAxis1",0);
    if(m_stackAxis1 < 0 || m_stackAxis1 > 2) {
      rError("Invalid 1st stack axis %d ",m_stackAxis1);
      throw RavlN::ExceptionBadConfigC("Invalid stack axis. ");
    }
    m_stackAxis2 = factory.AttributeInt("stackAxis2",1);
    if(m_stackAxis2 < 0 || m_stackAxis2 > 2) {
      rError("Invalid 2nd stack axis  %d ",m_stackAxis2);
      throw RavlN::ExceptionBadConfigC("Invalid stack axis. ");
    }
    m_gap = factory.AttributeReal("gap",0.1);
    m_minSize = factory.AttributeReal("minSize",1.0);
    LayoutC::Setup(factory);
    return true;
  }
  
  //: Called when owner handles drop to zero.
  
  void LayoutGridC::ZeroOwners()
  {
    LayoutC::ZeroOwners();
  }
  
  void LinkLayoutGrid()
  {}
  
  static RavlN::XMLFactoryRegisterConvertC<LayoutGridC,LayoutC> g_registerXMLFactoryLayout("RavlOSGN::LayoutGridC");

}
