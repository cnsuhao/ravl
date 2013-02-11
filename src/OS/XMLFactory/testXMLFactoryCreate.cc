// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2009, OmniPerception Ltd
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
//! lib=RavlXMLFactory
//! author="Charles Galambos"
//! docentry=Ravl.API.Core.IO.XMLFactory

#include "Ravl/XMLFactory.hh"
#include "Ravl/XMLFactoryRegister.hh"
#include "Ravl/OS/SysLog.hh"
#include "Ravl/StrStream.hh"
#include "Ravl/XMLTree.hh"

namespace RavlN {
  extern void linkXMLFactoryRegister();
  
  class TestClassC : public RavlN::RCBodyVC
  {
  public:
    TestClassC()
    {
      linkXMLFactoryRegister();
    }
    
    TestClassC(const XMLFactoryContextC &factory);

    IntT Value() const {
      return m_value;
    }

    //! Handle to model.
    typedef RavlN::SmartPtrC<TestClassC> RefT;

  protected:
    int m_value;
  };
  
  TestClassC::TestClassC(const XMLFactoryContextC &factory)
  {
    SysLog(SYSLOG_DEBUG, "Factory path '%s' XMLNode:'%s' ", factory.Path().chars(), factory.Node().Name().chars());
    
    // Read value from context, default to 10 if not specified.
    m_value = factory.AttributeInt("value", 10);
  }
  
  //need to declare stream operators too
  inline std::istream &operator>>(std::istream &strm, TestClassC &obj)
  {
    RavlAssertMsg(0, "Not implemented. ");
    return strm;
  }
  //: Load from a stream.
  // Uses virtual constructor.

  inline std::ostream &operator<<(std::ostream &out, const TestClassC &obj)
  {
    RavlAssertMsg(0, "Not implemented. ");
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.

  inline RavlN::BinIStreamC &operator>>(RavlN::BinIStreamC &strm, TestClassC &obj)
  {
    RavlAssertMsg(0, "Not implemented. ");
    return strm;
  }
  //: Load from a binary stream.
  // Uses virtual constructor.

  inline RavlN::BinOStreamC &operator<<(RavlN::BinOStreamC &out, const TestClassC &obj)
  {
    RavlAssertMsg(0, "Not implemented. ");
    return out;
  }
  //: Save to a stream.
  // Uses virtual constructor.
  
  XMLFactoryRegisterC<TestClassC> g_registerTestClass("RavlN::TestClassC");
  
  RavlN::TypeNameC g_regname(typeid(TestClassC), "RavlN::TestClassC");


  class BigClassC : public RavlN::RCBodyVC
   {
   public:
     BigClassC()
     {
       linkXMLFactoryRegister();
     }

     BigClassC(const XMLFactoryContextC &factory);

     IntT Value() const {
       return m_testClass->Value();
     }

     //! Handle to model.
     typedef RavlN::SmartPtrC<BigClassC> RefT;

   protected:
     TestClassC::RefT m_testClass;
   };

   BigClassC::BigClassC(const XMLFactoryContextC &factory)
   {
     SysLog(SYSLOG_DEBUG, "Factory path '%s' XMLNode:'%s' ", factory.Path().chars(), factory.Node().Name().chars());

     // Get member from factory
     factory.UseComponent("Test", m_testClass);
   }

   //need to declare stream operators too
   inline std::istream &operator>>(std::istream &strm, BigClassC &obj)
   {
     RavlAssertMsg(0, "Not implemented. ");
     return strm;
   }
   //: Load from a stream.
   // Uses virtual constructor.

   inline std::ostream &operator<<(std::ostream &out, const BigClassC &obj)
   {
     RavlAssertMsg(0, "Not implemented. ");
     return out;
   }
   //: Save to a stream.
   // Uses virtual constructor.

   inline RavlN::BinIStreamC &operator>>(RavlN::BinIStreamC &strm, BigClassC &obj)
   {
     RavlAssertMsg(0, "Not implemented. ");
     return strm;
   }
   //: Load from a binary stream.
   // Uses virtual constructor.

   inline RavlN::BinOStreamC &operator<<(RavlN::BinOStreamC &out, const BigClassC &obj)
   {
     RavlAssertMsg(0, "Not implemented. ");
     return out;
   }
   //: Save to a stream.
   // Uses virtual constructor.

   XMLFactoryRegisterC<BigClassC> g_registerBigClass("RavlN::BigClassC");

   RavlN::TypeNameC g_regnameBigClass(typeid(BigClassC), "RavlN::BigClassC");

}

int main()
{
  RavlN::SysLogOpen("exXMLFactory", false);
  
  RavlN::StrIStreamC ss("<?xml version='1.0' encoding='UTF-8' ?>\n"
      "<?RAVL class='RavlN::XMLTreeC' ?>\n"
      "<Config verbose=\"false\" checkConfig=\"true\" >\n"
      "  <Test typename=\"RavlN::TestClassC\" value=\"0\" />\n"
      "  <Big typename=\"RavlN::BigClassC\" >"
      "        <Test typename=\"RavlN::TestClassC\" value=\"1\" />"
      "  </Big>"
      "</Config>\n");
  
  RavlN::XMLTreeC xmlTree(true);
  if (!xmlTree.Read(ss))
    return __LINE__;
  
  RavlN::XMLFactoryHC factory("test.xml", xmlTree);
  
  RavlN::SysLog(RavlN::SYSLOG_DEBUG, "Requesting component 'Test' ");
  
  // If we have a use component
  RavlN::TestClassC::RefT testClass1;
  if (!factory.UseComponent("Test", testClass1)) {
    SysLog(RavlN::SYSLOG_ERR, "Failed to find instance. ");
  }

  RavlN::TestClassC::RefT testClass2;
  if (!factory.UseComponent("Test", testClass2)) {
    SysLog(RavlN::SYSLOG_ERR, "Failed to find instance. ");
  }

  RavlInfo("1: %d, 2 %d", testClass1->Value(), testClass2->Value());

  /*
   * Create a new big class...
   */
  RavlN::BigClassC::RefT bigClass;
  if(!factory.CreateComponent("Big", bigClass)) {
    SysLog(RavlN::SYSLOG_ERR, "Failed to create big class. ");
  }
  RavlInfo("Test: %d, Big %d", testClass1->Value(), bigClass->Value());



}
