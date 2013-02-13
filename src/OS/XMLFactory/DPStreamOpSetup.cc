/*
 * DPStreamOpSetup.cc
 *
 *  Created on: Feb 10, 2013
 *      Author: charlesgalambos
 */


#include "Ravl/DP/StreamOp.hh"
#include "Ravl/SysLog.hh"
#include "Ravl/XMLFactory.hh"
#include "Ravl/DP/SequenceIO.hh"

namespace RavlN {

  //: Setup using an xml factory.

  void DPStreamOpBodyC::Setup(const XMLFactoryContextC &factory)
  {
    XMLFactoryContextC conns;
    if(factory.ChildContext("Connect",conns)) {
      for(RavlN::DLIterC<XMLTreeC> it(conns.Children());it;it++) {
        bool verbose = it->AttributeBool("verbose",false);
        if(it->Name() == "IPort" ||
            it->Name() == "OPort" ||
            it->Name() == "IPlug" ||
            it->Name() == "OPlug") {
          StringC portName = it->AttributeString("port","");
          StringC plugName = it->AttributeString("plug","");
          StringC entity = it->AttributeString("entity","");
          DPStreamOpC otherEntity;
          factory.UseChildComponent(entity,otherEntity);
          if(it->Name() == "IPort") {
            DPIPortBaseC iport;
            if(!GetIPort(portName,iport)) {
              RavlError("Failed to find iport '%s' ",portName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find IPort");
            }
            if(!otherEntity.SetIPort(plugName,iport)) {
              RavlError("Failed to find plug '%s' in entity '%s' ",portName.c_str(),entity.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find IPlug");
            }
            continue;
          }
          if(it->Name() == "OPort") {
            DPOPortBaseC oport;
            if(!GetOPort(portName,oport)) {
              RavlError("Failed to find iport '%s' ",portName.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPort");
            }
            if(!otherEntity.SetOPort(plugName,oport)) {
              RavlError("Failed to find plug '%s' in entity '%s' ",portName.c_str(),entity.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug");
            }
            continue;
          }
          if(it->Name() == "OPlug") {
            DPOPortBaseC oport;
            if(!portName.IsEmpty()) {
              if(!otherEntity.GetOPort(portName,oport)) {
                RavlError("Failed to find OPort '%s' ",portName.c_str());
                throw RavlN::ExceptionBadConfigC("Failed to find OPort");
              }
            } else {
              oport = DPOPortBaseC(otherEntity);
              if(!oport.IsValid()) {
                RavlError("Entity not an OPort ");
                throw RavlN::ExceptionBadConfigC("Failed to find OPort");
              }
            }
            if(!SetOPort(plugName,oport)) {
              RavlError("Failed to find plug '%s' in entity '%s' ",portName.c_str(),entity.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find OPlug");
            }
            continue;
          }
          if(it->Name() == "IPlug") {
            DPIPortBaseC iport;
            if(!portName.IsEmpty()) {
              if(!otherEntity.GetIPort(portName,iport)) {
                RavlError("Failed to find IPort '%s' ",portName.c_str());
                throw RavlN::ExceptionBadConfigC("Failed to find IPort");
              }
            } else {
              iport = DPIPortBaseC(otherEntity);
              if(!iport.IsValid()) {
                RavlError("Entity not an IPort ");
                throw RavlN::ExceptionBadConfigC("Failed to find OPort");
              }
            }
            if(!SetIPort(plugName,iport)) {
              RavlError("Failed to find plug '%s' in entity '%s' ",portName.c_str(),entity.c_str());
              throw RavlN::ExceptionBadConfigC("Failed to find IPlug");
            }
            continue;
          }
          // Should never get here.
          RavlAssert(0);
        }
        if(it->Name() == "OFile") {
          StringC filename = it->AttributeString("filename","");
          StringC format = it->AttributeString("format","");
          StringC plugName = it->AttributeString("plug","");
          DPOPlugBaseC oplug;
          if(!GetOPlug(plugName,oplug)) {
            RavlError("Failed to find plug '%s' ",plugName.c_str());
            throw RavlN::ExceptionBadConfigC("Failed to find IPlug");
          }
          DPOPortBaseC op;
          DPSeekCtrlC sc;
          if(!OpenOSequenceBase(op,sc,filename,format,oplug.OutputType(),verbose)) {
            RavlError("Failed to open output ",filename.c_str());
            throw RavlN::ExceptionBadConfigC("Failed to open output");
          }
          if(!oplug.ConnectPort(op)) {
            RavlError("Failed to connect plug");
            throw RavlN::ExceptionBadConfigC("Failed to connect output");
          }
          continue;
        }
        if(it->Name() == "IFile") {
          StringC filename = it->AttributeString("filename","");
          StringC format = it->AttributeString("format","");
          StringC plugName = it->AttributeString("plug","");
          DPIPlugBaseC iplug;
          if(!GetIPlug(plugName,iplug)) {
            RavlError("Failed to find plug '%s' ",plugName.c_str());
            throw RavlN::ExceptionBadConfigC("Failed to find IPlug");
          }
          DPIPortBaseC ip;
          DPSeekCtrlC sc;
          if(!OpenISequenceBase(ip,sc,filename,format,iplug.InputType(),verbose)) {
            RavlError("Failed to open input ",filename.c_str());
            throw RavlN::ExceptionBadConfigC("Failed to open output");
          }
          if(!iplug.ConnectPort(ip)) {
            RavlError("Failed to connect plug");
            throw RavlN::ExceptionBadConfigC("Failed to connect output");
          }
          continue;
        }
        RavlError("Unknown connection type '%s' ",it->Name().c_str());
        throw RavlN::ExceptionBadConfigC("Unknown connection type. ");
      }
    }
  }

}
