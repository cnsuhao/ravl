// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlURLIO
//! file="Ravl/Contrib/HTTP/URLIStream.cc"

#include "Ravl/IO/URLStream.hh"
#include "Ravl/StreamType.hh"
#include "Ravl/Threads/LaunchThread.hh"

#include <curl/curl.h>
#include <curl/types.h>
#include <curl/easy.h>

#include <unistd.h>

#ifndef CURLOPT_WRITEDATA
#define CURLOPT_WRITEDATA CURLOPT_FILE
#endif

namespace RavlN {

   void InitURLStreamIO() {}

   size_t dataReady(void *ptr, size_t size, size_t nmemb, void *stream) {      
     return static_cast<URLIStreamC*>(stream)->Push(ptr,size,nmemb);
   }

   URLIStreamC::URLIStreamC(const StringC& url,bool buffered) {
     // Setup pipe
     if(pipe(fd) == 0) {
       // Recreate IStream from the read pipe
       (*this).IStreamC::operator=(IStreamC(fd[0],true,buffered));       
       // Get URL
       LaunchThreadR(*this,&URLIStreamC::Get,url);
     }
   }

   bool URLIStreamC::Get(StringC& url) {
      // Data
      CURL *handle = NULL;
      // Initialise CURL
      handle = curl_easy_init();
      if(handle) {
         // Set options
         curl_easy_setopt(handle, CURLOPT_URL, url.chars());
         curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, dataReady);
         curl_easy_setopt(handle, CURLOPT_WRITEDATA, this);
         curl_easy_setopt(handle, CURLOPT_MUTE, 1);
         curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 1);
         // Get the URL
         CURLcode res = curl_easy_perform(handle);
	 // Test results
	 if(res != CURLE_OK)
	   cerr << "CURL error: " << res << endl;
         // Clean up
         curl_easy_cleanup(handle);
      }
      close(fd[1]); // Mark the end of file..
      return true;
   }

   IntT URLIStreamC::Push(void *ptr, size_t size, size_t nmemb) {
     // Push data onto pipe
     return write(fd[1],ptr,size*nmemb);
   }

   /////////////////// STREAM TYPES /////////////////////

   static class StreamType_HTTPIStreamC 
      : public StreamTypeC
   {
   public:
      StreamType_HTTPIStreamC()
      {}
      //: Default constructor.
    
      virtual const char *TypeName()
      { return "http"; }
      //: Get type of stream.
    
      virtual IStreamC OpenI(const StringC &url, bool binary = false,bool buffered = true) { 
	StringC rurl("http:" + url);
	return URLIStreamC(rurl,buffered); 
      }
      //: Open input stream.
    
   } Inst_StreamType_HTTPStream;

   static class StreamType_FTPIStreamC 
      : public StreamTypeC
   {
   public:
      StreamType_FTPIStreamC()
      {}
      //: Default constructor.
    
      virtual const char *TypeName()
      { return "ftp"; }
      //: Get type of stream.
    
      virtual IStreamC OpenI(const StringC &url, bool binary = false,bool buffered = true) { 
	StringC rurl("ftp:" + url);
	return URLIStreamC(rurl,buffered); 
      }
      //: Open input stream.
    
   } Inst_StreamType_FTPStream;

   static class StreamType_LDAPIStreamC 
      : public StreamTypeC
   {
   public:
      StreamType_LDAPIStreamC()
      {}
      //: Default constructor.
    
      virtual const char *TypeName()
      { return "ldap"; }
      //: Get type of stream.
    
      virtual IStreamC OpenI(const StringC &url, bool binary = false,bool buffered = true) { 
	StringC rurl("ldap:" + url);
	return URLIStreamC(rurl,buffered); 
      }
      //: Open input stream.
    
   } Inst_StreamType_LDAPStream;

}
