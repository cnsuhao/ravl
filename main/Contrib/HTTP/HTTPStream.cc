// This file is part of RAVL, Recognition And Vision Library 
// Copyright (C) 2001, James Smith
// This code may be redistributed under the terms of the GNU Lesser
// General Public License (LGPL). See the lgpl.licence file for details or
// see http://www.gnu.org/copyleft/lesser.html
// file-header-ends-here
/////////////////////////////////////////////////////////
//! rcsid="$Id$"
//! lib=RavlHTTPIO
//! file="Ravl/Contrib/HTTP/HTTPIStream.cc"

#include "Ravl/IO/HTTPStream.hh"
#include "Ravl/StreamType.hh"
#include "Ravl/Threads/LaunchThread.hh"

#include <curl/curl.h>
#include <curl/types.h>
#include <curl/easy.h>

namespace RavlN {

   void InitHTTPStreamIO() {}


   size_t dataReady(void *ptr, size_t size, size_t nmemb, void *stream) {      
      return static_cast<HTTPIStreamC*>(stream)->Push(ptr,size,nmemb);
   }

   HTTPIStreamC::HTTPIStreamC(const StringC& url) {
      // Get URL
      LaunchThreadR(*this,&HTTPIStreamC::Get,url);
   }

   bool HTTPIStreamC::Get(StringC& url) {

      cerr << url << endl;

      // Restore full URL
      StringC fullurl("http:" + url);
      // Data
      CURL *handle = NULL;
      // Initialise CURL
      handle = curl_easy_init();
      if(handle) {
         // Set options
         curl_easy_setopt(handle, CURLOPT_URL, fullurl.chars());
         //curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, dataReady);
         //curl_easy_setopt(handle, CURLOPT_WRITEDATA, this);
         //curl_easy_setopt(handle, CURLOPT_MUTE, 1);
         //curl_easy_setopt(handle, CURLOPT_NOPROGRESS, 1);
         // Get the URL
         curl_easy_perform(handle);
         // Clean up
         curl_easy_cleanup(handle);
      }
      return true;
   }

   IntT HTTPIStreamC::Push(void *ptr, size_t size, size_t nmemb) {
      for (size_t i=0; i<size*nmemb; i++) {
         cerr << static_cast<unsigned char*>(ptr)[i];
      }
      return size*nmemb;
   }

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
    
      virtual IStreamC OpenI(const StringC &url, bool binary = false,bool buffered = true)
      { return HTTPIStreamC(url); }
      //: Open input stream.
    
   } Inst_StreamType_HTTPStream;

}
