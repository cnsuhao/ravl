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
      size_t written = fwrite(ptr, size, nmemb, (FILE *)stream);
      return written;
   }

   HTTPIStreamC::HTTPIStreamC(const StringC& url) {
      // Get URL
      LaunchThreadR(*this,&HTTPIStreamC::Get,url);
   }

   bool HTTPIStreamC::Get(StringC& url) {
      // Restore full URL
      StringC fullurl("http:" + url);
      // Data
      CURL *curl;
      CURLcode res;      
      // Initialise CURL
      curl = curl_easy_init();
      if(curl) {
         // Set options
         curl_easy_setopt(curl, CURLOPT_URL, fullurl.chars());
         curl_easy_setopt(curl, CURLOPT_READFUNCTION, dataReady);
         curl_easy_setopt(curl, CURLOPT_MUTE, 1);
         curl_easy_setopt(curl, CURLOPT_NOPROGRESS, FALSE);
         // Get the URL
         res = curl_easy_perform(curl);
         // Clean up
         curl_easy_cleanup(curl);
      }
      return true;
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
