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
     static_cast<OStreamC*>(stream)->write(static_cast<char*>(ptr),size*nmemb);
     return size*nmemb;
   }

  URLIStreamC::URLIStreamC(const StringC& url,bool buffered) :
    m_strTemp("/tmp/ravldl")
  {
     // Create temporary file
     m_strTemp.MkTemp();
     OStreamC tmpstrm(m_strTemp);
     // Fetch URL
     CURL *curl = NULL;
     // Initialise CURL
     curl = curl_easy_init();
     if(curl) {
       // Set options
       curl_easy_setopt(curl, CURLOPT_URL, url.chars());
       curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, dataReady);
       curl_easy_setopt(curl, CURLOPT_WRITEDATA, &tmpstrm);
       curl_easy_setopt(curl, CURLOPT_MUTE, 1);
       curl_easy_setopt(curl, CURLOPT_NOPROGRESS, 1);
       // Get the URL
       m_iError = curl_easy_perform(curl);
       // Clean up
       curl_easy_cleanup(curl);
     }
     // Recreate IStream from the read pipe
     (*this).IStreamC::operator=(IStreamC(m_strTemp,true,buffered));
   }

  URLIStreamC::~URLIStreamC() {
    m_strTemp.Remove();
  }

  StringC URLIStreamC::ErrorString() const {
    static const char* strerror[] = {
      "OK",
      "UNSUPPORTED_PROTOCOL",
      "FAILED_INIT",
      "URL_MALFORMAT",
      "URL_MALFORMAT_USER",
      "COULDNT_RESOLVE_PROXY",
      "COULDNT_RESOLVE_HOST",
      "COULDNT_CONNECT",
      "FTP_WEIRD_SERVER_REPLY",
      "FTP_ACCESS_DENIED",
      "FTP_USER_PASSWORD_INCORRECT",
      "FTP_WEIRD_PASS_REPLY",
      "FTP_WEIRD_USER_REPLY",
      "FTP_WEIRD_PASV_REPLY",
      "FTP_WEIRD_227_FORMAT",
      "FTP_CANT_GET_HOST",
      "FTP_CANT_RECONNECT",
      "FTP_COULDNT_SET_BINARY",
      "PARTIAL_FILE",
      "FTP_COULDNT_RETR_FILE",
      "FTP_WRITE_ERROR",
      "FTP_QUOTE_ERROR",
      "HTTP_NOT_FOUND",
      "WRITE_ERROR",
      "MALFORMAT_USER",          /* 24 - user name is illegally specified */
      "FTP_COULDNT_STOR_FILE",   /* 25 - failed FTP upload */
      "READ_ERROR",              /* 26 - could open/read from file */
      "OUT_OF_MEMORY",
      "OPERATION_TIMEOUTED",     /* 28 - the timeout time was reached */
      "FTP_COULDNT_SET_ASCII",   /* 29 - TYPE A failed */
      "FTP_PORT_FAILED",         /* 30 - FTP PORT operation failed */
      "FTP_COULDNT_USE_REST",    /* 31 - the REST command failed */
      "FTP_COULDNT_GET_SIZE",    /* 32 - the SIZE command failed */
      "HTTP_RANGE_ERROR",        /* 33 - RANGE "command" didn't work */
      "HTTP_POST_ERROR",
      "SSL_CONNECT_ERROR",       /* 35 - wrong when connecting with SSL */
      "FTP_BAD_DOWNLOAD_RESUME", /* 36 - couldn't resume download */
      "FILE_COULDNT_READ_FILE",
      "LDAP_CANNOT_BIND",
      "LDAP_SEARCH_FAILED",
      "LIBRARY_NOT_FOUND",
      "FUNCTION_NOT_FOUND",
      "ABORTED_BY_CALLBACK",
      "BAD_FUNCTION_ARGUMENT",
      "BAD_CALLING_ORDER",
      "HTTP_PORT_FAILED",        /* 45 - HTTP Interface operation failed */
      "BAD_PASSWORD_ENTERED",    /* 46 - my_getpass() returns fail */
      "TOO_MANY_REDIRECTS ",     /* 47 - catch endless re-direct loops */
      "UNKNOWN_TELNET_OPTION",   /* 48 - User specified an unknown option */
      "TELNET_OPTION_SYNTAX ",   /* 49 - Malformed telnet option */
      "OBSOLETE",                /* 50 - removed after 7.7.3 */
      "SSL_PEER_CERTIFICATE",    /* 51 - peer's certificate wasn't ok */
      "GOT_NOTHING",             /* 52 - when this is a specific error */
      "SSL_ENGINE_NOTFOUND",
      "SSL_ENGINE_SETFAILED"
    };
    return StringC(strerror[m_iError]);
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
