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
      "UNSUPPORTED PROTOCOL",
      "FAILED INIT",
      "URL MALFORMAT",
      "URL MALFORMAT USER",
      "COULDNT RESOLVE PROXY",
      "COULDNT RESOLVE HOST",
      "COULDNT CONNECT",
      "FTP WEIRD SERVER REPLY",
      "FTP ACCESS DENIED",
      "FTP USER PASSWORD INCORRECT",
      "FTP WEIRD PASS REPLY",
      "FTP WEIRD USER REPLY",
      "FTP WEIRD PASV REPLY",
      "FTP WEIRD 227 FORMAT",
      "FTP CANT GET HOST",
      "FTP CANT RECONNECT",
      "FTP COULDNT SET BINARY",
      "PARTIAL FILE",
      "FTP COULDNT RETR FILE",
      "FTP WRITE ERROR",
      "FTP QUOTE ERROR",
      "HTTP NOT FOUND",
      "WRITE ERROR",
      "MALFORMAT USER",          /* 24 - user name is illegally specified */
      "FTP COULDNT STOR FILE",   /* 25 - failed FTP upload */
      "READ ERROR",              /* 26 - could open/read from file */
      "OUT OF MEMORY",
      "OPERATION TIMEOUTED",     /* 28 - the timeout time was reached */
      "FTP COULDNT SET ASCII",   /* 29 - TYPE A failed */
      "FTP PORT FAILED",         /* 30 - FTP PORT operation failed */
      "FTP COULDNT USE REST",    /* 31 - the REST command failed */
      "FTP COULDNT GET SIZE",    /* 32 - the SIZE command failed */
      "HTTP RANGE ERROR",        /* 33 - RANGE "command" didn't work */
      "HTTP POST ERROR",
      "SSL CONNECT ERROR",       /* 35 - wrong when connecting with SSL */
      "FTP BAD DOWNLOAD RESUME", /* 36 - couldn't resume download */
      "FILE COULDNT READ FILE",
      "LDAP CANNOT BIND",
      "LDAP SEARCH FAILED",
      "LIBRARY NOT FOUND",
      "FUNCTION NOT FOUND",
      "ABORTED BY CALLBACK",
      "BAD FUNCTION ARGUMENT",
      "BAD CALLING ORDER",
      "HTTP PORT FAILED",        /* 45 - HTTP Interface operation failed */
      "BAD PASSWORD ENTERED",    /* 46 - my_getpass() returns fail */
      "TOO MANY REDIRECTS ",     /* 47 - catch endless re-direct loops */
      "UNKNOWN TELNET OPTION",   /* 48 - User specified an unknown option */
      "TELNET OPTION SYNTAX ",   /* 49 - Malformed telnet option */
      "OBSOLETE",                /* 50 - removed after 7.7.3 */
      "SSL PEER CERTIFICATE",    /* 51 - peer's certificate wasn't ok */
      "GOT NOTHING",             /* 52 - when this is a specific error */
      "SSL ENGINE NOTFOUND",
      "SSL ENGINE SETFAILED"
    };
    return StringC(strerror[m_iError]);
  }

  StringC URLIStreamC::URLEncode(const StringC& string) {
    StringC encoded;
    for (unsigned int i=0; i<string.Size(); i++) {
      if (!isalpha(string[i]) && !isdigit(string[i])) {
	StringC str;
	str.form("%%%X",string[i]);
	encoded += str;
      }
      else {
	encoded += string[i];
      }
    }
    cerr << encoded;
    return encoded;
  }

  StringC URLIStreamC::AddUserPass(const StringC& url,const StringC& user,const StringC& pass) {
    // URL-encode the username
    StringC euser = URLIStreamC::URLEncode(user);
    // URL-encode the password
    StringC epass = URLIStreamC::URLEncode(pass);
    // Split URL
    StringC address = url;
    StringC protocol = address.before(":");
    address = address.after("//");
    // Insert username and password after URL
    return protocol + StringC("://") + euser + StringC(":") + epass + StringC("@") + address;
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

   static class StreamType_HTTPSIStreamC 
      : public StreamTypeC
   {
   public:
      StreamType_HTTPSIStreamC()
      {}
      //: Default constructor.
    
      virtual const char *TypeName()
      { return "https"; }
      //: Get type of stream.
    
      virtual IStreamC OpenI(const StringC &url, bool binary = false,bool buffered = true) { 
	StringC rurl("https:" + url);
	return URLIStreamC(rurl,buffered); 
      }
      //: Open input stream.
    
   } Inst_StreamType_HTTPSStream;

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
