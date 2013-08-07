// This file is part of QMake, Quick Make System 
// Copyright (C) 2001-13, University of Surrey
// This code may be redistributed under the terms of the GNU General 
// Public License (GPL). See the gpl.licence file for details or
// see http://www.gnu.org/copyleft/gpl.html

// file-header-ends-here
//*********************************************************************************
/*
 * untouch, check that a file is at least a second old. (if it exists).
 */

/*! author="Charles Galambos" */
/*! docentry="Ravl.QMake.html" */

#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <time.h>
#include <utime.h>

#define DODEBUG 0
#if DODEBUG
#define ONDEBUG(x) x
#else
#define ONDEBUG(x)
#endif

short int verbose = 0;
short int force = 0;

int untouch(char *filename,int seconds) {
  struct stat buf;
  struct utimbuf utbuf;
  time_t now,target;
  
  // Find out what the time is currently set to.
  if(stat(filename,&buf)) {
    if(verbose)
      perror("can't stat file ");
    return 0;
  }
  ONDEBUG(printf("Time : %d \n",buf.st_mtime));

  // What time is it now ?
  now = time(0);
  target = now - seconds;

  // Is the file at least 'seconds' old already ?
  if(buf.st_mtime <= target)
    if (force==0 || buf.st_mtime==target) return 0; // No need to change anything.
  
  // Change the modification time.
  utbuf.modtime = target;
  utbuf.actime = target;
  if(utime(filename,&utbuf)) {
    perror("failed to change modification time. ");
    return 1;
  }
  return  0;
}

int main(int nargs,char **argv) {
  int seconds =1;
  short int show_help=0;
  ONDEBUG(printf("untouch '%s' \n",argv[1]));
  for(int i = 1;i < nargs;i++) {
    // Check for options.
    if(argv[i][0] == '-') {
      if(argv[i][1] == 'v') {
	if(argv[i][2] == 0) {
	  verbose = 1;
	  continue;
	}
      }
      if(argv[i][1] == 'f') {
	if(argv[i][2] == 0) {
	  force = 1;
	  continue;
	}
      }
      if(argv[i][1] == '?') {
	if(argv[i][2] == 0) {
	  show_help = 1;
	  break;
	}
      }
      printf("ERROR: Unrecognised option %s. \n",argv[i]);
      show_help=2;
      break;
    }
    // otherwise it must be a file.
    untouch(argv[i],seconds);
  }
  if(nargs < 2 || show_help ) {
    printf("Usage: untouch [OPTION]... FILE... \n");
    printf("Ensure the access and modification times of each FILE is at least one\n");
    printf("second prior to the current time.\n");
    printf("\n");
    printf("A FILE argument that does not exist is ignored.\n");
    printf("\n");
    printf("Valid OPTIONs are:\n");
    printf("  -f                     force the timestamp to be 1 second ago\n");
    printf("  -v                     verbose error reporting\n");
    printf("  -?                     print this help message and exit\n");
    printf("\n");


    exit(show_help!=1);
  }
  return 0;
}
