#!/usr/bin/perl

use DynaLoader;

sub PrintHelp {
  print "BinDep 2\n Some help: \n";
  print "  bindep [options] objectfiles \n";
  print "Options: \n";
  print "  -T(library name)  Specify a library that will be created soon. \n";
  print "  -L(directory)     Directory to seach for libraries. \n";
  print "  -l(library stem)  Stem of library name use in executable. \n";
  print "  -P(string)        Pass on string to stdout. \n";
  print "  -v                Verbose mode. \n";
  print " -I... -o... -O...  Are ingored. \n";
  exit ;
}

  my @LibPath, @Output, @Find, $found;
  my $libName, $fullPath;
  my $verbose=0;
  my $returnCode=0;
  
  # First check for verbose or help flags
  foreach (@ARGV) {
    if(/\A-v/) { # Verbose
      $verbose=1;
      next ;
    }
    if(/\A-h/) {  # Reqest for help.
      PrintHelp;
    }
  }

  foreach (@ARGV) {
    if(/\A-L([^ ]*)/) { # Library search dir - record it
      push @LibPath, "-L".$1 ;
      next ;
    }
#   if(/\A-R([^ ]*)/) {
#     push @LibPath, "-L".$1." " ;
#     next ;
#   }
    if(/\A-n32/) {  # Irix new 32 bit mode ?
      push @LibPath, "-L/usr/lib32" ;
      next ;
    }
    if(/\A-64/) {  # Irix 64 bit mode ?
      push @LibPath, "-L/usr/lib64" ;
      next ;
    }
    if(/\A-P(.*)/) { # Just pass on string.
      push @Output, $1 . " ";
      next ;
    }
    if(/\A-l([^ ]*)/) { # Library - find its path
      $libName = $1;
      if(exists $targLibs{$libName}) { # We know library may not exist yet
        push @Output, $targLibs{$libName} . " " ; # Use its path as specified
      }
      else
      { if($verbose) {
          print stderr "Searching for  '-l$libName' \n";
          print stderr "Using system path '@DynaLoader::dl_library_path' \n";
          print stderr "and '@LibPath' \n";
        }
        @Find = @LibPath ;
        push @Find, "-l" . $libName ;
        $found = DynaLoader::dl_findfile( @Find );
        if($verbose) { print stderr "Found '$found' \n"; }
        if ($found) { # Located library - use full path
          push @Output, $found . " " ;
        }
        else { # Library not found - drop
          $returnCode=-1;
          # push @Output, @Find[$#Find] . " " ;
          print stderr "Error - could not locate library @Find[$#Find].\n";
        }
      }
      next ;
    }
    if(/\A-T([^ ]*)/) { # Library we know may not exist yet - record the fact
      $fullPath = $1;
      $libName = $fullPath;
      $libName =~ s/\A\/.*\/lib//; # Strip off leading /path/lib
      $libName =~ s/\.a\Z//; # Strip off any trailing .a
      $libName =~ s/\.so\Z//; # Strip off any trailing .so
      if ($verbose) { print stderr "Recording $libName as '$fullPath' \n"; }
      $targLibs{$libName} = $fullPath;
      next ;
    }
    if(/\A-[WRLuIOvVgrpB][^ ]*/ )  { # Ignore these.
      next ;
    }
    if(/\A[^-][^ ]*/) { # Ingore direct dependancy on object files
      next ;
    }
    print stderr "Unknown argument $_\n ";
  }

  print @Output;
  print "\n";
  exit $returnCode;
