
# CG- This is experimental at the moment, the display of strings is helpful though!

import re
import gdb.printing

#_type_char_ptr = gdb.lookup_type('char').pointer()

class RAVLStringPrinter(object):
     "Print a RavlN::StringC"
 
     def __init__(self, val):
         self.val = val
 
     def to_string(self):
         theRep = self.val['rep']['Where']
         len = theRep['len']
         return theRep['s'].string(length=len)
 
     def display_hint(self):
         return 'string'

class RAVLRCHandlePrinter(object):
     "Print a RavlN::StringC"
 
     def __init__(self, val):
         self.val = val
 
     def to_string(self):
         return self.val['body']
 
     def display_hint(self):
         return 'string'

class RAVLSizeBufferAccessPrinter(object):
     "Print a RavlN::StringC"
 
     def __init__(self, val):
         self.val = val
 
     def to_string(self):
         return "RavlN::SizeBufferAccessC %u : %s " % (self.val['sz'] ,self.val['buff'])

def build_pretty_printer():
         pp = gdb.printing.RegexpCollectionPrettyPrinter(
             "RAVL")
         pp.add_printer('RavlN::StringC', '^RavlN::StringC$', RAVLStringPrinter)
         #pp.add_printer('RavlN::SizeBufferAccessC', '^RavlN::SizeBufferAccessC<.*>$', RAVLSizeBufferAccessPrinter)
         #pp.add_printer('RavlN::RCHandleC', '^RavlN::RCHandleC<.*>$', RAVLRCHandlePrinter)
         return pp
     
print "Loading RAVL pretty printer."

gdb.printing.register_pretty_printer(
         gdb.current_objfile(),
         build_pretty_printer())

#gdb.pretty_printers['^RavlN::StringC$'] = RAVLStringPrinter