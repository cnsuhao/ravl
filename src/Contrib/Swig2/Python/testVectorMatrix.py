#!/usr/bin/python

import Ravl

# make vector
v1 = Ravl.VectorC(10)

#fill it with some values
v1.Fill(1.0)

# we can print it out
print "v1 = {0}".format(v1)
print "Sum v1 = {0} ".format(v1.Sum())


# lets make another
v2 = Ravl.VectorC(10)
v2.Fill(2.0)
print "v2 = {0}".format(v2)

v3 = v1 + v2
print "v1 + v2 = {0}".format(v3)
print "0.5 * v1 = {0}".format(v1 * 0.5)
print "v1/10 = {0}".format(v1 / 10.0)
print "v1.Join(v2) = {0}".format(v1.Join(v2))
print "v1.EuclidDistance(v1) = {0}".format(v1.EuclidDistance(v1))
print "v1.EuclidDistance(v2) = {0}".format(v1.EuclidDistance(v2))
print "Log(v1) = {0}".format(Ravl.Log(v1))
print "Log(v2) = {0}".format(Ravl.Log(v2))

v4 = Ravl.RandomVector(10)
print v4

m = v4.OuterProduct()

print m

vm = Ravl.VectorMatrixC(v4, m)
print vm

