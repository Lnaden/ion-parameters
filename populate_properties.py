import numpy

ni = 21
#q,e,s
x=numpy.zeros([ni,3])
#LJ                 0           1           2           3          4           5    6
q = numpy.array([ 0.0,        0.0,        0.0,        0.0,       0.0,        0.0, 0.0])
e = numpy.array([0.10,       0.21,       0.32,       0.43,      0.54,       0.65, 0.8])
s = numpy.array([0.25, 0.57319535, 0.71205317, 0.81115873, 0.8906123, 0.95796625, 0.3])

#LJ                           7           8           9          10          11          12          13          14          15          16          17          18          19          20
q = numpy.append(q, [      -2.0,    -1.8516,    -1.6903,    -1.5119,    -1.3093,    -1.0690,    -0.7559,      2.000,     1.8516,     1.6903,     1.5119,     1.3093,     1.0690,     0.7559])
e = numpy.append(e, [      0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21,       0.21])
s = numpy.append(s, [0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535, 0.57319535])

for i in xrange(ni):
    x[i,0] = q[i]
    x[i,1] = e[i]
    x[i,2] = s[i]

numpy.save('n%i_init.npy'%ni, x)
