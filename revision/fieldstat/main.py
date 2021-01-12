import numpy as np
import pylab
import matplotlib as mpl
import os

exec(open('para.py').read())
exec(open('pop1d.py').read())

# grid1d_orient
fig = pylab.figure()
pylab.plot(x,grid1d_orient(x,l[0],oriarr[0],0,0,sig))
pylab.xlim(0,500)
pylab.ylabel('activity')
pylab.xlabel('location (cm)')

# load data if the file exists, else generate
fname = datapath+'v_1d_ori_l'+str(l)+'_M'+str(M2d)+'_R'+str(R)+'_sig'+str(sig)+'.txt'
if os.path.isfile(fname):
   with open(fname,'rb') as f:
       v = pickle.load(f)
else:
    generate_grid1d_orient(l,oriarr,M2d,R,sig)
    with open(fname,'rb') as f:
        v = pickle.load(f)

# random_weight model to generate input to place cells
a = random_weight(v,c=c,normalized='post')

# detect_fields
allfield,allfieldsize,th_bool = detectfield(a,nth=nth,mode='FIX')
    
# impose field constraint
if constrained:
    allfield,_,_ = constrainfield(allfield,allfieldsize,th_bool)

# printpc: print out example place cells
printpc(a,nth=nth)
        
# field_stat
r,p,nf = fieldstat(allfield,c,nth)

