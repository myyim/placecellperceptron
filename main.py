# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.
# Note that the figure order, label and the content may have changed during revision.

import numpy as np
import pylab
import matplotlib as mpl
import pickle
import itertools
import os
figpath = './figure/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
datapath = './data/'
if not os.path.exists(datapath):
    os.makedirs(datapath)
exec(open('gridplacefunc.py').read())
exec(open('mlfunc.py').read())
exec(open('mathfunc.py').read())

font_size = 14
mpl.rcParams['axes.titlesize'] = font_size
mpl.rcParams['xtick.labelsize'] = font_size-2
mpl.rcParams['ytick.labelsize'] = font_size-2
mpl.rcParams['axes.labelsize'] = font_size-2
mpl.rcParams['legend.fontsize'] = font_size-7
mpl.rcParams['font.size'] = font_size-1
new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
fs = 16

### Figures
color = ['#1e90ff','#ff8c00','#3cb371']
color4 = ['#DC143C','#3cb371','#9932CC','#FFD700','#1e90ff','#ff8c00','#3cb371','m']

def fig1_2():
    rng = np.random.RandomState(1)
    fig = pylab.figure(figsize=[7,5])
    fig.text(0.02,0.95,'A',fontsize=fs)
    fig.text(0.33,0.95,'B',fontsize=fs)
    fig.text(0.6,0.95,'C',fontsize=fs)
    fig.text(0.02,0.48,'D',fontsize=fs)
    fig.text(0.33,0.48,'E',fontsize=fs)
    fig.text(0.6,0.48,'F',fontsize=fs)
    # A
    ax = pylab.subplot(231)
    #ax.text(1,5,'pattern set $x_j, j=1,\dots,S$')
    temp = np.arange(1.75,2.25,0.05)
    #for j in [0,1,3]:
        #pylab.plot(j+temp,4+0.1*rng.randn(temp.size),'m.',markersize=3)
    pylab.plot([2,3,5],3*[3],'o',c='0.8',ms=20)
    ax.set_title('random input')
    ax.text(2,3,'1      2   $\cdots$      $N$')
    ax.text(2,2,'w x',fontweight='bold')
    pylab.plot([4],[2],'o',c='0.5',mfc='w',ms=22)
    #ax.text(3,2,'$\sum$')
    ax.text(2,1,'$f(x_j)$ = sgn$(w\cdot x_j)$')
    ax.text(3,0.5,'or')
    #ax.text(1,0,r'Learning rule: $\Delta w = \alpha(y_j-f(x_j))x_j$')
    pylab.xlim(1,7)
    pylab.ylim(0,5)
    ax.axis('off')
    # B
    ax = pylab.subplot(232)
    ax.text(0,4,'realizable',fontsize=14)
    ax.text(0,3,'unrealizable',fontsize=14)
    ax.text(0,1,'$+$',color='g',fontsize=15)
    ax.text(0,2,'$-$',color='r',fontsize=15)
    ax.text(1,1,'Field',color='g',fontsize=10)
    ax.text(1,2,'~Field',color='r',fontsize=10)
    ax.set_xlim(-1,4)
    ax.set_ylim(0,5)
    ax.axis('off')
    # C
    ax = pylab.subplot(233)
    N = 5
    R = 3*N
    pcover = []
    for x in range(1,R+1):
        pcover.append(cover(N,x)/float(2**x))
    pylab.plot(np.arange(1,R+1,1)/float(N),pcover,'0.8',label='$N$='+str(N))
    N = 50
    R = 3*N
    pcover = []
    for x in range(1,R+1):
        pcover.append(cover(N,x)/float(2**x))
    pylab.plot(np.arange(1,R+1,1)/float(N),pcover,'0.6',label='$N$='+str(N))
    N = 500
    R = 3*N
    pcover = []
    xrange = range(N/10,R+1,N/10)
    xrange.extend(range(2*N-100,2*N+100))
    xrange.sort()
    for x in xrange:
        if x <= 1024:
            pcover.append(float(cover(N,x)/2)/2**(x-1))
        else:
            pcover.append(1./(2**x/cover(N,x)))
    pylab.plot(np.array(xrange)/float(N),pcover,'k',label='$N$='+str(N))
    #pylab.plot([0,float(R)/N],[0,0],'k--',lw=1)
    pylab.plot([0,float(R)/N],[0.5,0.5],'k--',lw=1)
    #pylab.plot([0,float(R)/N],[1,1],'k--',lw=1)
    pylab.plot([1]*2,[0,1],'k--',lw=1)
    pylab.plot([2]*2,[0,1],'k--',lw=1)
    pylab.xticks(np.array([1,N,2*N])/float(N),('1','$N$','2$N$'))
    pylab.yticks(np.arange(0,1.1,0.5))
    pylab.ylim(0,1.05)
    pylab.xlabel('number of patterns $P$')
    pylab.ylabel('realizable fraction')
    pylab.legend(bbox_to_anchor=(0.67,0.6),frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # D
    ax = pylab.subplot(234)
    R = 6
    x = np.arange(-0.5,R-0.5+0.01,0.01)
    temp = grid(x,2,0.5)+grid(x,3,1./3)-6
    pylab.plot(x,grid(x,2,0.5)-0.5,c=color[0])
    pylab.plot(x,grid(x,3,1./3)-2.5,c=color[1])
    pylab.plot(x,temp,'r')
    pylab.plot(x[111:189],temp[111:189],'g')
    pylab.plot([x[0],x[-1]],-4.5*np.ones(2),'k--')
    #pylab.plot(x,grid(x,2,0)-7.5,c=color[0])
    #pylab.plot(x,grid(x,3,0)-9.5,c=color[1])
    #pylab.plot(x,grid(x,3,2./3)-11.5,c=color[1])
    #pylab.plot(x,grid(x,2,0)+0.5+grid(x,3,0)+0.5*grid(x,3,2./3)-14,'k')
    pylab.ylim(-7,1)
    pylab.xlim(-2,6)
    ax.set_title('place cell output')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    pylab.xticks([])
    pylab.yticks([])
    ax.axis('off')
    ax = pylab.subplot(235)
    temp = np.arange(1.75-0.5/12,2.25-0.5/12,0.01/12.)
    pylab.plot(4+0.3*grid(temp-1.75,2./12,0),temp,c=color[0])
    pylab.plot(4+0.3*grid(temp-1.75,2./12,0.5),temp+1,c=color[0])
    pylab.plot(4+0.3*grid(temp-1.75,3./12,2./3),temp+3,c=color[1])
    pylab.ylim(1,7)
    pylab.xlim(0,5)
    ax.set_title('grid cell input')
    ax.text(1,4,'place cell')
    ax.text(1,2.5,'perceptron')
    ax.axis('off')
    # F
    ax = pylab.subplot(236)
    for j in range(5):
        pylab.plot([0,R],np.ones(2)-j*1.5,c=color[int(j>1)],lw=1)
        pylab.plot([0,R],np.zeros(2)-j*1.5,c=color[int(j>1)],lw=1)
        for k in range(R+1):
            pylab.plot([k]*2,np.array([0,1])-j*1.5,c=color[int(j>1)],lw=1)
    mat = [['$x_1$','$x_2$','$x_3$','$x_4$','$x_5$','$x_6$'],['1','0','1','0','1','0'],['0','1','0','1','0','1'],['1','0','0','1','0','0'],['0','1','0','0','1','0'],['0','0','1','0','0','1']]
    for j in range(6):
        for k in range(R):
            ax.text(k+0.32,1.7-j*1.5,mat[j][k],fontsize=12)
    ax.text(0,1,'x',fontweight='bold',fontsize=12)
    ax.text(0,3,'$x_P$',fontsize=12)
    pylab.ylim(-7,2)
    pylab.xlim(-0.1,R+2)
    ax.axis('off')
    pylab.subplots_adjust(left=0.05,top=0.95,right=0.95,bottom=0.1,hspace=0.5,wspace=0.3)
    pylab.savefig(figpath+'f12.svg')

def fig6():
    pylab.figure(figsize=[7,7])
    ax = pylab.subplot(222)
    seed = 5
    rng = np.random.RandomState(seed)
    wp1 = 0
    wp2 = 1
    bin = 1
    l = [31,43,59]
    Ng = 15
    Np = 100
    sig = 0.16
    pid = 0
    R = l[0]
    for j in range(len(l)-1):
        R *= l[j+1]
    x = np.arange(R)
    w = rng.lognormal(wp1,wp2,size=[Np,Ng])
    p1d = rng.rand(Np,Ng)
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    print '1D'
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid(x,l[iM/(Ng/len(l))],p1d[iN,iM],phsh=0.,gridmodel='gau',sig=sig)
        a[iN,:] = np.dot(w[iN,:],v[iN,:,:])
    # 1D
    nth = 2
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid1d = []
    nf = np.zeros(Np)
    for iN in range(Np):
        fid1d.append(list(pylab.find(af[iN,:]==1)))
        nf[iN] = np.sum(af[iN])
    print nf
    ifiall = []
    for iN in range(Np):   # choose one neuron for the IFI histogram
        ifiall.extend(np.diff(fid1d[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    if 1:
        pylab.figure()
        for iNg in range(Ng):
            pylab.plot(v[0,iNg,:]+iNg+1,'k')
        pylab.plot(a[0,:]/np.max(a[0,:])-1,'r')
        for k in np.where(af[0]==1)[0]:
           pylab.plot([k]*2,[-1,16],'b--')
        pylab.xlim(0,1500)
        pylab.yticks([])
        pylab.xlabel('location (up to 78647 cm)')
        pylab.ylabel('15 GCs in black; sum in red, rescaled')
        pylab.title('blue lines denote place fields')
        pylab.subplots_adjust(left=0.05,right=0.95,bottom=0.15)
        pylab.savefig(figpath+'fig6A.svg')
    count = np.zeros(Ng+1)
    for iNp in range(Np):
        for k in np.where(af[iNp]==1)[0]:
            idx = 0
            for iNg in range(Ng):
                gpeaks = pylab.find(v[iNp,iNg,:]>0.95)
                idx += np.any(k==gpeaks)
            count[idx] += 1
    pylab.bar(range(Ng+1),count/np.sum(count),color='k')
    pylab.yticks(np.arange(0.,0.4,0.1))
    pylab.ylabel('fraction')
    pylab.xlabel('number of coincidence peaks')
    pylab.subplots_adjust(left=0.05,right=0.95,hspace=0.4)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # IFI
    ax = pylab.subplot(223)
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*l[j]]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    acf1 = np.correlate(v[pid,0,:],v[pid,0,:],'same')
    acf2 = np.correlate(v[pid,Ng/3,:],v[pid,Ng/3,:],'same')
    acf3 = np.correlate(v[pid,2*Ng/3,:],v[pid,2*Ng/3,:],'same')
    acf1 /= np.max(acf1)
    acf2 /= np.max(acf2)
    acf3 /= np.max(acf3)
    pylab.plot(range(-R/2,R/2),np.max(ifiall)*(acf1+acf2+acf3)/3.,'k',label='ACF')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    pylab.title('1D periodic')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # slices through 2D
    ax = pylab.subplot(224)
    l = [31,43]
    Np = 100
    Ng = Ng*2/3
    px = rng.rand(Np,Ng)
    py = rng.rand(Np,Ng)
    w = rng.lognormal(wp1,wp2,size=[Np,Ng])
    ori = [0.39,0.46]
    R = 20000
    x = np.arange(R)
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
        a[iN,:] = np.dot(w[iN,:],v[iN,:,:])
    print '2D'
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    print np.sum(af[0])
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    ymax = max(y[0])
    j = 0
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    #pylab.plot([l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]+2*l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    #pylab.plot([2*l[j]+l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    j = 1
    #pylab.plot([l[j]]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.plot([2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    #pylab.plot([l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    #pylab.plot([l[j]+2*l[j]*np.sqrt(3)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    #pylab.plot([l[j]+2*l[j]*np.sqrt(7)]*2,[0,ymax+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    acf1 = np.correlate(v[pid,0,:],v[pid,0,:],'same')
    acf2 = np.correlate(v[pid,Ng/2,:],v[pid,Ng/2,:],'same')
    acf1 /= np.max(acf1)
    acf2 /= np.max(acf2)
    pylab.plot(range(-R/2,R/2),np.max(ifiall)*(acf1+acf2)/2.,'k',label='ACF')
    pylab.xlim(0,300)
    #pylab.yticks([0,5,10])
    pylab.xlabel('IFI')
    pylab.title('1D slices through 2D grid')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(top=0.9,wspace=0.3,hspace=0.4)
    pylab.savefig(figpath+'f6_Ng_'+str(Ng/2)+'_seed'+str(seed)+'.svg')

def fig6b():
    Ng = 10  # dinstinct for each neuron
    Np = 5
    N2plot = 5
    l = [31,43] #[53,87]
    R = 10000
    sig = 0.16
    bin = 1
    nth = 2
    #ori = [0.2153,0.3162] # ori2
    ori = [0.39,0.46] # ori1
    rng = np.random.RandomState(52)
    x = np.arange(R)
    px = rng.rand(Np,Ng)
    py = rng.rand(Np,Ng)
    w = rng.lognormal(0,0.25,size=[Np,Ng])
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    acf = np.zeros((Np,2*R/bin-1))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
            #v[iN,iM,:] = grid(x,l[iN],float(iM)/l[iN],phsh=0.,gridmodel='del'')    #
            a[iN,:] += w[iN,iM]*v[iN,iM,:]
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    # Activity
    fig = pylab.figure(figsize=[12,7])
    for iN in range(N2plot):
        ax = pylab.subplot(N2plot,1,iN+1)
        pylab.plot(x,a[iN,:])
        pylab.plot(fid[iN],[tharr[iN]]*len(fid[iN]),'ko')
        pylab.plot([0,R],[tharr[iN]]*2,'k',lw=1)
        if iN != Np-1:
            pylab.xticks([])
        if iN == 0:
            pylab.title('Activity on track: $\lambda$='+str(l)+';#grid='+str(Ng)+';R='+str(R)+';#STD='+str(nth))
    pylab.savefig(figpath+'slices_act_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'nstd'+str(nth)+'.png')
    # IFI
    fig = pylab.figure(figsize=[12,7])
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
    for iN in range(N2plot):
        ax = pylab.subplot(2,3,iN+1)
        ifi = np.diff(fid[iN])
        pylab.hist(ifi,np.arange(0.5,np.max(ifiall)+1,bin))
        if iN == 1:
            pylab.title('IFI: 5 neurons & all; bin='+str(bin)+';$\lambda$='+str(l)+';#grid='+str(Ng)+';R='+str(R)+';#STD='+str(nth))
        pylab.xlim(0,200)
    ax = pylab.subplot(236)
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))

    j = 0
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    j = 1
    pylab.plot([l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.plot([2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    pylab.xlim(0,200)
    pylab.savefig(figpath+'slices_ifi_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'b'+str(bin)+'nstd'+str(nth)+'.png')
    # ACF
    x = np.arange(0,R,bin)
    fig = pylab.figure(figsize=[12,7])
    for iN in range(Np):
        afbin = []
        for j in range(R/bin):
            afbin.append(np.sum([af[iN,bin*j:bin*(j+1)]]))
        acf[iN,:] = np.correlate(afbin,afbin,'full')
        if iN < N2plot:
            ax = pylab.subplot(2,3,iN+1)
            pylab.plot(x[1:],acf[iN,R/bin:])
            if iN == 1:
                pylab.title('ACF: 5 neurons & all; bin='+str(bin)+';$\lambda$='+str(l)+';#grid='+str(Ng)+';R='+str(R)+';#STD='+str(nth))
            pylab.xlim(0,200)
    ax = pylab.subplot(236)
    acfmax = np.max(np.sum(acf[:,R/bin:],0))
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*l[j]]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
            pylab.plot([k*l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
            pylab.plot([k*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        # 2
        pylab.plot([l[j]+l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        # 3
        pylab.plot([2*l[j]+l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        # 4
        pylab.plot([3*l[j]+l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([3*l[j]+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([3*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+3*l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+3*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+3*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+2*l[j]*np.sqrt(3)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+2*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,acfmax+1],'--',c=color[j],lw=0.5)
    pylab.plot(x[1:],np.sum(acf[:,R/bin:],0))
    pylab.xlim(0,200)
    pylab.savefig(figpath+'slices_acf_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'b'+str(bin)+'nstd'+str(nth)+'.png')
    # PSD
    bin = 1
    acf = np.zeros((Np,2*R/bin-1))
    x = np.arange(2*R/bin)/(2.*R/bin)
    for iN in range(Np):
        afbin = []
        for j in range(R/bin):
            afbin.append(np.sum([af[iN,bin*j:bin*(j+1)]]))
        acf[iN,:] = np.correlate(afbin,afbin,'full')
    fig = pylab.figure(figsize=[12,7])
    for iN in range(N2plot):
        ax = pylab.subplot(2,3,iN+1)
        psd = np.abs(np.fft.fft(np.append([0],acf[iN,:])))
        psdmax = np.max(psd[1:])
        for j in range(len(l)):
            pylab.plot([2.*np.sin(ori[j])/(np.sqrt(3)*l[j])]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([1./l[j]*(np.cos(ori[j])-np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([1./l[j]*(np.cos(ori[j])+np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([2*2.*np.sin(ori[j])/(np.sqrt(3)*l[j])]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([2*1./l[j]*(np.cos(ori[j])-np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([2*1./l[j]*(np.cos(ori[j])+np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
        pylab.plot(x[1:],psd[1:])
        if iN == 1:
            pylab.title('PSD: 5 neurons & all; bin='+str(bin)+';$\lambda$='+str(l)+';#grid='+str(Ng)+';R='+str(R)+';#STD='+str(nth))
        pylab.xlim(0,0.05)
    ax = pylab.subplot(236)
    psd = np.abs(np.fft.fft(np.append([0],np.sum(acf,0))))
    psdmax = np.max(psd[1:])
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*2.*np.sin(ori[j])/(np.sqrt(3)*l[j])]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([k*1./l[j]*(np.cos(ori[j])-np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
            pylab.plot([k*1./l[j]*(np.cos(ori[j])+np.sin(ori[j])/np.sqrt(3))]*2,[0,psdmax+10],'--',c=color[j],lw=1)
    pylab.plot(x[1:],psd[1:])
    pylab.xlim(0,0.05)
    pylab.savefig(figpath+'slices_psd_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'nstd'+str(nth)+'.png')

def fig3():
    fig = pylab.figure(figsize=[7,7])
    fig.text(0.05,0.96,'A',fontsize=fs)
    fig.text(0.05,0.65,'B',fontsize=fs)
    fig.text(0.05,0.35,'C',fontsize=fs)
    # A
    ax = pylab.subplot(711)
    pylab.plot([-8,-8],[1,3],c=color[0],lw=1)
    pylab.plot([-8,-8],[1,3],'ko',ms=15,mfc='w')
    pylab.plot([-5.5,-3.5,-4.5,-5.5],[2.5,2.5,1.5,2.5],c=color[1],lw=1)
    pylab.plot([-5.5,-3.5,-4.5,-5.5],[2.5,2.5,1.5,2.5],'ko',ms=15,mfc='w')
    pylab.plot([-1,-1],[1,3],c=color[0],lw=1)
    pylab.plot([1,1],[1,3],c=color[0],lw=1)
    pylab.plot([0,0],[0,2],c=color[0],lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],c=color[1],lw=1)
    pylab.plot([-1,1,0,-1],[3,3,2,3],c=color[1],lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],'ko',ms=15,mfc='w')
    pylab.plot([-1,1,0,-1],[3,3,2,3],'ko',ms=15,mfc='w')
    pylab.text(-8.5,3.5,'(1,0)')
    pylab.text(-8.5,0,'(0,1)')
    pylab.text(-6,3,'(1,0,0)')
    pylab.text(-4,3,'(0,1,0)')
    pylab.text(-5,0,'(0,0,1)')
    pylab.text(-2.5,3.5,'(1,0,1,0,0)')
    pylab.text(0.5,3.5,'(1,0,0,1,0)')
    pylab.text(0.5,1.5,'(1,0,0,0,1)')
    pylab.text(-2.5,0,'(0,1,1,0,0)')
    pylab.text(0.5,0,'(0,1,0,1,0)')
    pylab.text(0,-1,'(0,1,0,0,1)')
    pylab.text(-7,2,'x',fontsize=20)
    pylab.text(-2.5,2,'=',fontsize=20)
    pylab.xlim(-9,3.5)
    pylab.ylim(-0.5,4)
    ax.axis('off')
    # B
    ax = pylab.subplot(745)
    pylab.plot([-1,-1],[1,3],c=color[0],lw=1)
    pylab.plot([1,1],[1,3],c=color[0],lw=1)
    pylab.plot([0,0],[0,2],c=color[0],lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],c=color[1],lw=1)
    pylab.plot([-1,1,0,-1],[3,3,2,3],c=color[1],lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],'ko',ms=10,mfc='w')
    pylab.plot([-1,1,0,-1],[3,3,2,3],'ko',ms=10,mfc='w')
    #pylab.plot([-1],[3],'go',ms=15,alpha=0.5)
    pylab.xlim(-2,2)
    pylab.ylim(-1,4.5)
    pylab.text(-1.45,-1,'$------$',color='r',fontsize=11,fontweight='bold')
    pylab.text(-1.9,-1,'$+$',color='g',fontsize=11,fontweight='bold')
    ax.axis('off')
    ax = pylab.subplot(713)
    pylab.text(0,5,'Remove 1st\n location',fontsize=10)
    pylab.text(0,2,'Remove 2nd\n location \n (adjacent to 1st)',fontsize=10)
    pylab.xlim(-1,6)
    pylab.ylim(-1,6)
    ax.axis('off')
    ax = pylab.subplot(223)
    R = 6
    x = np.arange(-0.5,R-0.5+0.01,0.01)
    temp = grid(x,2,0)+grid(x,2,0.5)+grid(x,3,0)+grid(x,3,2./3)-11
    pylab.plot(x,grid(x,2,0)-0.5,c=color[0])
    pylab.plot(x,grid(x,2,0.5)-2.5,c=color[0])
    pylab.plot(x,grid(x,3,0)-4.5,c=color[1])
    pylab.plot(x,grid(x,3,2./3)-6.5,c=color[1])
    pylab.plot(x,temp,'r')
    fw = 59
    pylab.plot(x[:fw],temp[:fw],'g')
    pylab.plot(x[-fw:],temp[-fw:],'g')
    pylab.plot(x[300-fw:300+fw],temp[300-fw:300+fw],'g')
    pylab.plot([x[0],x[-1]],-8.7*np.ones(2),'k--')
    #pylab.plot(x,grid(x,2,0)-7.5,c=color[0])
    #pylab.plot(x,grid(x,3,0)-9.5,c=color[1])
    #pylab.plot(x,grid(x,3,2./3)-11.5,c=color[1])
    #pylab.plot(x,grid(x,2,0)+0.5+grid(x,3,0)+0.5*grid(x,3,2./3)-14,'k')
    pylab.ylim(-10,1)
    pylab.xlim(-2,6)
    #ax.set_title('place cell output')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    pylab.xticks([])
    pylab.yticks([])
    ax.axis('off')
    pylab.subplots_adjust(left=0.1,top=0.97,right=0.95,bottom=0.1,wspace=0.2)
    pylab.savefig(figpath+'f3.svg')

def fig5():
    mpl.rcParams['legend.fontsize'] = font_size-5
    mpl.rcParams['axes.titlesize'] = font_size-3
    rng = np.random.RandomState(3)
    fig = pylab.figure(figsize=[7,7])
    fig.text(0.02,0.95,'A',fontsize=fs)
    fig.text(0.5,0.95,'B',fontsize=fs)
    fig.text(0.02,0.64,'C',fontsize=fs)
    fig.text(0.5,0.64,'D',fontsize=fs)
    fig.text(0.02,0.32,'E',fontsize=fs)
    fig.text(0.5,0.32,'F',fontsize=fs)
    # A
    l = [3,4]
    N = len(l)
    Sc = int(np.round(testrange(l)[0]))
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    fname = 'pcorr'+str(l[0])+str(l[1])
    if os.path.isfile(datapath+fname+'.txt'):
        with open(datapath+fname+'.txt','rb') as f:
            pall,p2,p3,p4,p5,p6 = pickle.load(f)
    else:
        pall,p2,p3,p4,p5,p6 = frac_vs_S(l,R,return6=1)
        with open(datapath+fname+'.txt','wb') as f:
            pickle.dump((pall,p2,p3,p4,p5,p6),f)
    ax = pylab.subplot(321)
    pylab.plot(range(1,R+1),np.ones(R),'o-',ms=5,label='1')
    pylab.plot(range(2,R+1),p2,'o-',ms=5,label='2')
    pylab.plot(range(3,R+1),p3,'o-',ms=5,label='3')
    pylab.plot(range(4,R+1),p4,'o-',ms=5,label='4')
    pylab.plot(range(5,R+1),p5,'o-',ms=5,label='5')
    pylab.plot(range(6,R+1),p6,'o-',ms=5,label='6')
    pylab.plot([R]*2,[0,1],'k--',lw=1)
    pylab.text(11,0.85,'$L$')
    pylab.ylim(0,1.05)
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    pylab.legend(loc=3,frameon=False)
    pylab.ylabel('realizable fraction')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('$\lambda=\{3,4\}$')
    # B
    ax = pylab.subplot(322)
    mpl.rcParams['legend.fontsize'] = font_size-7
    pylab.plot(range(1,R+1),pall,'ko-',ms=5,label='grid')
    pylab.plot([Sc]*2,[0,1],'k--',lw=1)
    pylab.plot([R]*2,[0,1],'k--',lw=1)
    pcover = []
    for x in range(1,R+1):
        pcover.append(cover(Sc,x)/float(2**x))
    pylab.plot(np.arange(1,R+1,1),pcover,'o-',c='c',ms=5,label='random (rank)')
    pcover = []
    for x in range(1,R+1):
        pcover.append(cover(np.sum(l),x)/float(2**x))
    pylab.plot(np.arange(1,R+1,1),pcover,'o-',c='#8A2BE2',ms=5,label='random (dim)')
    for k in range(5):
        pwcon = random_weightcon(np.sum(l),R,1,k+1)
        if k == 0:
            pylab.plot(np.arange(1,R+1,1),pwcon,'o-',c='#B0C4DE',ms=5,label='random,\nconstrained')
        else:
            pylab.plot(np.arange(1,R+1,1),pwcon,'o-',c='#B0C4DE',ms=5)
    pylab.legend(loc=6,frameon=False)
    pylab.plot(range(1,R+1),pall,'ko-',ms=5,label='grid')
    pylab.ylim(0,1.05)
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    pylab.text(6.2,0.05,'$l^*$')
    pylab.text(11,0.05,'$L$')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # C
    mpl.rcParams['legend.fontsize'] = font_size-5
    ax = pylab.subplot(323)
    lsum = []
    ctrackq0 = []
    ctrackq2 = []
    ctrackq4 = []
    # [lmin,lmax)
    lmin = 3
    lmax = 20
    for j in range(100):
        M = rng.randint(2,7)
        l = lmin + rng.rand(M)*(lmax-lmin)
        temp = testrange(l)
        lsum.append(np.sum(l))
        ctrackq0.append(temp[0])
        ctrackq2.append(temp[2])
        ctrackq4.append(temp[4])
        #ctrackq8.append(temp[-1])
        pylab.plot(np.array(temp)/np.sum(l),'-',lw=1)
    pylab.xlabel('resolution $q$')
    pylab.ylabel('$R^q_{re}/\Sigma$')
    pylab.ylim(0,1.05)
    pylab.yticks([0,1])
    pylab.xticks(range(0,9,2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # F
    ax = pylab.subplot(324)
    pylab.plot(lsum,ctrackq0,'.',lw=0,ms=5,label='$q$=0')
    pylab.plot(lsum,ctrackq2,'.',lw=0,ms=5,label='$q$=2')
    pylab.plot(lsum,ctrackq4,'.',lw=0,ms=5,label='$q$=4')
    #pylab.plot(lsum,ctrackq8,'.',lw=0,ms=5,label='$S_{real}^{q=8}$')
    pylab.legend(loc=2,frameon=False)
    pylab.plot([1,120],[1,120],'k--',lw=1)
    pylab.xlim(0,110.5)
    pylab.ylim(0,110.5)
    pylab.xticks(range(0,101,50))
    pylab.yticks(range(0,101,50))
    pylab.xlabel('period sum $\Sigma$')
    pylab.ylabel('$R^q_{re}$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.15,top=0.95,right=0.95,bottom=0.1,wspace=0.3,hspace=0.4)
    pylab.savefig(figpath+'f5.svg')

def fig8():
    is_qp = 0   # use quadratic programming with the options of constrained weights
    if is_qp:
        print 'Using quadratic programming'
    else:
        print 'Using sklearn SVM'
    mpl.rcParams['legend.fontsize'] = font_size-6
    rng = np.random.RandomState(4)
    import seaborn as sns
    gridmodel = 'del'
    mth = np.arange(0,1.0001,0.01)
    sym = ['^','+','x']
    num = 10
    # A
    fig = pylab.figure(figsize=[7,8.5*0.8])
    fig.text(0.02,0.65,'A',fontsize=fs)
    fig.text(0.48,0.65,'B',fontsize=fs)
    fig.text(0.02,0.3,'C',fontsize=fs)
    fig.text(0.48,0.3,'D',fontsize=fs)
    ax = fig.add_subplot(323)
    l = [31,43]
    N = len(l)
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    K = 6
    u = act_mat_grid_binary(l)
    u /= 2
    mode = 's1000'
    if os.path.exists(datapath+'f8_s1000.txt'):
        with open(datapath+'f8_s1000.txt','rb') as f:
            margin,rmargin,smargin,numKarr,rnumKarr,snumKarr = pickle.load(f)
    else:
        margin,rmargin,smargin,numKarr,rnumKarr,snumKarr = margin_gridvsrandom(l,K=K,num=num,mode=mode)
    margin = margin[:K]
    rmargin = rmargin[:K]
    smargin = smargin[:K]
    numKarr = numKarr[:K]
    rnumKarr = rnumKarr[:K]
    snumKarr = snumKarr[:K]
    # for violin plot: random
    kmat1 = []
    for k in range(K):
        kmat1.extend([k+1]*np.sum(rnumKarr[k]))
    mmat1 = [item for sublist in rmargin for subsublist in sublist for item in subsublist]
    nmat1 = ['random']*np.sum(np.sum(rnumKarr))
    # for violin plot: shuffle
    kmat2 = []
    for k in range(K):
        kmat2.extend([k+1]*np.sum(snumKarr[k]))
    mmat2 = [item for sublist in smargin for subsublist in sublist for item in subsublist]
    nmat2 = ['shuffled']*np.sum(np.sum(snumKarr))
    kmat1 = np.array(kmat1)
    kmat2 = np.array(kmat2)
    #sns.violinplot(np.append(kmat1,kmat2),np.append(mmat1,mmat2),np.append(nmat1,nmat2),inner=None,linewidth=.4,scale='width',width=0.5,bw=.2,gridsize=100)
    sns.violinplot(kmat1,mmat1,inner=None,linewidth=.4,scale='width',width=0.5,bw=.2,gridsize=100,color='m')
    sns.violinplot(kmat2,mmat2,inner=None,linewidth=.4,scale='width',width=0.5,bw=.2,gridsize=100,color='b')
    #sns.violinplot(kmat1,mmat1,nmat1,inner=None,linewidth=.4,bw=.2)
    #sns.violinplot(kmat2,mmat2,nmat2,inner=None,linewidth=.4,bw=.2,color='#ff7f0e')
    for k in range(1,K+1):
        for mu in margin[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    ax.set_yticks(np.arange(0,0.5,0.2))
    ax.set_xlim(-0.5,K-0.5)
    ax.set_ylim(0,0.4)
    ax.set_xlabel('number of fields ($K$)')
    ax.set_ylabel('margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # B
    ax = fig.add_subplot(324)
    m_inset = []
    for k in range(1,K+1):
        count,temp = np.histogram(rng.rand(int(mode[1:])*num),np.array([0.]+list(np.cumsum(numKarr[k-1]))+[nCr(R,k)])/nCr(R,k)) # randomly draw field arrangements based on fractions of margins
        for j in range(len(count)-1):
            m_inset.extend([margin[k-1][j]]*count[j])
        print np.array([0.]+list(np.cumsum(numKarr[k-1]))+[nCr(R,k)]),count,m_inset
    count0L = []
    for m in mth:
        count0L.append(np.sum(np.array(m_inset)>=m))
    countr = []
    for m in mth:
        countr.append(np.sum(mmat1>=m))
    counts = []
    for m in mth:
        counts.append(np.sum(mmat2>=m))
    temp = float(K)*int(mode[1:])*num
    ax.plot(mth,np.array(count0L)/temp,'k')
    ax.plot(mth,np.array(countr)/temp,color='m',lw=1.5) #1f77b4
    ax.plot(mth,np.array(counts)/temp,color='b',lw=1.5)    #ff7f0e
    ax.plot([0,1],2*[1],'k--',lw=1)
    ax.set_ylim(0,1)
    ax.set_xlim(0,0.4)
    ax.set_xlabel('margin $\kappa$')
    ax.set_ylabel('CDF')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # CD
    color = ['b','g','r','c','m']
    inp = [0,100]  # first entry is zero
    numr = len(inp)
    u = act_mat_grid_binary(l)
    if 1:
        for id in range(2):
            margin_spatial = []
            margin_new = []
            for n in range(num):
                for r in range(numr): # from no addition to numr additions
                    if n == 0:
                        margin_spatial.append([])
                        if r > 0:
                            margin_new.append([])
                    if n > 0 and r == 0:
                        continue
                    if inp[r] > 0:
                        if id == 0:
                            # uniform
                            randinp = 0.054*0.2*rng.rand(inp[r],u.shape[1])    #2*2/74 *0.2
                            # gaussian
                            #randinp = 0.1*rng.randn(inp[r],u.shape[1])
                        elif id == 1:
                            randinp = np.zeros((inp[r],u.shape[1]))
                            for j in range(inp[r]):
                                #randinp[j,rng.choice(range(u.shape[1]),10,replace=False)] = 1
                                randinp[j,rng.choice(range(u.shape[1]),7,replace=False)] = 1    # 1333*2/74.*0.2
                        v = np.append(u,randinp,axis=0)
                    else:
                        v = np.copy(u)
                    # Normalization
                    for jj in range(u.shape[1]):
                        # L1
                        v[:,jj] = v[:,jj]/np.sum(v[:,jj])
                        # L2
                        #v[:,jj] = v[:,jj]/pylab.norm(v[:,jj])
                    for k in range(1,K+1):
                        if n == 0:
                            margin_spatial[r].append([])
                            if r > 0:
                                margin_new[r-1].append([])
                        partition = partitions(k)
                        part = []
                        for p in partition:
                            if np.all(np.array(p)<=np.min(l)):
                                part.append(list(p))
                                # Young diagram
                                mat = np.zeros((l[0],l[1]))
                                for j in range(len(p)):
                                    mat[:p[j],j] = 1
                                #pylab.figure()
                                #pylab.imshow(mat,aspect='auto')
                                i1 = np.tile(range(l[0]),l[1])
                                i2 = np.tile(range(l[1]),l[0])
                                Y = mat[i1,i2]
                                print Y
                                m,w,b = svm_margin(v.T,Y)
                                margin_spatial[r][k-1].append(m)
                                dec = np.sign(np.dot(w.T,v)+b)
                                dec[dec<0] = 0
                                print k,p,r,abs(np.sum(np.abs(Y-dec))),m
                        is_fit = 0
                        while r>0 and k>1 and not is_fit:     # unrealizable becomes realizable
                            print n,r,k
                            Y = np.zeros(u.shape[1])
                            Y[0:k] = 1
                            rng.shuffle(Y)
                            m,w,b = svm_margin(u.T,Y)
                            dec = np.sign(np.dot(w.T,u)+b)
                            dec[dec<0] = 0
                            if abs(np.sum(np.abs(Y-dec)))>1e-10: # not realizable
                                print 'found not realizable with u'
                                m,w,b = svm_margin(v.T,Y)
                                dec = np.sign(np.dot(w.T,v)+b)
                                dec[dec<0] = 0
                                if abs(np.sum(np.abs(Y-dec)))<1e-10:    # realizable
                                    print 'found realizable with v'
                                    margin_new[r-1][k-1].append(m)
                                    is_fit = 1
            with open(datapath+'fig8CD_'+str(id)+'.txt','wb') as f:
                pickle.dump((margin_spatial,margin_new),f)
    color = ['b','#0f9b8e','#0cfc73']
    for id in range(2):
        ax = pylab.subplot(3,2,5+id)
        with open(datapath+'fig8CD_'+str(id)+'.txt','r') as f:
            margin_spatial,margin_new = pickle.load(f)
        # violin
        if 1:
            mmat1 = [item for sublist in margin_spatial[numr-1][:K] for item in sublist]
            mmat2 = [item for sublist in margin_new[numr-2][:K] for item in sublist]
            kmat1 = []
            kmat2 = []
            for k in range(1,K+1):
                kmat1.extend([k]*len(margin_spatial[numr-1][k-1]))
                kmat2.extend([k]*len(margin_new[numr-2][k-1]))
            nmat1 = ['exiting']*len(kmat1)
            nmat2 = ['new']*len(kmat2)
            sns.violinplot(kmat1,mmat1,inner=None,linewidth=.4,scale='width',width=0.5,bw=.2,gridsize=100,color=color4[0])
            sns.violinplot(np.append([1],kmat2),np.append([-1],mmat2),inner=None,linewidth=.4,scale='width',width=0.5,bw=.2,gridsize=100,color='#ff7f0e')
        for k in range(1,K+1):
            for r in [0,numr-1]:#range(numr):
                if r > 0:
                    # mean
                    if 0:
                        pn = len(margin_spatial[r][k-1])/num
                        for j in range(pn):
                            pylab.plot([k],[np.mean(margin_spatial[r][k-1][j::pn])],'.',ms=10,fillstyle='none',c=color[r-1],label=str(inp[r]))
                        pylab.plot([k],[np.mean(margin_new[r-1][k-1])],'^',ms=6,fillstyle='none',c=color[r-1])
                    # all trials
                    if 0:
                        pylab.plot([k]*len(margin_spatial[r][k-1]),margin_spatial[r][k-1],'x',c=color[r-1],label=str(inp[r])+' random input')
                else:
                    for mu in margin_spatial[0][k-1]:
                        pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k',label='grid')
            #if k == 1 and id == 0:
                #pylab.legend(loc=1,frameon=False)
        pylab.xlim(-0.5,K-0.5)
        pylab.ylim(0,0.45)
        pylab.yticks([0,0.2,0.4])
        #pylab.xticks(range(1,K+1))
        if id == 0:
            pylab.ylabel('margin')
        pylab.xlabel('# fields ($K$)')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    fig.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.1,wspace=0.4,hspace=0.5)
    fig.savefig(figpath+'f8'+'qp'*is_qp+'.svg')

def fig7(seed):
    rng = np.random.RandomState(seed)
    wp1 = 0
    wp2 = 1
    bin = 1
    #l = [31,43,59]
    l = [31,43]
    ori = [0.39,0.46]
    Ng = 600
    Np = 10#0
    sig = 0.16
    R = l[0]
    for j in range(len(l)-1):
        R *= l[j+1]
    R = 2000
    nK = 4
    nadd = 4
    nex = 10
    x = np.arange(R)
    w = rng.lognormal(wp1,wp2,size=[Np,Ng])
    for iN in range(Np):
        if np.sum(w[iN,:]) > 0:
            w[iN,:] /= sum(w[iN,:])
    p1d = rng.rand(Np,Ng)
    p1dy = rng.rand(Np,Ng)
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    a1 = np.zeros((Np,R))
    fieldshift = []
    nfieldchange = [] # number of fields to begin with
    nfieldstay = []
    for j in range(nK):
        nfieldchange.append([])
        nfieldstay.append([])
        for jj in range(nadd):
            nfieldchange[j].append([])
            nfieldstay[j].append([])
    for iN in range(Np):
        for iM in range(Ng):
            #v[iN,iM,:] = grid(x,l[iM/(Ng/len(l))],p1d[iN,iM],phsh=0.,gridmodel='gau',sig=sig)
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],xph=p1d[iN,iM],yph=p1dy[iN,iM],sig=0.16,full=1,mode='exp')
        a[iN,:] = np.dot(w[iN,:],v[iN,:,:])
        print 'Neuron ',iN
        idx = fieldloc(a[iN])
        pk = a[iN,idx]
        sid = pylab.argsort(pk)
        sid = sid[-1::-1]
        pkarr = pk[sid]
        idall = idx[sid]
        pool = range(R)
        for K in range(1,nK+1):   # number of existing fields
            print 'K = ',K
            idarr = idall[:K]
            th = np.sum(pkarr[K-1:K+1])/2  # set threshold by searching for the bumps
            pool = [elem for elem in pool if (elem < idarr[K-1]-10 or elem > idarr[K-1]+10)]
            for k in range(1,nadd+1):   # added fields
                for j in range(nex):  # total number of examples at difficult target locations
                    print 'Example '+str(j)
                    pylab.figure(figsize=[6,7])
                    ax = pylab.subplot(511) # original
                    #pylab.plot(a[iN,:])
                    pylab.plot(np.dot(w[iN,:],v[iN,:,:]))
                    pylab.plot([0,R],[th]*2,'k',lw=1)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(a[iN,:]),np.max(a[iN,:])],'r--',lw=1)
                    ax = pylab.subplot(512) # original thresholded
                    ath = a[iN,:]-th
                    ath[ath<0] = 0
                    pylab.plot(ath)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(ath),np.max(ath)],'r--',lw=1)
                    # weight perturbation
                    ax = pylab.subplot(513)
                    dw = rng.rand(Ng)
                    w1 = w[iN,:] + dw*2/1000. # 0.0001
                    w1 /= np.sum(w1)
                    a1 = np.dot(w1,v[iN,:,:])
                    ath = a1-th
                    ath[ath<0] = 0
                    pylab.plot(ath)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(ath),np.max(ath)],'r--',lw=1)
                    ax = pylab.subplot(514)
                    dw = rng.rand(Ng)
                    w1 = w[iN,:] + dw*2/1000.  # 0.001
                    w1 /= np.sum(w1)
                    a1 = np.dot(w1,v[iN,:,:])
                    ath = a1-th
                    ath[ath<0] = 0
                    pylab.plot(ath)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(ath),np.max(ath)],'r--',lw=1)
                    ax = pylab.subplot(515)
                    dw = rng.rand(Ng)
                    w1 = w[iN,:] + dw*2/1000.  # 0.001
                    w1 /= np.sum(w1)
                    a1 = np.dot(w1,v[iN,:,:])
                    ath = a1-th
                    ath[ath<0] = 0
                    pylab.plot(ath)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(ath),np.max(ath)],'r--',lw=1)
                    ax = pylab.subplot(513) # added activity
                    loc = rng.choice(pool,k,replace=False)
                    dw = np.sum(v[iN,:,loc],0) #np.sum(v[iN,:,loc-25:loc+26],1)  # size of the induced plateau
                    alpha = 1e-5
                    w1 = np.copy(w[iN,:])
                    while np.min(np.dot(v[iN,:,loc],w1)) < th: # when field is not there
                        if alpha > 0.1:
                            print 'alpha = ',alpha,' --> reset in ',iN,K,k,j
                            loc = rng.choice(pool,k,replace=False)
                            dw = np.sum(v[iN,:,loc],0) #np.sum(v[iN,:,loc-25:loc+26],1)  # size of the induced plateau
                            alpha = 1e-5
                        alpha += 1e-5
                        w1 = w[iN,:] + alpha*dw
                        if np.sum(w1) > 0:
                            w1 /= np.sum(w1)
                    a_dw = np.dot(alpha*dw/np.sum(w1),v[iN,:,:])
                    pylab.plot(a_dw)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(a_dw),np.max(a_dw)],'r--',lw=1)
                    for id in loc:
                        pylab.plot([id]*2,[np.min(a_dw),np.max(a_dw)],'g--',lw=1)
                    #print alpha,np.dot(v[iN,:,loc],w1),th
                    ax = pylab.subplot(514) # after learning
                    a1 = np.dot(w1,v[iN,:,:])   # a1 is activity after learning
                    pylab.plot(a1)
                    pylab.plot([0,R],[th]*2,'k',lw=1)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(a[iN,:]),np.max(a[iN,:])],'r--',lw=1)
                    for id in loc:
                        pylab.plot([id]*2,[np.min(a1),np.max(a1)],'g--',lw=1)
                    ax = pylab.subplot(515) # after learning thresholded
                    ath1 = a1-th
                    ath1[ath1<0] = 0
                    pylab.plot(ath1)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(ath1),np.max(ath1)],'r--',lw=1)
                    for id in loc:
                        pylab.plot([id]*2,[np.min(ath1),np.max(ath1)],'g--',lw=1)
                    loc1 = fieldloc(ath1)
                    for id in idarr:
                        fieldshift.extend(loc1-id)
                    nfieldchange[K-1][k-1].append(loc1.size-k-np.sum(ath1[idarr]>0)+np.sum(ath1[idarr]==0))
                    temp = 0
                    for id in idarr:
                        if np.any(id==loc1):
                            temp += 1
                    nfieldstay[K-1][k-1].append(temp)
                pylab.savefig(figpath+'seed{}pc{}K{}A{}eg{}.svg'.format(seed,iN,K,k,j))
                pylab.close('all')
                with open(datapath+'fieldchange_seed'+str(seed)+'.txt','wb') as f:
                    pickle.dump((fieldshift,nfieldchange,nfieldstay),f)

def readfig7():
    nK = 4
    nadd = 4
    R = 2000
    x = np.arange(R)
    fieldshift = []
    nfieldchange = []
    nfieldstay = []
    l = [31,43]
    ori = [0.39,0.46]
    v1 = grid1d_orient(x,l[0],ori[0],xph=0,yph=0,sig=0.16,full=1,mode='exp')
    v2 = grid1d_orient(x,l[1],ori[1],xph=0,yph=0,sig=0.16,full=1,mode='exp')
    for j in range(nK):
        nfieldchange.append([])
        nfieldstay.append([])
        for jj in range(nadd):
            nfieldchange[j].append([])
            nfieldstay[j].append([])
    for seed in range(1,11):
        with open(datapath+'fieldchange_seed'+str(seed)+'.txt','rb') as f:
            a,b,c = pickle.load(f)
        fieldshift.extend(a)
        for j in range(nK):
            for jj in range(nadd):
                nfieldchange[j][jj].extend(b[j][jj])
                nfieldstay[j][jj].extend(c[j][jj])
    pylab.figure(figsize=[9,3])
    ax = pylab.subplot(131)
    fieldshifthist,dump = pylab.histogram(fieldshift,np.arange(-R+0.5,R))
    pylab.plot(range(-R+1,R),fieldshifthist/(100.*10*4*(1+2+3+4)),label='field %')
    meanfieldchange = np.zeros((nK,nadd))
    pylab.xlim(-70,70)
    pylab.xticks(range(-70,71,35))
    pylab.xlabel('centroid shift')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(132)
    fieldshifthist,dump = pylab.histogram(fieldshift,np.arange(-R+0.5,R))
    pylab.plot(range(-R+1,R),fieldshifthist/float(np.max(fieldshifthist)),label='shift')
    acf1 = np.correlate(v1,v1,'same')
    acf2 = np.correlate(v2,v2,'same')
    acf1 /= np.max(acf1)
    acf2 /= np.max(acf2)
    pylab.plot(range(-R/2,R/2),(acf1+acf2)/2.,'k',label='ACF')
    pylab.legend(loc=2,frameon=False)
    pylab.xlim(-300,300)
    pylab.xticks(range(-300,301,150))
    pylab.xlabel('centroid shift / lag')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(133)
    meanfieldchange = np.zeros((nK,nadd))
    for j in range(nK):
        for jj in range(nadd):
            meanfieldchange[j,jj] = np.mean(nfieldchange[j][jj])
    nadd_vec = []
    nfc_vec = []
    nK_vec = []
    for j in range(nK):
        for jj in range(nadd):
            nadd_vec.extend([jj+1]*len(nfieldchange[j][jj]))
            nfc_vec.extend(nfieldchange[j][jj])
            nK_vec.extend(['$K$='+str(j+1)]*len(nfieldchange[j][jj]))
    import seaborn as sns
    sns.violinplot(nadd_vec,nfc_vec,nK_vec,inner=None,bw=.2,gridsize=100)
    pylab.legend(loc=2,frameon=False)
    pylab.xticks(range(nadd),range(1,nadd+1))
    pylab.ylabel('num of changes')
    pylab.xlabel('num of added fields')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,right=0.9,wspace=0.4,hspace=0.4,bottom=0.2)
    pylab.savefig(figpath+'fig7.svg')

#fig1_2()
#fig3()
#fig5()
#fig5()
#fig6()
#fig6b()
fig8()
#for j in range(1,11):
#    fig7(j)
#readfig7()
