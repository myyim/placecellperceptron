# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.

import numpy as np
import pylab
import matplotlib as mpl
import pickle
import itertools
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

def fig5_noncoprime():
    ax = pylab.subplot(323)
    l = [4,6]
    N = len(l)
    Sc = int(np.round(testrange(l)[0]))
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    fname = 'pcorr'+str(l[0])+str(l[1])
    if os.path.isfile(fname+'.txt'):
        with open(fname+'.txt','rb') as f:
            pall,p2,p3,p4,p5,p6 = pickle.load(f)
    else:
        pall,p2,p3,p4,p5,p6 = frac_vs_S(l,R,return6=1)
        with open(fname+'.txt','wb') as f:
            pickle.dump((pall,p2,p3,p4,p5,p6),f)
    pylab.plot(range(1,R+1),np.ones(R),'o-',ms=5,label='$K$=1')
    pylab.plot(range(2,R+1),p2,'o-',ms=5,label='$K$=2')
    pylab.plot(range(3,R+1),p3,'o-',ms=5,label='$K$=3')
    pylab.plot(range(4,R+1),p4,'o-',ms=5,label='$K$=4')
    pylab.plot(range(5,R+1),p5,'o-',ms=5,label='$K$=5')
    pylab.plot(range(6,R+1),p6,'o-',ms=5,label='$K$=6')
    #pylab.plot([1,R],[0,0],'k--',lw=1)
    #pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
    #pylab.plot([1,R],[1,1],'k--',lw=1)
    #pylab.plot([Sc]*2,[0,1],'k--',lw=1)
    pylab.plot([R]*2,[0,1],'k--',lw=1)
    pylab.text(11,0.05,'$L$')
    pylab.ylim(0,1.05)
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    pylab.ylabel('realizable fraction')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('$\lambda=\{4,6\}$')
    # D
    ax = pylab.subplot(324)
    #pylab.plot([1,R],[0,0],'k--',lw=1)
    #pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
    #pylab.plot([1,R],[1,1],'k--',lw=1)
    #pylab.plot([Sc]*2,[0,1.05],'y')
    pylab.plot([Sc]*2,[0,1],'k--',lw=1)
    pylab.plot([R]*2,[0,1],'k--',lw=1)
    pcover = []
    for x in range(1,R+1):
        pcover.append(cover(Sc,x)/float(2**x))
    for k in range(5):
        pwcon = random_weightcon(np.sum(l),R,1,k+1)
        if k == 0:
            pylab.plot(np.arange(1,R+1,1),pwcon,'o-',c='#B0C4DE',ms=5,label='random, constrained')
        else:
            pylab.plot(np.arange(1,R+1,1),pwcon,'o-',c='#B0C4DE',ms=5)
    pylab.plot(np.arange(1,R+1,1),pcover,'o-',c='c',ms=5,label='random')
    pylab.plot(range(1,R+1),pall,'ko-',ms=5)
    #pcover = []
    #for x in range(1,R+1):
    #    pcover.append(cover(np.sum(l),x)/float(2**x))
    #pylab.plot(np.arange(1,R+1,1),pcover,'o-',c='0.8',ms=6,label='random ($N$)')
    #pylab.text(6,0.05,'$R_{int}$',color='y')
    pylab.text(8.2,0.05,'$l^*$')
    pylab.text(11,0.05,'$L$')
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    pylab.ylim(0,1.05)
    #pylab.ylabel('Fraction')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def fig8_l23():
    is_qp = 0   # use quadratic programming with the options of constrained weights
    #is_randphase = 1
    #rate_rescaled = 1
    if is_qp:
        print 'Using quadratic programming'
    else:
        print 'Using sklearn SVM'
    #if is_randphase:
    #    print 'Phases in panel D are random'
    #else:
    #   print 'Phases in panel D are equally spaced'
    mpl.rcParams['legend.fontsize'] = font_size-6
    rng = np.random.RandomState(4)
    import seaborn as sns
    gridmodel = 'del'
    mth = np.arange(0,1.0001,0.01)
    sym = ['^','+','x']
    l = np.array([2,3])
    N = len(l)
    Ng = np.sum(l)
    num = 10
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    K = int(np.ceil(R/2.))
    u = act_mat_grid_binary(l)
    u /= 2
    # A
    ax = fig.add_subplot(433)
    if is_qp:
        marr0,darr0,karr0 = input_margin_qp(u)
    else:
        marr0,darr0,karr0 = input_margin(u)
    realizable0 = (np.abs(darr0)<1e-10)
    karr0un = karr0[~realizable0]
    karr0 = karr0[realizable0]
    actn = np.copy(marr0)
    marr0 = marr0[realizable0]
    count0 = []
    for m in mth:
        count0.append(np.sum(marr0>=m))
    munique = []
    mlabel = []
    mtemp = np.around(marr0*1000)/1000.
    for k in range(1,K+1):
        munique.append(list(np.unique(mtemp[karr0==k])))
        for m in mtemp[karr0==k]:
            mlabel.append(int(str(k)+str(int(m))))
    mlabel = np.array(mlabel)
    for k in range(1,K+1):
        for mu in munique[k-1]:
            pylab.plot(k+np.array([-0.2,0.2]),2*[mu],'k')
    #ax.plot(karr0,marr0,'ko',lw=0)
    ax.set_xticks(range(1,4))
    ax.set_yticks(np.arange(0,0.9,0.2))
    ax.set_xlim(0.5,3.5)
    ax.set_ylim(0,0.9)
    ax.set_xlabel('number of fields $K$')
    ax.set_ylabel('margin $\kappa$')
    ax.set_title('$\lambda=\{2,3\}$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # B
    ax = fig.add_subplot(323)
    ax.plot([-1],[0],'k',label='grid')
    margin,rmargin,smargin,numKarr,rnumKarr,snumKarr = margin_gridvsrandom(l=[2,3],K=3,num=num,mode='ext')
    temp = []
    for j in range(K):
        temp.extend(rmargin[j][0])
    temp = np.array(temp)
    countr1 = []
    for m in mth:
        countr1.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(rmargin[j][1])
    temp = np.array(temp)
    countr2 = []
    for m in mth:
        countr2.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(smargin[j][0])
    temp = np.array(temp)
    counts1 = []
    for m in mth:
        counts1.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(smargin[j][1])
    temp = np.array(temp)
    counts2 = []
    for m in mth:
        counts2.append(np.sum(temp>=m))
    kmat = []
    for k in range(3):
        kmat.extend([k+1]*np.sum(rnumKarr[k]))
    mmat = np.sum(rmargin)
    nmat = ['random']*np.sum(np.sum(rnumKarr))
    for k in range(3):
        kmat.extend([k+1]*np.sum(snumKarr[k]))
    mmat.extend(np.sum(smargin))
    nmat.extend(['shuffled']*np.sum(np.sum(snumKarr)))
    """
    """
    kmat = []
    mmat = []
    nmat = []
    for j in range(num):
        # random
        v = rng.rand(u.shape[0],R)
        for jj in range(R):
            v[:,jj] = v[:,jj]/np.sum(v[:,jj])
        if is_qp:
            marr,darr,karr = input_margin_qp(v)   # N
        else:
            marr,darr,karr = input_margin(v)
        realizable = (np.abs(darr)<1e-10)
        kmat.extend(karr[realizable])
        mmat.extend(marr[realizable])
        nmat.extend(np.sum([realizable])*['random'])
        if j == 0:
            countr1 = []
            for m in mth:
                countr1.append(np.sum(marr[realizable]>=m))
        elif j == 1:
            countr2 = []
            for m in mth:
                countr2.append(np.sum(marr[realizable]>=m))
        # shuffled
        v = np.copy(u)
        v = v.ravel()
        rng.shuffle(v)
        v = v.reshape(u.shape)
        if is_qp:
            marrs,darrs,karrs = input_margin_qp(v)
        else:
            marrs,darrs,karrs = input_margin(v)
        realizables = (np.abs(darrs)<1e-10)
        kmat.extend(karrs[realizables])
        mmat.extend(marrs[realizables])
        nmat.extend(np.sum([realizables])*['shuffled'])
        if j == 0:
            counts1 = []
            for m in mth:
                counts1.append(np.sum(marrs[realizables]>=m))
        elif j == 1:
            counts2 = []
            for m in mth:
                counts2.append(np.sum(marrs[realizables]>=m))
    marr_rand = np.copy(marr[realizable])
    sns.violinplot(kmat,mmat,nmat,inner="point",linewidth=.4,bw=.2)
    for k in range(1,4):
        for mu in margin[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    ax.set_yticks(np.arange(0,1.1,0.5))
    ax.set_xlim(-0.5,K-0.5)
    ax.set_ylim(0,1.5)
    ax.legend(loc=2,frameon=False)
    #ax.set_xlabel('number of fields $K$')
    ax.set_ylabel('margin $\kappa$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def fig8_l23_addnoise():
    l = [2,3]
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    K = 3
    u = act_mat_grid_binary(l)
    u /= 2
    s = [0.1,0.2]
    kmat1 = []  # realizable in binary grid
    mmat1 = []
    nmat1 = []
    kmat2 = []  # newly realizable
    mmat2 = []
    nmat2 = []
    #kmat3 = []  # become unrealizable
    #mmat3 = []
    #nmat3 = []
    for j in range(num):
        #randinp = rng.randn(Ng,R)
        randinp = rng.rand(Ng,R)*np.sqrt(12)
        if is_qp:
            marr1,darr1,karr1 = input_margin_qp(u+s[0]*randinp)
            marr2,darr2,karr2 = input_margin_qp(u+s[1]*randinp)
        else:
            marr1,darr1,karr1 = input_margin(u+s[0]*randinp)
            marr2,darr2,karr2 = input_margin(u+s[1]*randinp)
        realizable1 = (np.abs(darr1)<1e-10)
        realizable2 = (np.abs(darr2)<1e-10)
        print sum(realizable1*realizable0),sum(realizable2*realizable0)
        act = pylab.vstack([marr0,marr1[realizable0],marr2[realizable0]])
        kmat1.extend(karr1[realizable0])
        mmat1.extend(marr1[realizable0])
        nmat1.extend(np.sum([realizable0])*[0.1])
        kmat2.extend(karr1[realizable1*~realizable0])
        mmat2.extend(marr1[realizable1*~realizable0])
        nmat2.extend(np.sum([realizable1*~realizable0])*[0.1])
        kmat1.extend(karr2[realizable0])
        mmat1.extend(marr2[realizable0])
        nmat1.extend(np.sum([realizable0])*[0.2])
        kmat2.extend(karr2[realizable2*~realizable0])
        mmat2.extend(marr2[realizable2*~realizable0])
        nmat2.extend(np.sum([realizable2*~realizable0])*[0.2])
        # for constrained weight
        #kmat3.extend(karr1[~realizable1*realizable0])
        #mmat3.extend(marr1[~realizable1*realizable0])
        #nmat3.extend(np.sum([~realizable1*realizable0])*[0.1])
        #kmat3.extend(karr2[~realizable2*realizable0])
        #mmat3.extend(marr2[~realizable2*realizable0])
        #nmat3.extend(np.sum([~realizable2*realizable0])*[0.2])
        print '%%%%%',np.sum(realizable0),np.sum(realizable1*realizable0),np.sum(realizable2*realizable0)
    kmat1 = np.array(kmat1)
    mmat1 = np.array(mmat1)
    nmat1 = np.array(nmat1)
    kmat2 = np.array(kmat2)
    mmat2 = np.array(mmat2)
    nmat2 = np.array(nmat2)
    ax = fig.add_subplot(3,6,16)
    sns.violinplot(x=nmat1[np.tile((mlabel==10),2*num)*(mmat1!=np.inf)],y=mmat1[np.tile((mlabel==10),2*num)*(mmat1!=np.inf)],bw=.2,color=color4[0])
    pylab.plot(np.array([-0.3,1.3]),2*[munique[0][0]],'k')
    ax.set_yticks(np.arange(0,1.9,0.5))
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(0,2.8)
    #ax.set_xlabel('number of fields $K$')
    #ax.set_ylabel('margin $\kappa$')
    ax.set_title('$K$=1')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = fig.add_subplot(3,6,17)
    sns.violinplot(x=nmat1[np.tile(mlabel==20,2*num)*(mmat1!=np.inf)],y=mmat1[np.tile(mlabel==20,2*num)*(mmat1!=np.inf)],bw=.2,color=color4[0])
    sns.violinplot(x=nmat1[np.tile(mlabel==21,2*num)*(mmat1!=np.inf)],y=mmat1[np.tile(mlabel==21,2*num)*(mmat1!=np.inf)],bw=.2,color=color4[1])
    sns.violinplot(x=nmat2[kmat2==2],y=mmat2[kmat2==2],bw=.2,color='m')
    pylab.plot(np.array([-0.3,1.3]),2*[munique[1][0]],'k')
    pylab.plot(np.array([-0.3,1.3]),2*[munique[1][1]],'k')
    ax.set_yticks([])
    #ax.set_xticks([0,1],('0.1','0.2'))
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(0,2.8)
    ax.set_xlabel('standard deviation $\sigma$')
    ax.set_title('$K$=2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ###
    #sns.violinplot(x=kmat1[np.tile((mlabel==10)+(mlabel==20)+(mlabel==30),2*num)],y=mmat1[np.tile((mlabel==10)+(mlabel==20)+(mlabel==30),2*num)],hue=nmat1[np.tile((mlabel==10)+(mlabel==20)+(mlabel==30),2*num)],bw=.2,palette='Set2')
    #sns.violinplot(x=np.append([1],kmat1[np.tile((mlabel==21)+(mlabel==31),2*num)]),y=np.append([-1],mmat1[np.tile((mlabel==21)+(mlabel==31),2*num)]),hue=np.append(['$\sigma$=0.1'],nmat1[np.tile((mlabel==21)+(mlabel==31),2*num)]),bw=.2,palette='Set2')
    #sns.violinplot(x=np.append([1],kmat2),y=np.append([-1],mmat2),hue=np.append(['$\sigma$=0.1'],nmat2),bw=.2,palette='Set2')
    #ax.legend(loc=2,frameon=False)
    #for k in range(1,4):
        #for mu in munique[k-1]:
            #pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    ax = fig.add_subplot(3,6,18)
    #sns.violinplot(x=nmat1[np.tile((mlabel==30)+(mlabel==31),2*num)],y=mmat1[np.tile((mlabel==30)+(mlabel==31),2*num)],hue=kmat1[np.tile((mlabel==30)+(mlabel==31),2*num)],bw=.2,palette='Set2')
    sns.violinplot(x=nmat1[np.tile(mlabel==30,2*num)*(mmat1!=np.inf)],y=mmat1[np.tile(mlabel==30,2*num)*(mmat1!=np.inf)],bw=.2,color=color4[0])
    sns.violinplot(x=nmat1[np.tile(mlabel==31,2*num)*(mmat1!=np.inf)],y=mmat1[np.tile(mlabel==31,2*num)*(mmat1!=np.inf)],bw=.2,color=color4[1])
    sns.violinplot(x=nmat2[kmat2==3],y=mmat2[kmat2==3],bw=.2,color='m')
    pylab.plot(np.array([-0.3,1.3]),2*[munique[2][0]],'k')
    pylab.plot(np.array([-0.3,1.3]),2*[munique[2][1]],'k')
    ax.set_yticks([])
    #ax.set_xticks([0,1],('0.1','0.2'))
    ax.set_xlim(-0.5,1.5)
    ax.set_ylim(0,2.8)
    ax.set_title('$K$=3')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # obsolete below
    #ax = fig.add_subplot(326)
    #ax = fig.add_axes([0.41, 0.1, 0.55, 0.23])
    
def fig8_l23_gaussprofile():
    sig = 0.212
    gridmodel = 'gau'
    color = ['r']
    narr = [5,10,20]
    mthn = np.arange(0,2.3,0.01)
    pylab.plot([-1],[0],'k',label='binary')
    kmat = []
    mmat = []
    nmat = []
    for k in range(num):
        for j in range(len(narr)):
            v = []
            if j == 0:
                for iN in range(N):
                    for iM in range(l[iN]):
                        if is_randphase:
                            v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
                        else:
                            v.append(grid(np.arange(R),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel,sig=sig))
            else:
                for iN in range(N):
                    for iM in range(narr[j]/2):
                        if is_randphase:
                            v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
                        else:
                            v.append(grid(np.arange(R),l[iN],float(iM)/(narr[j]/2.),phsh=0.,gridmodel=gridmodel,sig=sig))
            if rate_rescaled:
                v = np.array(v)/(narr[j]/5.)
            else:
                v = np.array(v)
            if is_qp:
                marr,darr,karr = input_margin_qp(v)
            else:
                marr,darr,karr = input_margin(v)
            realizable = (np.abs(darr)<1e-10)
            print '@@@@@',k,j,np.sum(realizable)
            kmat.extend(karr[realizable0])
            mmat.extend(marr[realizable0])
            nmat.extend(np.sum([realizable0])*[narr[j]])
            if k == num-1:
                v1 = np.copy(v[:,:-1])
                if j == 0:
                    v1[:2,2:] = 0
                    v1[2:,:2] = 0
                else:
                    v1[:narr[j]/2,2:] = 0
                    v1[narr[j]/2:,:2] = 0
                print narr[j],v1
                actn = pylab.vstack([actn,marr])
                if j == 0:
                    countn1 = []
                    for m in mthn:
                        countn1.append(np.sum(marr[realizable]>=m))
                elif j == 1:
                    countn2 = []
                    for m in mthn:
                        countn2.append(np.sum(marr[realizable]>=m))
                elif j == 2:
                    countn3 = []
                    for m in mthn:
                        countn3.append(np.sum(marr[realizable]>=m))
    kmat = np.array(kmat)
    mmat = np.array(mmat)
    nmat = np.array(nmat)
    ax = fig.add_axes([0.43, 0.1, 0.15, 0.23])
    sns.violinplot(x=nmat[np.tile((mlabel==10),len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile((mlabel==10),len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[2])
    pylab.plot(np.array([-0.3,2.3]),2*[munique[0][0]],'k')
    ax.set_yticks(np.arange(0,3.1,1))
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(0,4)
    ax.set_ylabel('margin $\kappa$')
    ax.set_title('$K$=1')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = fig.add_axes([0.61, 0.1, 0.15, 0.23])
    sns.violinplot(x=nmat[np.tile(mlabel==20,len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile(mlabel==20,len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[2])
    sns.violinplot(x=nmat[np.tile(mlabel==21,len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile(mlabel==21,len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[3])
    pylab.plot(np.array([-0.3,2.3]),2*[munique[1][0]],'k')
    pylab.plot(np.array([-0.3,2.3]),2*[munique[1][1]],'k')
    ax.set_yticks([])
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(0,4)
    ax.set_xlabel('number of grid cells $N$')
    ax.set_title('$K$=2')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = fig.add_axes([0.79, 0.1, 0.15, 0.23])
    sns.violinplot(x=nmat[np.tile(mlabel==30,len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile(mlabel==30,len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[2])
    sns.violinplot(x=nmat[np.tile(mlabel==31,len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile(mlabel==31,len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[3])
    pylab.plot(np.array([-0.3,2.3]),2*[munique[1][0]],'k')
    pylab.plot(np.array([-0.3,2.3]),2*[munique[1][1]],'k')
    ax.set_yticks([])
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(0,4)
    ax.set_title('$K$=3')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.1,wspace=0.4,hspace=0.5)
    fig.savefig('f8'+'qp'*is_qp+'.svg')

def fig8_l23_inset():
    # Inset
    print 'Inset'
    inset = pylab.figure(figsize=[7,5])
    ax = inset.add_subplot(221)
    kvsN = 0
    if kvsN:
        print 'Please update or remove'
        ax.plot(count0,mth,'k')
        ax.plot(countr1,mth,color='#1f77b4')
        ax.plot(countr2,mth,'--',color='#1f77b4')
        ax.plot(counts1,mth,color='#ff7f0e')
        ax.plot(counts2,mth,'--',color='#ff7f0e')
        ax.plot(2*[41],[0,1.5],'k--',lw=1)
        ax.set_xlim(0,41+1)
        ax.set_ylim(0,1.7)
        ax.set_yticks(np.arange(0,1.6,0.5))
        ax.set_xlabel('cumulative number')
        ax.set_ylabel('margin $\kappa$')
    else:
        ax.plot(mth,np.array(count0)/41.,'k')
        ax.plot(mth,np.array(countr1)/41.,color='#1f77b4')
        ax.plot(mth,np.array(counts1)/41.,color='#ff7f0e')
        if num > 1:
            ax.plot(mth,np.array(countr2)/41.,'--',color='#1f77b4')
            ax.plot(mth,np.array(counts2)/41.,'--',color='#ff7f0e')
        ax.plot([0,1],2*[1],'k--',lw=1)
        ax.set_ylim(0,1)
        ax.set_xlim(0,1)
        ax.set_xticks(np.arange(0,1.4,0.5))
        ax.set_xlabel('margin $\kappa$')
        ax.set_ylabel('cumulative fraction')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #mpl.rcParams['xtick.labelsize'] = font_size-5
    # remove below
    ax = inset.add_subplot(222)
    l = [31,43]
    R = l[0]*l[1]
    K = 6
    #temp = 0.
    #for j in range(K):
    #    temp += nCr(l[0]*l[1],j+1)
    #ax.plot(mth,np.array(count0L)/temp,'k')
    m_inset = []
    for k in range(1,K+1):
        count,temp = np.histogram(rng.rand(int(mode[1:])),np.array([0.]+list(np.cumsum(numKarr[k-1]))+[nCr(R,k)])/nCr(R,k))
        for j in range(len(count)-1):
            m_inset.extend([margin[k-1][j]]*count[j])
        print np.array([0.]+list(np.cumsum(numKarr[k-1]))+[nCr(R,k)]),count,m_inset
    count0L = []
    for m in mth:
        count0L.append(np.sum(np.array(m_inset)>=m))
    temp = float(K)*int(mode[1:])
    ax.plot(mth,np.array(count0L)/temp,'k')
    ax.plot(mth,np.array(countr1L)/temp,color='#1f77b4')
    ax.plot(mth,np.array(counts1L)/temp,color='#ff7f0e')
    if num > 1:
        ax.plot(mth,np.array(countr2L)/temp,'--',color='#1f77b4')
        ax.plot(mth,np.array(counts2L)/temp,'--',color='#ff7f0e')
    ax.plot([0,1],2*[1],'k--',lw=1)
    ax.set_ylim(0,1)
    ax.set_xlim(0,0.8)
    ax.set_xticks(np.arange(0,1,0.2))
    ax.set_xlabel('margin $\kappa$')
    ax.set_ylabel('cumulative fraction')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    """
    """
    ax = inset.add_subplot(222)
    #ax.plot([-1],[-1],'ko',lw=0,label='no noise')
    #ax.plot([-1],[-1],'k^',lw=0,label='with noise')
    for k in range(np.sum(realizable0)):
        pylab.plot(karr0[k]+[0,0.3,0.6],act[:,k],'k',lw=1)
    ax.plot(karr0,marr0,'ko',lw=0)
    ax.plot(karr1[realizable0][(mlabel==10)+(mlabel==20)+(mlabel==30)]+0.3,marr1[realizable0][(mlabel==10)+(mlabel==20)+(mlabel==30)],'o',c=color4[0],lw=0)
    ax.plot(karr1[realizable0][(mlabel==21)+(mlabel==31)]+0.3,marr1[realizable0][(mlabel==21)+(mlabel==31)],'o',c=color4[1],lw=0)
    ax.plot(karr2[realizable0][(mlabel==10)+(mlabel==20)+(mlabel==30)]+0.6,marr2[realizable0][(mlabel==10)+(mlabel==20)+(mlabel==30)],'o',c=color4[0],lw=0)
    ax.plot(karr2[realizable0][(mlabel==21)+(mlabel==31)]+0.6,marr2[realizable0][(mlabel==21)+(mlabel==31)],'o',c=color4[1],lw=0)
    act = pylab.vstack([marr1[~realizable0],marr2[~realizable0]])
    for k in range(np.sum(~realizable0)):
        if np.sum(act[:,k]==np.inf) == 0:
            pylab.plot(karr0un[k]+[0.3,0.6],act[:,k],color='m')
    ax.plot(karr1[realizable1*~realizable0]+0.3,marr1[realizable1*~realizable0],color='m',marker='^',lw=0)
    ax.plot(karr2[realizable2*~realizable0]+0.6,marr2[realizable2*~realizable0],color='m',marker='^',lw=0) #,label='realizable with noise')
    pylab.xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],('0',str(s[0]),str(s[1]),'0',str(s[0]),str(s[1]),'0',str(s[0]),str(s[1])))
    ax.set_yticks(np.arange(0,1.6,0.5))
    ax.set_xlim(0.5,4)
    ax.set_ylim(0,1.8)
    #ax.legend(loc=3,frameon=False)
    ax.set_ylabel('margin $\kappa$')
    ax.set_xlabel('noise strength $\sigma$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mpl.rcParams['xtick.labelsize'] = font_size-2
    ax = inset.add_subplot(223)
    if kvsN:
        ax.plot(np.append(count0,np.zeros(len(mthn)-len(mth))),mthn,'k',label='binary')
        ax.plot(countn1,mthn,color='#1f77b4',label='$N$=5')
        ax.plot(countn2,mthn,color='#ff7f0e',label='$N$=10')
        ax.plot(countn3,mthn,color='#2ca02c',label='$N$=20')
        ax.plot(2*[41],[0,mthn[-1]],'k--',lw=1)
        ax.set_xlim(0,41+1)
        ax.set_ylim(0,mthn[-1]+0.1)
        ax.set_yticks(np.arange(0,mthn[-1],0.5))
        ax.set_xlabel('cumulative number')
        ax.set_ylabel('margin $\kappa$')
    else:
        ax.plot(mthn,np.append(count0,np.zeros(len(mthn)-len(mth))),'k',label='binary')
        ax.plot(mthn,countn1,color='#1f77b4',label='$N$=5')
        ax.plot(mthn,countn2,color='#ff7f0e',label='$N$=10')
        ax.plot(mthn,countn3,color='#2ca02c',label='$N$=20')
        ax.plot([0,mthn[-1]],2*[41],'k--',lw=1)
        ax.set_ylim(0,41+1)
        ax.set_xlim(0,mthn[-1]+0.1)
        ax.set_xticks(np.arange(0,mthn[-1],0.5))
        ax.set_xlabel('margin $\kappa$')
        ax.set_ylabel('cumulative number')
    ax.legend(loc=1,frameon=False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    mpl.rcParams['xtick.labelsize'] = font_size-5
    ax = inset.add_subplot(224)
    actn = actn[:,realizable0]
    for k in range(np.sum(realizable0)):
        if np.sum(actn[1:,k]==np.inf) == 0:
            pylab.plot(karr0[k]+[0,0.3,0.6],actn[1:,k],'k',lw=1)
    pylab.plot(karr0[((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[1,:]!=np.inf)],actn[1,((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[1,:]!=np.inf)],'s',c=color4[2],lw=0)
    pylab.plot(karr0[((mlabel==21)+(mlabel==31))*(actn[1,:]!=np.inf)],actn[1,((mlabel==21)+(mlabel==31))*(actn[1,:]!=np.inf)],'s',c=color4[3],lw=0)

    pylab.plot(karr0[((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[2,:]!=np.inf)]+0.3,actn[2,((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[2,:]!=np.inf)],'s',c=color4[2],lw=0)
    pylab.plot(karr0[((mlabel==21)+(mlabel==31))*(actn[2,:]!=np.inf)]+0.3,actn[2,((mlabel==21)+(mlabel==31))*(actn[2,:]!=np.inf)],'s',c=color4[3],lw=0)

    pylab.plot(karr0[((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[3,:]!=np.inf)]+0.6,actn[3,((mlabel==10)+(mlabel==20)+(mlabel==30))*(actn[3,:]!=np.inf)],'s',c=color4[2],lw=0)
    pylab.plot(karr0[((mlabel==21)+(mlabel==31))*(actn[3,:]!=np.inf)]+0.6,actn[3,((mlabel==21)+(mlabel==31))*(actn[3,:]!=np.inf)],'s',c=color4[3],lw=0)
    pylab.xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],(str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2])))
    ax.set_yticks(np.arange(0,3.2,1))
    ax.set_xlim(0.5,4)
    ax.set_ylim(0,mthn[-1])
    ax.legend(loc=3,frameon=False)
    ax.set_xlabel('number of grid cells $N$')
    ax.set_ylabel('margin $\kappa$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #inset.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.1,wspace=0.3,hspace=0.3)
    #inset.savefig('f4b'+'qp'*is_qp+'.svg')

def extra_fig8():
    rng = np.random.RandomState(4)
    import seaborn as sns
    gridmodel = 'del'
    mth = np.arange(0,1.6,0.01)
    sym = ['^','+','x']
    l = np.array([2,3])
    N = len(l)
    Ng = np.sum(l)
    R = l[0]
    num = 100
    for j in range(N-1):
        R = lcm(R,l[j+1])
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(np.max(R)),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel))
    u = np.array(u)
    ### grid vs random
    marr0,darr0,karr0 = input_margin(u)
    realizable0 = (np.abs(darr0)<1e-10)
    karr0un = karr0[~realizable0]
    karr0 = karr0[realizable0]
    actn = np.copy(marr0)
    marr0 = marr0[realizable0]
    count0 = []
    for m in mth:
        count0.append(np.sum(marr0>=m))
    munique = []
    mlabel = []
    mtemp = np.around(marr0*1000)/1000.
    for k in [1,2,3]:
        munique.append(list(np.unique(mtemp[karr0==k])))
        for m in mtemp[karr0==k]:
            mlabel.append(int(str(k)+str(int(m))))
    mlabel = np.array(mlabel)
    kmat = []
    mmat = []
    nmat = []
    for j in range(num):
        randinp = rng.rand(u.shape[0],R)
        marr,darr,karr = input_margin(randinp)
        realizable = (np.abs(darr)<1e-10)
        kmat.extend(karr[realizable])
        mmat.extend(marr[realizable])
        nmat.extend(np.sum([realizable])*['random - same range'])
        if j == 0:
            countr1 = []
            for m in mth:
                countr1.append(np.sum(marr[realizable]>=m))
        elif j == 1:
            countr2 = []
            for m in mth:
                countr2.append(np.sum(marr[realizable]>=m))
        # std
        marrs,darrs,karrs = input_margin(np.std(u)*randinp/np.std(randinp))
        realizables = (np.abs(darrs)<1e-10)
        kmat.extend(karrs[realizables])
        mmat.extend(marrs[realizables])
        nmat.extend(np.sum([realizables])*['random - same std'])
        if j == 0:
            counts1 = []
            for m in mth:
                counts1.append(np.sum(marrs[realizables]>=m))
        elif j == 1:
            counts2 = []
            for m in mth:
                counts2.append(np.sum(marrs[realizables]>=m))
    marr_rand = np.copy(marr[realizable])
    """
    fig = pylab.figure()
    ax = pylab.subplot(111)
    sns.violinplot(kmat,mmat,nmat,bw=.2)
    for k in range(1,4):
        for mu in munique[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    ax.set_yticks(np.arange(0,1.9,0.5))
    ax.set_xlim(-0.5,2.5)
    ax.set_ylim(0,2.8)
    ax.legend(loc=2,frameon=False)
    ax.set_xlabel('number of fields $K$')
    ax.set_ylabel('margin $\kappa$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    """
    ### Add noise after training
    K = int(np.ceil(R/2.))
    sarr = np.array([0]) #np.array([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
    ml = np.copy(mlabel)
    ml[ml==10] = 0
    ml[ml==20] = 1
    ml[ml==21] = 2
    ml[ml==30] = 3
    ml[ml==31] = 4
    count29 = np.zeros(((5,sarr.size,num)))
    count = 0
    #randnoise = array([[0.48966653,0.0834978,0.66300532,-0.37685467,0.39140418,-1.60732775]])
    randnoise = rng.randn(num,u.shape[1])
    temparr = []
    for k in range(1,4):
        com = [list(temp) for temp in itertools.combinations(range(R),k)]
        for j in range(len(com)):
            Y = np.zeros(R)
            Y[com[j]] = 1
            m,w,b = svm_margin(u.T,Y)
            dec = np.sign(np.dot(w.T,u)+b)
            dec[dec<0] = 0
            if abs(np.sum(np.abs(Y-dec)))<1e-10:
                for t in range(num):
                    for s in range(sarr.size):
                        dec = np.sign(np.dot(w.T,u)+b+sarr[s]*randnoise[t])
                        dec[dec<0] = 0
                        print t,com[j],abs(np.sum(np.abs(Y-dec))),m,ml[count],np.dot(w.T,u)+b,w,b
                        temparr.extend(np.dot(w.T,u)+b)
                        if 0: #abs(np.sum(np.abs(Y-dec))) > 0:
                            print t,com[j],abs(np.sum(np.abs(Y-dec))),m,ml[count]
                            print np.dot(w.T,u)+b,np.sign(np.dot(w.T,u)+b)
                            print np.dot(w.T,u)+b+sarr[s]*randnoise[t],np.sign(np.dot(w.T,u)+b+sarr[s]*randnoise[t])
                        if abs(np.sum(np.abs(Y-dec)))<1e-10:
                            count29[ml[count],s,t] += 1
                count += 1
    fig = pylab.figure(figsize=[8,4])
    ax = pylab.subplot(121)
    pylab.plot([0],[29],'ko')
    for t in range(num):
        pylab.plot(sarr,np.sum(count29[:,:,t],0),'ko')
    pylab.xlabel('STD of GWN')
    pylab.ylabel('number of realizable arrangements')
    ax = pylab.subplot(122)
    count29 = np.sum(count29,2)
    mflat = [item for sublist in munique for item in sublist]
    for j in range(5):
        count29[j,:] = count29[j,:]/(np.sum(ml==j)*num)
        pylab.plot(sarr,count29[j,:],'o-',label='$K$={:d}; $\kappa$={:.1f}'.format((j+3)/2,mflat[j]))
    pylab.legend(loc=1)
    pylab.xlabel('STD of GWN')
    pylab.ylabel('fraction robust against noise')
    # kernel width
    sigarr = [0.106,0.16,0.212]
    gridmodel = 'gau'
    color = ['r']
    narr = [5]
    mthn = np.arange(0,2.3,0.01)
    pylab.plot([-1],[0],'k',label='binary')
    kmat = []
    mmat = []
    nmat = []
    for s in range(len(sigarr)):
        for j in range(len(narr)):
            v = []
            if j == 0:
                for iN in range(N):
                    for iM in range(l[iN]):
                        if is_randphase:
                            v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sigarr[s]))
                        else:
                            v.append(grid(np.arange(R),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel,sig=sigarr[s]))
            else:
                for iN in range(N):
                    for iM in range(narr[j]/2):
                        if is_randphase:
                            v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sigarr[s]))
                        else:
                            v.append(grid(np.arange(R),l[iN],float(iM)/(narr[j]/2.),phsh=0.,gridmodel=gridmodel,sig=sigarr[s]))
            v = np.array(v)
            marr,darr,karr = input_margin(v)
            realizable = (np.abs(darr)<1e-10)
            print '@@@@@',k,j,np.sum(realizable)
            kmat.extend(karr[realizable0])
            mmat.extend(marr[realizable0])
            nmat.extend(np.sum([realizable0])*[narr[j]])
            if k == num-1:
                v1 = np.copy(v[:,:-1])
                if j == 0:
                    v1[:2,2:] = 0
                    v1[2:,:2] = 0
                else:
                    v1[:narr[j]/2,2:] = 0
                    v1[narr[j]/2:,:2] = 0
                print narr[j],v1
                actn = pylab.vstack([actn,marr])
                if j == 0:
                    countn1 = []
                    for m in mthn:
                        countn1.append(np.sum(marr[realizable]>=m))
                elif j == 1:
                    countn2 = []
                    for m in mthn:
                        countn2.append(np.sum(marr[realizable]>=m))
                elif j == 2:
                    countn3 = []
                    for m in mthn:
                        countn3.append(np.sum(marr[realizable]>=m))
        kmat = np.array(kmat)
        mmat = np.array(mmat)
        nmat = np.array(nmat)
        ax = fig.add_axes([0.43, 0.1, 0.15, 0.23])
        sns.violinplot(x=nmat[np.tile((mlabel==10),len(narr)*num)*(mmat!=np.inf)],y=mmat[np.tile((mlabel==10),len(narr)*num)*(mmat!=np.inf)],bw=.2,color=color4[2])
        pylab.plot(np.array([-0.3,2.3]),2*[munique[0][0]],'k')
        ax.set_yticks(np.arange(0,3.1,1))
        ax.set_xlim(-0.5,2.5)
        ax.set_ylim(0,4)
        ax.set_ylabel('margin $\kappa$')
        ax.set_title('$K$=1')

def morefig8b():
    fig = pylab.figure(figsize=[7,5])
    l = [2,3]
    N = len(l)
    Sc = int(np.round(testrange(l)[0]))
    mth = np.arange(0,1.6,0.01)
    gridmodel = 'del'
    sym = ['^','+','x']
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(np.max(R)),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel))
    u = np.array(u)
    # Grid
    if is_qp:
        marr0,darr0,karr0 = input_margin_qp(u)
    else:
        marr0,darr0,karr0 = input_margin(u)
    realizable0 = (np.abs(darr0)<1e-10)
    karr0 = karr0[realizable0]
    marr0 = marr0[realizable0]
    count0 = []
    kall = []
    mall = []
    lall = []
    for m in mth:
        count0.append(np.sum(marr0>=m))
    munique = []
    mtemp = np.around(marr0*1000)/1000.
    for k in [1,2,3]:
        munique.append(list(np.unique(mtemp[karr0==k])))
    for j in range(5):
        rng = np.random.RandomState(j+1)
        ax = pylab.subplot(2,3,j+1)
        ax.plot(mth,count0,'k',label='grid')
        # Random
        v = rng.rand(Sc,R)
        marr,darr,karr = input_margin(v)
        realizable = (np.abs(darr)<1e-10)
        print np.sum(realizable),np.linalg.matrix_rank(v)
        karr = karr[realizable]
        marr = marr[realizable]
        kall.extend(karr)
        mall.extend(marr)
        lall.extend(np.sum(realizable)*['rand (rk)'])
        count = []
        for m in mth:
            count.append(np.sum(marr>=m))
        ax.plot(mth,count,color=color4[2],label='rand (rk)')
        v = rng.rand(u.shape[0],R)
        marr,darr,karr = input_margin(v)
        realizable = (np.abs(darr)<1e-10)
        print np.sum(realizable),np.linalg.matrix_rank(v)
        karr = karr[realizable]
        marr = marr[realizable]
        kall.extend(karr)
        mall.extend(marr)
        lall.extend(np.sum(realizable)*['random ($N$)'])
        count = []
        for m in mth:
            count.append(np.sum(marr>=m))
        ax.plot(mth,count,color=color4[0],label='rand ($N$)')
        # Shuffled
        v = np.copy(u)
        v = v.ravel()
        rng.shuffle(v)
        v = v.reshape(u.shape)
        marr2,darr2,karr2 = input_margin(v)
        realizable2 = (np.abs(darr2)<1e-10)
        print np.sum(realizable2),np.linalg.matrix_rank(v)
        karr2 = karr2[realizable2]
        marr2 = marr2[realizable2]
        kall.extend(karr2)
        mall.extend(marr2)
        lall.extend(np.sum(realizable2)*['shuffled'])
        count = []
        for m in mth:
            count.append(np.sum(marr2>=m))
        ax.plot(mth,count,color=color4[1],label='shuffled')
        if j == 0:
            ax.legend(loc=1)
            #ax.legend(['grid','random ($N$)','random (rank)','shuffled'],loc=1)
        ax.plot([0,1.5],2*[41],'k--',lw=1)
        ax.set_ylim(0,41+1)
        ax.set_xlim(0,1.7)
        ax.set_xlabel('maximum margin')
        ax.set_ylabel('cumulative number')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    ax = pylab.subplot(236)
    import seaborn as sns
    sns.violinplot(x=kall,y=mall,hue=lall,bw=.2)
    for k in range(1,4):
        for mu in munique[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    pylab.yticks(np.arange(0,1.9,0.5))
    pylab.xlim(-0.5,2.5)
    pylab.ylim(0,2.5)
    pylab.legend(loc=2,frameon=False)
    pylab.xlabel('number of fields $K$')
    pylab.ylabel('maximum margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.15,hspace=0.3,wspace=0.3)
    pylab.savefig('morefig8b.svg')

def morefig8c():
    mpl.rcParams['xtick.labelsize'] = font_size-6
    fig = pylab.figure(figsize=[7,10])
    l = [2,3]
    N = len(l)
    Sc = int(np.round(testrange(l)[0]))
    mth = np.arange(0,1.5,0.01)
    gridmodel = 'del'
    sym = ['^','+','x']
    s = [0.1,0.2]
    R = l[0]
    num = 10
    for j in range(N-1):
        R = lcm(R,l[j+1])
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(np.max(R)),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel))
    u = np.array(u)
    # Grid
    marr0,darr0,karr0 = input_margin(u)
    realizable0 = (np.abs(darr0)<1e-10)
    karr0un = karr0[~realizable0]
    karr0 = karr0[realizable0]
    marr0 = marr0[realizable0]
    count0 = []
    for m in mth:
        count0.append(np.sum(marr0>=m))
    munique = []
    mlabel = []
    mtemp = np.around(marr0*1000)/1000.
    for k in [1,2,3]:
        munique.append(list(np.unique(mtemp[karr0==k])))
        for m in mtemp[karr0==k]:
            mlabel.append(str(m)+';'+str(k))
    kall = []
    mall = []
    lall = []
    allmarg1 = []
    allmarg2 = []
    for j in range(num):
        rng = np.random.RandomState(j+1)
        # Random - tree
        randinp = rng.randn(u.shape[0],R)
        marr1,darr1,karr1 = input_margin(u+s[0]*randinp)
        marr2,darr2,karr2 = input_margin(u+s[1]*randinp)
        realizable1 = (np.abs(darr1)<1e-10)
        realizable2 = (np.abs(darr2)<1e-10)
        allmarg1.extend(marr1[realizable0])
        allmarg2.extend(marr2[realizable0])
        act = pylab.vstack([marr0,marr1[realizable0],marr2[realizable0]])
        if j < 3:
            ax = pylab.subplot(4,3,j+1)
            for k in range(np.sum(realizable0)):
                pylab.plot(karr0[k]+[0,0.3,0.6],act[:,k],'k')
                pylab.plot(karr0,marr0,'ko',lw=0,label='no noise')
                pylab.plot(karr1[realizable0]+0.3,marr1[realizable0],'k^',lw=0,label='with noise')
                pylab.plot(karr2[realizable0]+0.6,marr2[realizable0],'k^',lw=0)
            act = pylab.vstack([marr1[~realizable0],marr2[~realizable0]])
            for k in range(np.sum(realizable1*~realizable0)):
                pylab.plot(karr0un[k]+[0.3,0.6],act[:,k],color=color4[2])
            pylab.plot(karr1[realizable1*~realizable0]+0.3,marr1[realizable1*~realizable0],color=color4[2],marker='^',lw=0)
            pylab.plot(karr2[realizable2*~realizable0]+0.6,marr2[realizable2*~realizable0],color=color4[2],marker='^',lw=0,label='realizable with noise')
            pylab.xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],('0',str(s[0]),str(s[1]),'0',str(s[0]),str(s[1]),'0',str(s[0]),str(s[1])))
            pylab.yticks(np.arange(0,1.6,0.5))
            pylab.xlim(0.5,4)
            pylab.ylim(0,1.7)
            if j == 0:
                pylab.ylabel('maximum margin')
            pylab.xlabel('noise strength $\sigma$')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        if j < 5:
            # Random - cumulative
            ax = pylab.subplot(4,3,j+7)
            ax.plot(mth,count0,'k')
            karr1 = karr1[realizable1]
            marr1 = marr1[realizable1]
            karr2 = karr2[realizable2]
            marr2 = marr2[realizable2]
            kall.extend(karr1)
            kall.extend(karr2)
            mall.extend(marr1)
            mall.extend(marr2)
            lall.extend(np.sum(realizable1)*['$\sigma$='+str(s[0])])
            lall.extend(np.sum(realizable2)*['$\sigma$='+str(s[1])])
            count = []
            for m in mth:
                count.append(np.sum(marr1>=m))
            ax.plot(mth,count,color='b')
            count = []
            for m in mth:
                count.append(np.sum(marr2>=m))
            ax.plot(mth,count,color=color4[2])
            ax.plot([0,1.5],2*[41],'k--',lw=1)
            if j == 0:
                ax.legend(loc=1)
            ax.set_ylim(0,41+1)
            ax.set_xlim(0,1.7)
            if j > 2:
                ax.set_xlabel('maximum margin')
            if np.mod(j,3) == 0:
                ax.set_ylabel('cumulative number')
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
    import seaborn as sns
    ax = pylab.subplot(423)
    munique1 = [munique[0][0],munique[1][0],munique[1][1],munique[2][0],munique[2][1]]
    sns.violinplot(x=list(mlabel)*num,y=allmarg1,bw=.2)
    for k in range(5):
        pylab.plot(k+np.array([-0.3,0.3]),2*[munique1[k]],'k')
    pylab.yticks(np.arange(0,1.9,0.5))
    pylab.legend(loc=2,frameon=False)
    pylab.xlabel('number of fields $K$')
    pylab.ylabel('maximum margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(424)
    sns.violinplot(x=list(mlabel)*num,y=allmarg2,bw=.2)
    for k in range(5):
        pylab.plot(k+np.array([-0.3,0.3]),2*[munique1[k]],'k')
    pylab.yticks(np.arange(0,1.9,0.5))
    pylab.legend(loc=2,frameon=False)
    pylab.xlabel('number of fields $K$')
    #pylab.ylabel('maximum margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(4,3,12)
    sns.violinplot(x=kall,y=mall,hue=lall,bw=.2)
    for k in range(1,4):
        for mu in munique[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    pylab.yticks(np.arange(0,1.9,0.5))
    pylab.xlim(-0.5,2.5)
    pylab.ylim(0,2)
    pylab.legend(loc=2,frameon=False)
    pylab.xlabel('number of fields $K$')
    pylab.ylabel('maximum margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.15,hspace=0.3,wspace=0.3)
    pylab.savefig('morefig8c.svg')

def morefig8d():
    mpl.rcParams['legend.fontsize'] = font_size-7
    mpl.rcParams['xtick.labelsize'] = font_size-7
    rng = np.random.RandomState(4)
    fig = pylab.figure(figsize=[7,5])
    gridmodel = 'del'
    sym = ['^','+','x']
    l = np.array([2,3])
    N = len(l)
    Ng = np.sum(l)
    mth = np.arange(0,2.5,0.01)
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(np.max(R)),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel))
    u = np.array(u)
    # Grid
    marr0,darr0,karr0 = input_margin(u)
    realizable0 = (np.abs(darr0)<1e-10)
    karr0un = karr0[~realizable0]
    karr0 = karr0[realizable0]
    marr0 = marr0[realizable0]
    act = np.copy(marr0)
    count0 = []
    for m in mth:
        count0.append(np.sum(marr0>=m))
    munique = []
    mlabel = []
    mtemp = np.around(marr0*1000)/1000.
    for k in [1,2,3]:
        munique.append(list(np.unique(mtemp[karr0==k])))
        for m in mtemp[karr0==k]:
            mlabel.append(str(m)+';'+str(k))
    sig = 0.212
    gridmodel = 'gau'
    color = ['r']
    narr = [5,10,20]
    # regularly spaced phases
    carray = np.copy(count0)
    for j in range(len(narr)):
        v = []
        if j == 0:
            for iN in range(N):
                for iM in range(l[iN]):
                    v.append(grid(np.arange(R),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel,sig=sig))
        else:
            for iN in range(N):
                for iM in range(narr[j]/2):
                    v.append(grid(np.arange(R),l[iN],float(iM)/(narr[j]/2),phsh=0.,gridmodel=gridmodel,sig=sig))
        v = np.array(v)
        marr,darr,karr = input_margin(v)
        realizable = (np.abs(darr)<1e-10)
        act = pylab.vstack([act,marr[realizable]])
        count = []
        for m in mth:
            count.append(np.sum(marr[realizable]>=m))
        carray = pylab.vstack([carray,count])
    ax = pylab.subplot(221)
    for k in range(np.sum(realizable0)):
        pylab.plot(karr0[k]+[0,0.3,0.6],act[1:,k],'g')
    pylab.plot(karr0,act[1,:],'go',lw=0,label='$N$'+str(narr[0]))
    pylab.plot(karr0+0.3,act[2,:],'g^',lw=0,label='$N$'+str(narr[1]))
    pylab.plot(karr0+0.6,act[3,:],'g^',lw=0,label='$N$'+str(narr[2]))
    #pylab.legend(loc=2,frameon=False)
    for k in range(1,4):
        for mu in munique[k-1]:
            pylab.plot(k+np.array([0,0.6]),2*[mu],'k')
    pylab.xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],(str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2])))
    pylab.yticks(np.arange(0,4,1))
    pylab.xlim(0.5,4)
    pylab.ylim(0,mth[-1]+0.1)
    pylab.ylabel('maximum margin')
    pylab.xlabel('$N$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(222)
    ax.plot(mth,carray[0,:],'k',label='binary')
    for j in range(len(narr)):
        ax.plot(mth,carray[j+1,:],color4[j],label='$N$='+str(narr[j]))
    ax.plot([0,mth[-1]],2*[41],'k--',lw=1)
    for j in np.arange(0,3.5,0.5):
        ax.plot([j,j],[0,41],'k--',lw=1)
    ax.legend(loc=1)
    ax.set_ylim(0,41+1)
    ax.set_xlim(0,mth[-1]+0.1)
    ax.set_xlabel('maximum margin')
    ax.set_ylabel('cumulative number')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # random phases
    act = np.copy(marr0)
    carray = np.copy(count0)
    for j in range(len(narr)):
        v = []
        if j == 0:
            for iN in range(N):
                for iM in range(l[iN]):
                    v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
        else:
            for iN in range(N):
                for iM in range(narr[j]/2):
                    v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
        v = np.array(v)
        marr,darr,karr = input_margin(v)
        realizable = (np.abs(darr)<1e-10)
        act = pylab.vstack([act,marr[realizable]])
        count = []
        for m in mth:
            count.append(np.sum(marr[realizable]>=m))
        carray = pylab.vstack([carray,count])
    ax = pylab.subplot(223)
    for k in range(np.sum(realizable0)):
        pylab.plot(karr0[k]+[0,0.3,0.6],act[1:,k],'g')
        pylab.plot(karr0,act[1,:],'go',lw=0,label='$N$'+str(narr[0]))
        pylab.plot(karr0+0.3,act[2,:],'g^',lw=0,label='$N$'+str(narr[1]))
        pylab.plot(karr0+0.6,act[3,:],'g^',lw=0,label='$N$'+str(narr[2]))
    for k in range(1,4):
        for mu in munique[k-1]:
            pylab.plot(k+np.array([0,0.6]),2*[mu],'k')
    pylab.xticks([1,1.3,1.6,2,2.3,2.6,3,3.3,3.6],(str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2]),str(narr[0]),str(narr[1]),str(narr[2])))
    pylab.yticks(np.arange(0,4,1))
    pylab.xlim(0.5,4)
    pylab.ylim(0,mth[-1]+0.1)
    pylab.ylabel('maximum margin')
    pylab.xlabel('$N$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(224)
    ax.plot(mth,carray[0,:],'k',label='binary')
    for j in range(len(narr)):
        ax.plot(mth,carray[j+1,:],color4[j],label='$N$='+str(narr[j]))
    ax.plot([0,mth[-1]],2*[41],'k--',lw=1)
    for j in np.arange(0,3.5,0.5):
        ax.plot([j,j],[0,41],'k--',lw=1)
    ax.legend(loc=1)
    ax.set_ylim(0,41+1)
    ax.set_xlim(0,mth[-1]+0.1)
    ax.set_xlabel('maximum margin')
    ax.set_ylabel('cumulative number')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.15,hspace=0.3,wspace=0.3)
    pylab.savefig('morefig8d.svg')
