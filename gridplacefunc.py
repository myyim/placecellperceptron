# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.

import numpy as np
import pylab
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import pickle
import math
import itertools
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

def phase(x,period):
    """phase(x,period) returns the phase of location x with respect to the module with spacing period."""
    return np.mod(x/period,1)

def grid(x,period,prefphase,phsh=0.,gridmodel='gau',sig=0.212):
    """grid(x,period,prefphase,phsh) returns the grid cell activity with prefphase at all location x."""
    if gridmodel == 'gau':
        temp_array = np.array((abs(phase(x-phsh*period,period)-prefphase),1-abs(phase(x-phsh*period,period)-prefphase)))
        temp = np.exp(-np.min(temp_array,axis=0)**2/(2*sig**2))
    elif gridmodel == 'del':
        temp = np.zeros(x.size)
        temp[int(np.round((phsh+prefphase)*period)):x.size:period] = 1
    return temp

def grid1d_orient(x,period,ori,xph=0.,yph=0.,sig=0.106,full=0,mode='exp'):
    r = np.zeros(x.size)
    xv = x*np.cos(ori)
    yv = x*np.sin(ori)
    if mode == 'exp':
        if full == 1:
            for n in range(-1,int(np.ceil(2*yv[-1]/(np.sqrt(3)*period)))+2):
                for m in range(-n/2-1,int(np.ceil(xv[-1]/period-n/2)+1)):
                    r += np.exp((-(xv-(m-xph+(n-yph)/2.)*period)**2-(yv-((n-yph)*np.sqrt(3)/2)*period)**2)/(2*(sig*period)**2))
        elif full == 0:
            for m in range(0,int(np.ceil(xv[-1]/period))+2):
                for n in range(0,m+2):
                    r += np.exp((-(xv-(m-xph+(n-yph)/2.)*period)**2-(yv-((n-yph)*np.sqrt(3)/2)*period)**2)/(2*(sig*period)**2))
    elif mode == 'cos':
        b = np.array([[0,2./np.sqrt(3)],[1.,-1./np.sqrt(3)],[1.,1./np.sqrt(3)]])/period
        for j in range(3):
            r += np.cos(2*np.pi*(b[j,0]*(xv-xph)+b[j,1]*(yv-yph)))/3.
        r += 1
    return r

def frac_vs_S(l,R,Xarr=None,samples=1000,return6=0):
    is_svm = 0
    if is_svm:
        print 'Using SVM'
    else:
        print 'Using perceptron'
    rng = np.random.RandomState(33) # used for smapling only
    N = len(l)
    Ng = np.sum(l)
    pcorr = []
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(R),l[iN],float(iM)/l[iN],0,'del'))
    u = np.array(u)
    Sc = np.linalg.matrix_rank(u)
    print 'rank = ',Sc
    if np.all(Xarr == None):
        Xarray = range(Sc+1,R+1)
    else:
        Xarray = Xarr
    for X in Xarray:
        pc = [1]
        for K in range(2,X/2+1):
            print '----'
            print 'X = '+str(X)+'; K = '+str(K)
            if np.all(Xarr == None):
                com = [list(temp) for temp in itertools.combinations(range(X),K)]
            else:
                com = []
                for j in range(samples):
                    for k in range(K):
                        fds = rng.choice(range(X),K,replace=False)
                        fds.sort()
                        fds = list(fds)
                    com.append(fds)
            count = 0
            for ic in range(len(com)):
                #print X,K,com[ic],nCr(X,K)
                if is_svm:
                    Y = -np.ones(X)
                    Y[com[ic]] = 1
                    m,w,b = svm_margin(u[:,:X].T,Y)
                    dec = np.sign(np.dot(w.T,u[:,:X])+b)
                    if np.sum(np.abs(Y-dec))==0:
                        count += 1
                else:
                    count += perceptron(u[:,:X],com[ic])
            pc.append(float(count)/len(com))
            print count
        pcorr.append(pc)
    print pcorr
    p = []
    if np.all(Xarr == None):
        for j in range(Sc):
            p.append([1]*(j+1))
    for j in range(len(Xarray)):
        pcorr0 = np.copy(pcorr[j])
        pcorr0 = list(pcorr0)
        p0 = np.copy(pcorr0)
        p0 = list(p0)
        pcorr0.reverse()
        p0.extend(pcorr0[np.mod(Xarray[j]+1,2):])
        p0.append(1)
        p.append(p0)
    print p
    psum = []
    pall = []
    if np.all(Xarr == None):
        Xarray = range(1,R+1)
    for j in range(len(Xarray)):
        lsp = []
        lspK = []
        nCrsum = 1
        for K in range(1,Xarray[j]+1):
            nlsp = p[j][K-1]*nCr(Xarray[j],K)
            if np.abs(nlsp-np.round(nlsp)) < 1e-8:
                lspK.append(int(np.round(nlsp)))
                nCrsum += nCr(Xarray[j],K)
            else:
                print 'Please check',Xarray[j],K,nlsp,np.abs(nlsp-np.round(nlsp))
        lsp.append(lspK)
        psum.append(np.sum(lspK)+1)
        pall.append(psum[-1]/float(nCrsum))
        #print psum[-1],nCrsum
    print psum
    if not return6:
        return pall,p
    else:
        p2 = []
        p3 = []
        p4 = []
        p5 = []
        p6 = []
        if np.all(Xarr == None):
            for X in range(1,R+1):
                if X >= 2:
                    p2.append(p[X-1][1])
                if X >= 3:
                    p3.append(p[X-1][2])
                if X >= 4:
                    p4.append(p[X-1][3])
                if X >= 5:
                    p5.append(p[X-1][4])
                if X >= 6:
                    p6.append(p[X-1][5])
        else:
            for j in range(len(Xarray)):
                p2.append(p[j][1])
                p3.append(p[j][2])
                p4.append(p[j][3])
                p5.append(p[j][4])
                p6.append(p[j][5])
        return pall,p2,p3,p4,p5,p6

def testrange(l,itermax=8):
    import fractions
    m = 0
    l = np.array(l)
    N = len(l)
    print 'Sum = '+str(np.sum(l))
    is_int = 0
    critical = []
    while m < itermax+1 and is_int == 0:
        is_int = 1
        for k in l:
            if k*10**m != int(k*10**m):
                is_int = 0
        if is_int == 1:
            rk = np.sum(l*10**m)
            for j in range(2,N+1):
                com = [list(temp) for temp in itertools.combinations(l*10**m,j)]
                for k in range(len(com)):
                    rk += GCD(com[k])*(-1)**(j-1)
            print 'Range = '+str(rk/float(10**m))
        else:
            l0 = []
            for k in l:
                l0.append(int(k*10**m))
            rk = np.sum(l0)
            for j in range(2,N+1):
                com = [list(temp) for temp in itertools.combinations(l0,j)]
                for k in range(len(com)):
                    rk += GCD(com[k])*(-1)**(j-1)
            print 'Range = '+str(rk/float(10**m))+' ; correct to '+str(m)+' decimal places'
        critical.append(rk/float(10**m))
        m += 1
    return critical

def input_margin(X,K=None):
    R = X.shape[1]
    marr = []
    darr = []
    karr = []
    if K == None:
        K = R/2
    for k in range(1,K+1):
        com = [list(temp) for temp in itertools.combinations(range(R),k)]
        for j in range(len(com)):
            Y = np.zeros(R)
            Y[com[j]] = 1
            m,w,b = svm_margin(X.T,Y)
            dec = np.sign(np.dot(w.T,X)+b)
            dec[dec<0] = 0
            if abs(np.sum(np.abs(Y-dec)))<1e-10:
                print list(np.array(com[j])+1),dec,m
            else:
                m = np.inf
            marr.append(m)
            darr.append(np.sum(np.abs(Y-dec)))
            karr.append(k)
    return np.array(marr),np.array(darr),np.array(karr)

def act_mat_grid_binary(l,R=None):
    if R == None:
        R = l[0]
        for j in range(len(l)-1):
            R = lcm(R,l[j+1])
    u = []
    for iN in range(len(l)):
        for iM in range(l[iN]):
            u.append(grid(np.arange(R),l[iN],float(iM)/l[iN],0,'del'))
    return np.array(u)

def input_margin_qp(X,K=None,is_thre=1,is_wconstrained=1):
    R = X.shape[1]
    marr = []
    darr = []
    karr = []
    if K == None:
        K = R/2
    for k in range(1,K+1):
        com = [list(temp) for temp in itertools.combinations(range(R),k)]
        for j in range(len(com)):
            Y = -np.ones(R)
            Y[com[j]] = 1
            try:
                if is_thre:
                    m,w,b = svm_qp(X,Y,is_thre,is_wconstrained)
                    dec = np.sign(np.dot(w.T,X)-b)  # minus here
                else:
                    m,w = svm_qp(X,Y,is_thre,is_wconstrained)
                    dec = np.sign(np.dot(w.T,X))  # minus here
                dec[dec<0] = -1
                print list(np.array(com[j])+1),dec,m
            except:
                m = np.inf
                dec = np.zeros(Y.size)
            marr.append(m)
            darr.append(np.sum(np.abs(Y-dec)))
            karr.append(k)
    return np.array(marr),np.array(darr),np.array(karr)

def random_weightcon(N,X,is_wconstrained=1,seed=99):
    rng = np.random.RandomState(seed)
    xall = rng.rand(N,X)
    print xall
    p = [] # X=1,...,R
    for R in range(1,X+1):
        x = xall[:,:R]
        print '%%%% track = ',R
        count = 0
        for k in range(R/2+1):
            com = [list(temp) for temp in itertools.combinations(range(R),k)]
            for j in range(len(com)):
                Y = -np.ones(R)
                Y[com[j]] = 1
                try:
                    if is_wconstrained:
                        m,w,b = svm_qp(x,Y,1,is_wconstrained)
                        dec = np.sign(np.dot(w.T,x)-b)  # minus here
                    else:
                        m,w = svm_qp(x,Y,0,0)
                        dec = np.sign(np.dot(w.T,x))  # minus here
                    dec[dec<0] = -1
                    print list(np.array(com[j])+1),dec,m
                    if np.abs(np.sum(Y-dec)) == 0:
                        if k == R/2.:
                            count += 1
                        else:
                            count += 2
                except:
                    m = np.inf
                    dec = np.zeros(Y.size)
        p.append(count/(2.**R))
    return p

def detect_field(a,nth,fmerge=10,fwidthmin=0):
    af = np.zeros(a.shape)
    tharr = []
    for j in range(a.shape[0]):
        th = np.mean(a[j,:]) + nth*np.std(a[j,:])
        tharr.append(th)
        idx = pylab.find(a[j,:]-th>=0)  # index of non-zero activity location
        if idx.size > 0:
            di = np.diff(idx)
            ii = idx[np.append([True],di>1)]   # index of the start bump location
            if ii.size > 1:
                bi = pylab.find(di>1)   # bump separating index
                bw = np.append(np.append(bi[0]+1,np.diff(bi)),di.size-bi[-1]) # bump width-1
            else:
                bw = [di.size + 1]
            fc = []
            for l in range(ii.size):
                if bw[l] >= fwidthmin:  # COM
                    fc.append(int(np.round(np.dot(np.arange(ii[l],ii[l]+bw[l]),a[j,ii[l]:ii[l]+bw[l]])/float(np.sum(a[j,ii[l]:ii[l]+bw[l]])))))
            dfc = np.diff(fc)
            if np.any(dfc<fmerge):
                imerge = pylab.find(dfc<fmerge)
                dimage = np.diff(imerge)
                for l in np.flipud(imerge):
                    bw[l] = ii[l+1]-ii[l]+bw[l+1]
                ii[imerge+1] = 0
                ii = ii[ii>0]
                bw[imerge+1] = 0
                bw = bw[bw>0]
                fc = []
                for l in range(ii.size):
                    if bw[l] >= fwidthmin:
                        fc.append(int(np.round(np.dot(np.arange(ii[l],ii[l]+bw[l]),a[j,ii[l]:ii[l]+bw[l]])/float(np.sum(a[j,ii[l]:ii[l]+bw[l]])))))
            af[j,fc] = 1
    return af,tharr

def margin_gridvsrandom(l,K=6,num=10,mode='ext'): # ext=exact, sX=sample X without replacement
    #if 1:
    l = [31,43] #[35,51] #[2,3]
    K = 6
    num = 10
    mode = 's1000'
    u = act_mat_grid_binary(l)
    u /= len(l)
    rng = np.random.RandomState(1)
    margin = []
    rmargin = []    # rmargin[K][trial]
    smargin = []    # smargin[K][trial]
    numKarr = []
    rnumKarr = []
    snumKarr = []
    partfunc = []
    for k in range(1,K+1):
        partition = partitions(k)
        part = []
        margin.append([])
        rmargin.append([])
        smargin.append([])
        numKarr.append([])
        rnumKarr.append([])
        snumKarr.append([])
        numK = 0
        # grid
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
                m,w,b = svm_margin(u.T,Y)
                margin[k-1].append(m)
                dec = np.sign(np.dot(w.T,u)+b)
                dec[dec<0] = 0
                denominator = math.factorial(l[0]-p[0])*math.factorial(p[-1])*math.factorial(l[1]-len(p))
                for j in np.diff(p):
                    denominator *= math.factorial(abs(j))
                (chist,temp) = np.histogram(p,np.arange(0.5,p[0]+1))
                chist = chist[chist>0]
                for j in chist:
                    denominator *= math.factorial(j)
                #print k,p,abs(np.sum(np.abs(Y-dec))),m
                #print p,math.factorial(l[0])*math.factorial(l[1])/denominator,m
                numK = math.factorial(l[0])*math.factorial(l[1])/denominator
                numKarr[k-1].append(numK)
        partfunc.append(len(part))
    # random
    for j in range(num):
        print 'Random '+str(j)
        v = rng.rand(u.shape[0],u.shape[1])
        for jj in range(u.shape[1]):
            v[:,jj] = v[:,jj]/np.sum(v[:,jj])
        for k in range(1,K+1):
            print 'Number of fields: '+str(k)
            rmargin[k-1].append([])
            if mode == 'ext':
                com = [list(temp) for temp in itertools.combinations(range(u.shape[1]),k)]
            elif mode[0] == 's':
                com = []
                for jj in range(int(mode[1:])):
                    temp = rng.choice(range(u.shape[1]),k,replace=False)
                    temp.sort()
                    com.append(list(temp))
            numK = 0
            for icom in com: # len(com) or partfunc[k-1] or 1
                Y = np.zeros(u.shape[1])
                Y[icom] = 1
                m,w,b = svm_margin(v.T,Y)
                dec = np.sign(np.dot(w.T,v)+b)
                dec[dec<0] = 0
                if abs(np.sum(np.abs(Y-dec))) < 1e-6:
                    numK += 1
                    rmargin[k-1][j].append(m)
            #print Y,abs(np.sum(np.abs(Y-dec)))
            #print k,j,abs(np.sum(np.abs(Y-dec)))
            rnumKarr[k-1].append(numK)
    # shuffled
    for j in range(num):
        print 'Shuffled '+str(j)
        v = np.copy(u)
        if 1:
            # shuffle column only
            for jj in range(u.shape[1]):
                temp = v[:,jj]
                rng.shuffle(temp)
                v[:,jj] = temp
        if 0:
            # shuffle both row and column
            v = v.ravel()
            rng.shuffle(v)
            v = v.reshape(u.shape)
        for k in range(1,K+1):
            print 'Number of fields: '+str(k)
            smargin[k-1].append([])
            if mode == 'ext':
                com = [list(temp) for temp in itertools.combinations(range(u.shape[1]),k)]
            elif mode[0] == 's':
                com = []
                for jj in range(int(mode[1:])):
                    temp = rng.choice(range(u.shape[1]),k,replace=False)
                    temp.sort()
                    com.append(list(temp))
            numK = 0
            for icom in com: # len(com) or partfunc[k-1] or 1
                Y = np.zeros(u.shape[1])
                Y[icom] = 1
                m,w,b = svm_margin(v.T,Y)
                dec = np.sign(np.dot(w.T,v)+b)
                dec[dec<0] = 0
                if abs(np.sum(np.abs(Y-dec))) < 1e-6:
                    numK += 1
                    smargin[k-1][j].append(m)
            snumKarr[k-1].append(numK)
    if 0:
        pylab.figure()
        pylab.plot([-1,-1],[0,0],'k',label='grid')
        pylab.plot([-1,-1],[0,0],'bx',label='random')
        pylab.plot([-1,-1],[0,0],'r+',label='shuffled')
        pylab.legend(loc=1)
        for k in range(1,K+1):
            for j in range(len(rmargin[k-1])):
                pylab.plot([k]*len(rmargin[k-1][j]),rmargin[k-1][j],'bx')
                pylab.plot([k]*len(smargin[k-1][j]),smargin[k-1][j],'r+')
                for mu in margin[k-1]:
                    pylab.plot(k+np.array([-0.2,0.2]),2*[mu],'k')
        pylab.xlim(0.5,K+0.5)
        pylab.title('$\lambda$='+str(l)+'; grid std={:.2f}; rand std={:.2f}'.format(np.std(u),np.std(v)))
        pylab.xlabel('number of fields $K$')
        pylab.ylabel('margin $\kappa$')
    with open('fig4A'+mode+'.txt','wb') as f:
        pickle.dump((margin,rmargin,smargin,numKarr,rnumKarr,snumKarr),f)
    return margin,rmargin,smargin,numKarr,rnumKarr,snumKarr
