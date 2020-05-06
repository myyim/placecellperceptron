import numpy as np
import pylab
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import pickle
import math
import itertools

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

### Parameters
eps = 1e-8
### Figures
color = ['#1e90ff','#ff8c00','#3cb371']
color4 = ['#DC143C','#3cb371','#9932CC','#FFD700','#1e90ff','#ff8c00','#3cb371','m']

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

def lcm(a,b):
    """ lcm(a,b) returns the LCM of a and b"""
    import fractions
    return abs(a*b)/fractions.gcd(a,b) if a and b else 0

def nCr(n,r):
    """nCr(n,r) returns n choose r"""
    if n>=r:
        return math.factorial(n)/math.factorial(r)/math.factorial(n-r)
    else:
        return 0

def cover(n,p):
    """cover(n,p) returns the number of linearly separable pattern sets out of p sets given n inputs in general position based on Cover's counting theorem"""
    temp = 0
    for j in range(np.min([n,p])):
        temp += 2*nCr(p-1,j)
    return temp

def perceptron(u,yloc):
    eta = 1
    Nshot = 100
    Ng = u.shape[0]
    R = u.shape[1]
    y = np.zeros(R)
    y[yloc] = 1
    w = np.zeros(Ng)
    theta = 0.
    sigma = np.zeros(R)
    itrial = 0
    while itrial < Nshot and np.sum(np.abs(y-sigma))>0:
        for ix in range(R):
            sigma[ix] = (np.dot(w,u[:,ix])-theta) > 0
            w += eta*(y[ix]-sigma[ix])*u[:,ix]
            #theta += -eta*(y[ix]-sigma[ix])
        itrial += 1
        sloc = np.argwhere(np.dot(w,u)-theta>0).T[0]
    #if len(sloc)==len(yloc) and np.sum(sloc==yloc)==len(yloc):
        #print sloc+1, np.diff(sloc)
    if len(sloc)!=len(yloc) or np.sum(sloc==yloc)!=len(yloc):
        return 0
    else:
        #print sloc+1, np.diff(sloc)
        return 1

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
            if np.abs(nlsp-np.round(nlsp)) < eps:
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
        """
        p2 = []
        p3 = []
        p4 = []
        if np.all(Xarr == None):
            for X in range(1,R+1):
                if X >= 2:
                    p2.append(p[X-1][1])
                if X >= 3:
                    p3.append(p[X-1][2])
                if X >= 4:
                    p4.append(p[X-1][3])
        else:
            for j in range(len(Xarray)):
                p2.append(p[j][1])
                p3.append(p[j][2])
                p4.append(p[j][3])
        return pall,p2,p3,p4"""
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

"""
to be removed
def frac_vs_S_block(l,R,Xarr,samples=1000):
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
    """

#def frac4real():
if 0:
    # q = 0,1,2,3(,4)
    # X = 7,8,9,10,11,12
    #l = np.array([np.sqrt(2),np.sqrt(3),5])
    #l = np.array([np.sqrt(2),np.exp(1),np.pi])
    l = np.array([np.sqrt(8),np.sqrt(15)])
    #R = l[0]
    #for j in range(len(l)-1):
    #    R *= l[j+1]
    for q in [1]: #range(1,9):
        print ' q = '+str(q)
        ltrunc = np.floor(l*10**q).astype(int)
        if q == 0:
            pall,p = frac_vs_S(ltrunc,6)
        else:
            Xarr = np.arange(8,13)*10**q
            pall,p = frac_vs_S(ltrunc,int(Xarr[-1]),Xarr=Xarr,samples=1000)
    with open('frac_real'+str(q)+'X'+str(Xarr[-1])+'.txt','wb') as f:
        pickle.dump((pall,p),f)

#with open('frac_real'+str(q)+'X'+str(Xarr[-1])+'.txt','r') as f:
#pall,p = pickle.load(f)
def GCD_num(l):
    import fractions
    gcd = l[0]
    for j in range(1,len(l)):
        gcd = fractions.gcd(gcd,l[j])
    return gcd

def testrange(l,itermax=8):
    import fractions
    import itertools
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
                    rk += GCD_num(com[k])*(-1)**(j-1)
            print 'Range = '+str(rk/float(10**m))
        else:
            l0 = []
            for k in l:
                l0.append(int(k*10**m))
            rk = np.sum(l0)
            for j in range(2,N+1):
                com = [list(temp) for temp in itertools.combinations(l0,j)]
                for k in range(len(com)):
                    rk += GCD_num(com[k])*(-1)**(j-1)
            print 'Range = '+str(rk/float(10**m))+' ; correct to '+str(m)+' decimal places'
        critical.append(rk/float(10**m))
        m += 1
    return critical

def svm_margin(X,Y):
    num = X.shape[0]
    if Y.shape[0] != num:
        print 'Dimensions mismatched!'
        return
    dim = X.shape[1]
    w = np.zeros(dim)
    from sklearn import svm
    hyp = svm.SVC(kernel='linear',C=10000,cache_size=20000,tol=1e-5)
    hyp.fit(X,Y)
    for j in range(hyp.support_.size):
        w += hyp.dual_coef_[0][j]*hyp.support_vectors_[j]
    return 2./pylab.norm(w),w,hyp.intercept_[0]

def input_margin(X,K=None):
    import itertools
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
            """
            if np.sum(np.abs(Y-dec))<1e-10 and gridmodel == 'del':
                pinvA = pylab.dot(pylab.inv(np.dot(A.T,A)),A.T)
                print list(np.array(com[j])+1),m,2./(pylab.norm(np.dot(w,pinvA)))#np.sum(np.abs(Y-dec))
                print w
                print np.dot(w,pinvA)
            elif np.sum(np.abs(Y-dec))<1e-10:
                print list(np.array(com[j])+1),m
                print w
                """
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

def svm_qp(x,y,is_thre=1,is_wconstrained=1):
    """x is the input matrix with dimension N (number of neurons+threshold if any) by P (number of patterns). y is the desired output vector of dimension P."""
    import qpsolvers
    R = x.shape[1]
    G = -(x*y).T
    if is_thre:
        N = x.shape[0] + 1
        G = np.append(G.T,y)
        G = G.reshape(N,R)
        G = G.T
        P = np.identity(N)
        P[-1,-1] = 1e-12    # epsilon
        #for j in range(N):
        #P[j,j] += 1e-16
        #P += 1e-10
    else:
        N = x.shape[0]
        P = np.identity(N)
    if is_wconstrained:
        if is_thre:
            G = np.append(G,-np.identity(N)[:N-1,:])
            G = G.reshape(R+N-1,N)
            h = np.array([-1.]*R+[0]*(N-1))
        else:
            G = np.append(G,-np.identity(N))
            G = G.reshape(R+N,N)
            h = np.array([-1.]*R+[0]*N)
    else:
        h = np.array([-1.]*R)
    w = qpsolvers.solve_qp(P,np.zeros(N),G,h)
    #w = qpsolvers.solve_qp(np.identity(N),np.zeros(N),G,h,np.zeros(N),0) #CVXOPT,qpOASES,quadprog
    if is_thre:
        return 2/pylab.norm(w[:-1]),w[:-1],w[-1]
    else:
        return 2/pylab.norm(w),w

def input_margin_qp(X,K=None,is_thre=1,is_wconstrained=1):
    import itertools
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

def partitions(n, I=1):
    yield (n,)
    for j in range(I, n//2 + 1):
        for p in partitions(n-j, j):
            yield p + (j,)

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

def margin_perturbedgrid(l,K=6,num=10): # ext=exact, sX=sample X without replacement
    #if 1:
    #l = [31,43] #[35,51] #[2,3]
    #K = 6
    #num = 10
    u = act_mat_grid_binary(l)
    u /= len(l)
    sarr = [0.1,0.5]
    rng = np.random.RandomState(1)
    margin = []
    for s in range(len(sarr)):
        margin.append([])
        for k in range(K):
            margin[s].append([])
    com = []
    for k in range(1,K+1):
        com.append([])
        partition = partitions(k)
        for p in partition:
            # Young diagram
            mat = np.zeros((l[0],l[1]),dtype='int')
            for j in range(len(p)):
                mat[:p[j],j] = 1
            i1 = np.tile(range(l[0]),l[1])
            i2 = np.tile(range(l[1]),l[0])
            com[k-1].append(mat[i1,i2])
    for j in range(num):
        print 'Random '+str(j)
        rmat = rng.rand(u.shape[0],u.shape[1])
        for s in range(len(sarr)):
            v = u + sarr[s]*rmat/2 #+ sarr[s]*sum(u)*2*rmat/u.size
            for jj in range(u.shape[1]):
                v[:,jj] = v[:,jj]/np.sum(v[:,jj])
            for k in range(1,K+1):
                print 'Number of fields: '+str(k)
                margin[s][k-1].append([])
                for Y in com[k-1]: # len(com) or partfunc[k-1] or 1
                    m,w,b = svm_margin(v.T,Y)
                    dec = np.sign(np.dot(w.T,v)+b)
                    dec[dec<0] = 0
                    if abs(np.sum(np.abs(Y-dec))) < 1e-6:
                        margin[s][k-1][j].append(m)
                    else:
                        print j,s,k,abs(np.sum(np.abs(Y-dec)))
    with open('test_fig4_perturbedgrid.txt','wb') as f:
        pickle.dump(margin,f)

def plot_margin_perturbedgrid():
    with open('fig4As1000.txt','rb') as f:
        margin0,rmargin,smargin,numKarr,rnumKarr,snumKarr = pickle.load(f)
    margin0 = margin0[:6]
    with open('test_fig4_perturbedgrid.txt','rb') as f:
        margin = pickle.load(f)
    K = len(margin[0])
    num = len(margin[0][0])
    import seaborn as sns
    fig = pylab.figure(figsize=[7,8.5])
    ax = fig.add_subplot(323)
    for k in range(1,len(margin0)+1):
        for mu in margin0[k-1]:
            pylab.plot(k-1+np.array([-0.3,0.3]),2*[mu],'k')
    mmat1 = [item for sublist in margin[0] for subsublist in sublist for item in subsublist]
    nmat1 = ['100%']*len(mmat1)
    mmat2 = [item for sublist in margin[1] for subsublist in sublist for item in subsublist]
    nmat2 = ['200%']*len(mmat2)
    kmat1 = []
    kmat2 = []
    for k in range(K):
        kmat1.extend([k+1]*(len(margin0[k])*num))
        kmat2.extend([k+1]*(len(margin0[k])*num))
    sns.violinplot(np.append(kmat1,kmat2),np.append(mmat1,mmat2),np.append(nmat1,nmat2),inner=None,linewidth=.4,bw=.2,gridsize=100)
    ax.set_yticks(np.arange(0,0.5,0.2))
    ax.set_xlim(-0.5,K-0.5)
    ax.set_ylim(0,0.4)
    ax.set_xlabel('number of fields (K)')
    ax.set_ylabel('margin')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

#margin_perturbedgrid([31,43])
#plot_margin_perturbedgrid()

def fig1():
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
    pylab.savefig('f1.svg')

def fig5():
    pylab.figure(figsize=[7,7])
    """
    ax = pylab.subplot(221)
    l = np.array([31,42]) #np.array([3,4])
    R = 150
    sig = 0.16#0.212
    x = np.arange(-0.5,R-0.5+0.01,0.01)
    grid1 = grid(x,l[0],0.38,sig=sig)
    grid2 = grid(x,l[1],0.25,sig=sig)
    grid3 = grid(x,l[1],0.5,sig=sig)
    temp = grid1+grid2+2*grid3-8.7
    pylab.plot(2*[15],[-9,1],'k--',lw=0.5)
    pylab.plot(2*[104],[-9,1],'k--',lw=0.5)
    #pylab.plot(2*[140],[-9,1],'k--',lw=0.5)
    pylab.plot(x,grid1-0.5,c=color[0])
    pylab.plot(x,grid2-2.5,c=color[1])
    pylab.plot(x,grid3-4.5,c=color[1])
    pylab.plot(x,temp,'r')
    pylab.plot(x[1201:1803],temp[1201:1803],'g')
    pylab.plot(x[9970:10785],temp[9970:10785],'g')
    #pylab.plot(x[973:1114],temp[973:1114],'g')
    pylab.plot([x[0],x[-1]],-5.9*np.ones(2),'k--')
    #pylab.text(5,-6,'$\lambda_1$',color=color[0],fontsize=font_size-2)
    #pylab.text(7,-6,'$\lambda_2$',color=color[1],fontsize=font_size-2)
    #pylab.text(9,-6,'$\lambda_1$',color='g',fontsize=font_size-2)
    #pylab.text(11,-6,'2$\lambda_2$',color='g',fontsize=font_size-2)
    pylab.ylim(-9.3,1)
    pylab.xlim(-0.5,R)
    #ax.set_title('place cell output')
    pylab.xticks([])
    pylab.yticks([])
    ax.axis('off')
    """
    """
    # 1D slices through 2D illustration
    ax = pylab.subplot(223)
    l = [31,43]
    R = 240
    x = np.arange(-0.5,R-0.5+0.01,0.1)
    ori = [0.39,0.46]
    g1 = grid1d_orient(x,l[0],ori[0],xph=1./3,yph=0.,sig=sig)
    g2 = grid1d_orient(x,l[1],ori[1],xph=0.5,yph=0.2,sig=sig)
    g3 = grid1d_orient(x,l[1],ori[1],xph=0.2,yph=0.5,sig=sig)
    #temp = grid(x,3,2./3,sig=sig)+grid(x,4,0.25,sig=sig)+1.5*grid(x,4,0.5,sig=sig)-8.7
    temp = (g1+g2+g3)*1.6-8.8
    pylab.plot(2*[45],[-8.5,1],'k--',lw=0.5)
    pylab.plot(2*[120],[-8.5,1],'k--',lw=0.5)
    pylab.plot(2*[154],[-8.5,1],'k--',lw=0.5)
    pylab.plot(2*[230],[-8.5,1],'k--',lw=0.5)
    pylab.plot(x,g1-0.5,c=color[0])
    pylab.plot(x,g2-2.5,c=color[1])
    pylab.plot(x,g3-4.5,c=color[1])
    pylab.plot(x,temp,'r')
    pylab.plot(x[438:515],temp[438:515],'g')
    pylab.plot(x[1180:1270],temp[1180:1270],'g')
    pylab.plot(x[1525:1596],temp[1525:1596],'g')
    pylab.plot(x[2280:2338],temp[2280:2338],'g')
    pylab.plot([x[0],x[-1]],-7*np.ones(2),'k--')
    pylab.ylim(-9.3,1)
    pylab.xlim(-0.5,R)
    pylab.xticks([])
    pylab.yticks([])
    ax.axis('off')
    """
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
    if 0:
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

    """
    # prob vs w (polar plot; not used)
    N = len(l)
    Ngr = 72
    vr = np.zeros(((N,Ngr,R))) # for reference
    for iN in range(N):
        for iM in range(Ngr):
            vr[iN,iM,:] = grid(x,l[iN],iM/float(Ngr),phsh=0.,gridmodel='gau',sig=sig)
    for iNp in [19]:
        pid = iNp
        vmodule = np.zeros((len(l),np.max(l)))
        for j in range(len(l)):
            vmodule[j,:np.max(l)] = np.dot(w[pid,j*Ng/len(l):(j+1)*Ng/len(l)],v[pid,j*Ng/len(l):(j+1)*Ng/len(l),:np.max(l)])
        vmodule -= np.min(vmodule) #nth*np.mean(vmodule)
        vmodule[vmodule<0] = 0
        vmodule /= np.max(vmodule)
        count = np.zeros((N,Ngr))
        for iN in range(N):
            for j in range(Ngr):
                temp = pylab.find(vr[iN,j,:]>0.95)
                for k in fid1d[pid]:
                    count[iN,j] += np.any(k==temp)
            count[iN,:] /= nf[pid]
        fig = pylab.figure(figsize=[8,6])
        fig.text(0.05,0.95,'A',fontsize=fs)
        fig.text(0.5,0.95,'B',fontsize=fs)
        fig.text(0.05,0.5,'C',fontsize=fs)
        fig.text(0.5,0.5,'D',fontsize=fs)
        for j in range(N):
            ax = fig.add_subplot(2,4,j+2, projection='polar')
            for iw in range(w.shape[1]/3):
                pylab.polar([p1d[pid,j*(w.shape[1]/3)+iw]*2*np.pi]*2,[0,w[pid,j*(w.shape[1]/3)+iw]/np.max(w[pid])],'#98FB98',label='weights') # pale green
            pylab.polar(np.append(np.arange(l[j])*2*np.pi/l[j],[0]),np.append(vmodule[j,:l[j]],vmodule[j,0]),'#008000',label='grid input sum') # green
            pylab.polar(np.append(np.arange(Ngr)*2*np.pi/Ngr,[0]),np.append(count[j,:],count[j,0]),'m',label='frac coincidence')
            #if j == 0:
                #pylab.legend((line1,line2,line3),('frac coincidence','weights','grid input sum'),frameon=False)
            pylab.title('$\lambda=$'+str(l[j]))
            pylab.ylim(0,1)
            pylab.xticks([])
            pylab.yticks([])
        pylab.legend(loc=1,frameon=False)
        ax = fig.add_subplot(223)
        for j in range(5):
            pylab.plot(range(l[0]),w[pid,j]*v[pid,j,:l[0]],'#98FB98')
            print w[pid,j],np.argmax(v[pid,j,:l[0]])
        pylab.plot(range(l[0]),np.dot(w[pid,:5],v[pid,:5,:l[0]]),'#008000')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        pylab.savefig('temp'+str(pid)+'.svg')
        pylab.close('all')
    fig = pylab.figure(figsize=[8,6])
    ax = fig.add_subplot(223)
    acf1 = np.correlate(v[pid,0,:],v[pid,0,:],'same')
    acf2 = np.correlate(v[pid,Ng/3,:],v[pid,Ng/3,:],'same')
    acf3 = np.correlate(v[pid,2*Ng/3,:],v[pid,2*Ng/3,:],'same')
    acf1 /= np.max(acf1)
    acf2 /= np.max(acf2)
    acf3 /= np.max(acf3)
    pylab.plot(range(-R/2,R/2),(acf1+acf2+acf3)/3.,'0.6',lw=1,label='ACF')
    pylab.legend(loc=1,frameon=False)
    pylab.xlim(0,300)
    ax = fig.add_subplot(224)
    print 'plot 2D'
    l = [31,43]
    ori = [0.39,0.46]
    Ng = Ng*2/3
    iN = 0
    v1 = grid1d_orient(x,l[0],ori[0],0,0,sig=sig)
    v2 = grid1d_orient(x,l[1],ori[1],0,0,sig=sig)
    acf1 = np.correlate(v1,v1,'same')
    acf2 = np.correlate(v2,v2,'same')
    acf1 /= np.max(acf1)
    acf2 /= np.max(acf2)
    pylab.plot(range(-R/2,R/2),(acf1+acf2)/2.,'0.6',lw=1,label='ACF')
    pylab.xlim(0,300)
    """
    """
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
    pylab.plot(range(-R/2,R/2),(acf1+acf2+acf3)/3.,'k',label='ACF')
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
    pylab.plot(range(-R/2,R/2),(acf1+acf2)/2.,'k',label='ACF')
    pylab.xlim(0,300)
    #pylab.yticks([0,5,10])
    pylab.xlabel('IFI')
    pylab.title('1D slices through 2D grid')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(top=0.9,wspace=0.3,hspace=0.4)
    pylab.savefig('f5_Ng_'+str(Ng/2)+'_seed'+str(seed)+'.svg')
    """
    """
        # Albert Lee's data
    N = 650
    L = 9
    R = 4000
    ifi = []
    nf = np.zeros(N)
    for k in range(N):
        ifi.append([])
    #af = np.zeros(((L,N,R)))
    for j in range(L):
        #af[j,:,:] = np.loadtxt('field_lap'+str(j+1)+'.txt')
        for k in range(N):
            nf[k] += np.sum(af[j,k,:])
            idx = pylab.find(af[j,k,:]==1)
            ifi[k].extend(np.diff(idx))
    nforder = []
    nf2 = np.copy(nf)
    while len(nforder)<N:
        nfmax = np.max(nf2)
        nfidx = pylab.find(nf2==nfmax)
        nforder.extend(nf2[nfidx])
        nf2[nfidx] = 0
        print len(nforder)
    ntop = [5,10,20,N]
    sig = 10
    ker = np.exp(-0.5*np.arange(-100/2,100/2+1)**2/(sig**2))/np.sqrt(2*np.pi*(sig**2))
    #np.convolve(inp[iNr,:],np.exp(-0.5*np.arange(-convw/2,convw/2+1)**2/(2**2))/np.sqrt(2*np.pi*(2**2)),'same')
    fig = pylab.figure(figsize=[7,7])
    for j in range(len(ntop)):
        ax = pylab.subplot(len(ntop),1,j+1)
        idx = pylab.find(nf>=nforder[ntop[j]-1])
        ifiplot = []
        for k in idx:
            ifiplot.extend(ifi[k])
        #pylab.hist(ifiplot,np.arange(0.5,R+1,bin),color='k')
        temp = np.histogram(ifiplot,np.arange(0.5,R+1,1))
        pylab.plot(convolve(temp[0],ker,'same'))
        pylab.xlim(0,500)
        if j == 0:
            pylab.title('kernal $\sigma$ = '+str(sig))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        pylab.savefig('ifi_width'+str(sig)+'.png')

    sig = 10
    ker = np.exp(-0.5*np.arange(-100/2,100/2+1)**2/(sig**2))/np.sqrt(2*np.pi*(sig**2))
    fig = pylab.figure(figsize=[12,6])
    for j in range(5):
        #idx = pylab.find(nf==nforder[j])   # top cells
        idx = range(j*650/5,(j+1)*650/5)
        ifiplot = []
        for k in idx:
            ifiplot.extend(ifi[k])
        #pylab.hist(ifiplot,np.arange(0.5,R+1,bin),color='k')
        temp = np.histogram(ifiplot,np.arange(0.5,R+1,1))
        pylab.plot(convolve(temp[0],ker,'same'))
    idx = range(N)
    ifiplot = []
    for k in idx:
        ifiplot.extend(ifi[k])
    temp = np.histogram(ifiplot,np.arange(0.5,R+1,1))
    pylab.plot(convolve(temp[0],ker,'same'),'k')
    pylab.xlim(0,500)
    pylab.title('width = '+str(sig))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.savefig('ifigroup_width'+str(sig)+'.png')"""

# Remove
#def poster_ifi():
if 0:
    print 'Poster'
    Ng = 10  # dinstinct for each neuron
    Np = 20
    l = [31,43] #[53,87] #[31,43]
    R = 10000
    sig = 0.16
    bin = 1
    #ori = [0.2153,0.3162] # ori2
    #ori = [0.39,0.46] # ori1
    rng = np.random.RandomState(52)
    x = np.arange(R)
    px = rng.rand(Np,Ng)
    py = rng.rand(Np,Ng)
    w = rng.lognormal(0,0.25,size=[Np,Ng])
    fig = pylab.figure(figsize=[10,4])
    # 1D
    ax = pylab.subplot(121)
    ori = [0,0]
    nth = 2
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    acf = np.zeros((Np,2*R/bin-1))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
            #v[iN,iM,:] = grid(x,l[iN],float(iM)/l[iN],phsh=0.,gridmodel='del'')    #
            a[iN,:] += w[iN,iM]*v[iN,iM,:]
    print '1D'
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
    y = np.histogram(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    for j in range(len(l)):
        for k in range(1,10):
            pylab.plot([k*l[j]]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    pylab.ylabel('number')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # slices through 2D
    ax = pylab.subplot(122)
    nth = 2
    ori = [0.39,0.46]
    v = np.zeros(((Np,Ng,R)))
    a = np.zeros((Np,R))
    acf = np.zeros((Np,2*R/bin-1))
    for iN in range(Np):
        for iM in range(Ng):
            v[iN,iM,:] = grid1d_orient(x,l[iM/(Ng/len(l))],ori[iM/(Ng/len(l))],px[iN,iM],py[iN,iM],sig=sig)
            #v[iN,iM,:] = grid(x,l[iN],float(iM)/l[iN],phsh=0.,gridmodel='del'')    #
            a[iN,:] += w[iN,iM]*v[iN,iM,:]
    print '2D'
    af,tharr = detect_field(a,nth,fmerge=10,fwidthmin=0)
    fid = []
    for iN in range(Np):
        fid.append(list(pylab.find(af[iN,:]==1)))
    ifiall = []
    for iN in range(Np):
        ifiall.extend(np.diff(fid[iN]))
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
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin),color='k')
    pylab.xlim(0,300)
    pylab.xlabel('IFI')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.subplots_adjust(left=0.1,top=0.9,right=0.95,bottom=0.17,hspace=0.5,wspace=0.35)
    pylab.savefig('ifi'+str(Np)+'.svg')

#def fig5b():
if 0:
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
    pylab.savefig('slices_act_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'nstd'+str(nth)+'.png')
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
    """
    for j in range(len(l)):
        for k in range(1,10):
            #print k*l[j]
            #print k*l[j]*np.sqrt(3)
            #print k*l[j]*np.sqrt(7)
            pylab.plot([k*l[j]]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
            #pylab.plot([k*l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
            #pylab.plot([k*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
    """
    """
        print '##### 2'
        print l[j]+l[j]*np.sqrt(3)
        print l[j]+l[j]*np.sqrt(7)
        print l[j]*np.sqrt(3)+l[j]*np.sqrt(7)
        print '##### 3'
        print 2*l[j]+l[j]*np.sqrt(3)
        print 2*l[j]+l[j]*np.sqrt(7)
        print 2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)
        print l[j]+2*l[j]*np.sqrt(3)
        print l[j]+2*l[j]*np.sqrt(7)
        print l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)
        print l[j]+l[j]*np.sqrt(3)+l[j]*np.sqrt(7)
        # 2
        #pylab.plot([l[j]+l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        #pylab.plot([l[j]+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        # 3
        pylab.plot([2*l[j]+l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        # 4
        pylab.plot([3*l[j]+l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([3*l[j]+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([3*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+3*l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+3*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]*np.sqrt(3)+3*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+2*l[j]*np.sqrt(3)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([2*l[j]+l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+2*l[j]*np.sqrt(3)+l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)
        pylab.plot([l[j]+l[j]*np.sqrt(3)+2*l[j]*np.sqrt(7)]*2,[0,max(y[0])+1],'--',c=color[j],lw=0.5)"""
    pylab.hist(ifiall,np.arange(0.5,np.max(ifiall)+1,bin))
    pylab.xlim(0,200)
    pylab.savefig('slices_ifi_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'b'+str(bin)+'nstd'+str(nth)+'.png')
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
    pylab.savefig('slices_acf_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'b'+str(bin)+'nstd'+str(nth)+'.png')
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
    pylab.savefig('slices_psd_l'+str(l)+'g'+str(Ng)+'R'+str(R)+'nstd'+str(nth)+'.png')

def fig2():
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
    #pylab.plot([-1,-1],[1,3],'go',ms=15,alpha=0.5)
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
    pylab.savefig('f2.svg')
    """
    ax = pylab.subplot(234)
    pylab.plot([-1,-1],[1,3],'k',lw=1)
    pylab.plot([1,1],[1,3],'k',lw=1)
    pylab.plot([0,0],[0,2],'k',lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,1,0,-1],[3,3,2,3],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,-1],[1,3],'go',ms=15,alpha=0.5)
    pylab.text(-1,5,'A B C D E F',fontsize=10)
    pylab.xlim(-1.5,1.5)
    pylab.ylim(-0.5,4.5)
    pylab.text(-1.1,4,'Realizable')
    ax.axis('off')
    ax = pylab.subplot(235)
    pylab.plot([-1,-1],[1,3],'k',lw=1)
    pylab.plot([1,1],[1,3],'k',lw=1)
    pylab.plot([0,0],[0,2],'k',lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,1,0,-1],[3,3,2,3],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,0,1],[1,2,1],'ro',ms=15,alpha=0.5)
    pylab.xlim(-1.5,1.5)
    pylab.ylim(-0.5,4.5)
    pylab.text(-1.3,4,'Unrealizable')
    ax.axis('off')
    # C
    ax = pylab.subplot(236)
    pylab.plot([-1,-1],[1,3],'k',lw=1)
    pylab.plot([1,1],[1,3],'k',lw=1)
    pylab.plot([0,0],[0,2],'k--',lw=1)
    pylab.plot([-1,1,0,-1],[1,1,0,1],'k--',lw=1)
    pylab.plot([-1,1],[1,1],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,1,0,-1],[3,3,2,3],'ko-',lw=1,ms=15,mfc='w')
    pylab.plot([-1,0,1],[1,2,1],'go',ms=15,alpha=0.5)
    pylab.xlim(-1.5,1.5)
    pylab.ylim(-0.5,4.5)
    pylab.text(-1.1,4,'Realizable')
    ax.axis('off')
    pylab.subplots_adjust(left=0.05,top=0.95,right=0.95,bottom=0.1,wspace=0.2)
    """

def fig3():
    import os.path
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
    if os.path.isfile(fname+'.txt'):
        with open(fname+'.txt','rb') as f:
            pall,p2,p3,p4,p5,p6 = pickle.load(f)
    else:
        pall,p2,p3,p4,p5,p6 = frac_vs_S(l,R,return6=1)
        with open(fname+'.txt','wb') as f:
            pickle.dump((pall,p2,p3,p4,p5,p6),f)
    ax = pylab.subplot(321)
    pylab.plot(range(1,R+1),np.ones(R),'o-',ms=5,label='1')
    pylab.plot(range(2,R+1),p2,'o-',ms=5,label='2')
    pylab.plot(range(3,R+1),p3,'o-',ms=5,label='3')
    pylab.plot(range(4,R+1),p4,'o-',ms=5,label='4')
    pylab.plot(range(5,R+1),p5,'o-',ms=5,label='5')
    pylab.plot(range(6,R+1),p6,'o-',ms=5,label='6')
    #pylab.plot([1,R],[0,0],'k--',lw=1)
    #pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
    #pylab.plot([1,R],[1,1],'k--',lw=1)
    #pylab.plot([Sc]*2,[0,1],'k--',lw=1)
    pylab.plot([R]*2,[0,1],'k--',lw=1)
    pylab.text(11,0.85,'$L$')
    pylab.ylim(0,1.05)
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    pylab.legend(loc=3,frameon=False)
    #pylab.legend(bbox_to_anchor=(0.38,0.53),frameon=False)
    pylab.ylabel('realizable fraction')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_title('$\lambda=\{3,4\}$')
    # B
    ax = pylab.subplot(322)
    mpl.rcParams['legend.fontsize'] = font_size-7
    pylab.plot(range(1,R+1),pall,'ko-',ms=5,label='grid')
    #pylab.plot([1,R],[0,0],'k--',lw=1)
    #pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
    #pylab.plot([1,R],[1,1],'k--',lw=1)
    #pylab.plot([Sc]*2,[0,1.05],'y')
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
    #pcover = []
    #for x in range(1,R+1):
    #    pcover.append(cover(np.sum(l),x)/float(2**x))
    #pylab.plot(np.arange(1,R+1,1),pcover,'o-',c='0.8',ms=6,label='random ($N$)')
    pylab.legend(loc=6,frameon=False)
    pylab.plot(range(1,R+1),pall,'ko-',ms=5,label='grid')
    #pylab.legend(bbox_to_anchor=(0.38,0.53),frameon=False)
    pylab.ylim(0,1.05)
    pylab.xticks([1,Sc,R],('1',str(Sc),str(R)))
    pylab.yticks([0,1])
    #pylab.text(3.7,0.9,'$R_{copr}$',color='y')
    pylab.text(6.2,0.05,'$l^*$')
    pylab.text(11,0.05,'$L$')
    #pylab.ylabel('Fraction')
    pylab.xlabel('$l$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # C
    """
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
    """
    """
    # E
    #l = np.array([np.sqrt(2),np.sqrt(3),5])
    l = np.array([np.sqrt(10),np.sqrt(17)])
    if 0:
        temp = testrange(l)
        for q in range(9):
            print q,np.floor(l*10**q),int(np.round(temp[q]*10**q)),temp[q]
    #l = np.array([np.sqrt(8),np.sqrt(15)])
    if 0:
        temp = testrange(l)
        for q in range(9):
            print q,np.floor(l*10**q),int(np.round(temp[q]*10**q)),temp[q]
    """
    # E
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
    pylab.savefig('f3.svg')

def morefig3():
    rng = np.random.RandomState(30)
    pylab.figure(figsize=[8,4])
    ax = pylab.subplot(121)
    M = 2
    lsum = []
    lsum_int = []
    ctrackq0 = []
    ctrackq2 = []
    ctrackq4 = []
    # [lmin,lmax)
    lmin = 3
    lmax = 100
    for j in range(100):
        l = lmin + rng.rand(M)*(lmax-lmin)
        temp = testrange(l)
        lsum.append(np.sum(l))
        lsum_int.append(np.sum(np.floor(l)))
        ctrackq0.append(temp[0])
        ctrackq2.append(temp[2])
        ctrackq4.append(temp[4])
    pylab.plot(lsum_int,ctrackq0,'.',lw=0,ms=5,label='$R_{int}$')
    pylab.plot(lsum,ctrackq2,'.',lw=0,ms=5,label='$R_{re}^{q=2}$')
    pylab.plot(lsum,ctrackq4,'.',lw=0,ms=5,label='$R_{re}^{q=4}$')
    #pylab.plot(lsum,ctrackq8,'.',lw=0,ms=5,label='$S_{real}^{q=8}$')
    pylab.legend(loc=2,frameon=False)
    pylab.plot([1,200],[1,200],'k--',lw=1)
    pylab.xlim(0,200.5)
    pylab.ylim(0,200.5)
    pylab.xticks(range(0,201,50))
    pylab.yticks(range(0,201,50))
    pylab.xlabel('$\Sigma$')
    pylab.ylabel('$R^q_{re}$')
    pylab.title('2 modules: 3-100')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax = pylab.subplot(122)
    lsum = []
    lsum_int = []
    ctrackq0 = []
    ctrackq2 = []
    ctrackq4 = []
    # [lmin,lmax)
    lmin = 30
    lmax = 1000
    for j in range(100):
        l = lmin + rng.rand(M)*(lmax-lmin)
        temp = testrange(l)
        lsum.append(np.sum(l))
        lsum_int.append(np.sum(np.floor(l)))
        ctrackq0.append(temp[0])
        ctrackq2.append(temp[2])
        ctrackq4.append(temp[4])
    pylab.plot(lsum_int,ctrackq0,'.',lw=0,ms=5,label='$R_{int}$')
    #pylab.plot(lsum,ctrackq2,'.',lw=0,ms=5,label='$R_{re}^{q=2}$')
    #pylab.plot(lsum,ctrackq4,'.',lw=0,ms=5,label='$R_{re}^{q=4}$')
    pylab.plot([1,2000],[1,2000],'k--',lw=1)
    pylab.xlim(0,2000.5)
    pylab.ylim(0,2000.5)
    pylab.xticks(range(0,2001,1000))
    pylab.yticks(range(0,2001,1000))
    pylab.xlabel('$\Sigma$')
    pylab.title('2 modules: 30-1000')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.savefig('morefig3.svg')

def fig4():
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
    """
    fig = pylab.figure(figsize=[7,10])
    fig.text(0.02,0.95,'A',fontsize=fs)
    fig.text(0.02,0.65,'B',fontsize=fs)
    fig.text(0.48,0.65,'C',fontsize=fs)
    fig.text(0.02,0.3,'D',fontsize=fs)
    fig.text(0.48,0.3,'E',fontsize=fs)
    """
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
    """
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
    marr_rand = np.copy(marr[realizable])"""
    """
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
    """
    # A
    fig = pylab.figure(figsize=[7,8.5*0.8])
    fig.text(0.02,0.65,'A',fontsize=fs)
    fig.text(0.48,0.65,'B',fontsize=fs)
    fig.text(0.02,0.3,'C',fontsize=fs)
    fig.text(0.48,0.3,'D',fontsize=fs)
    ax = fig.add_subplot(323)
    l = [31,43]
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    K = 6
    u = act_mat_grid_binary(l)
    u /= 2
    mode = 's1000'
    with open('fig4As1000.txt','rb') as f:
        margin,rmargin,smargin,numKarr,rnumKarr,snumKarr = pickle.load(f)
    margin = margin[:K]
    rmargin = rmargin[:K]
    smargin = smargin[:K]
    numKarr = numKarr[:K]
    rnumKarr = rnumKarr[:K]
    snumKarr = snumKarr[:K]
    #margin,rmargin,smargin,numKarr,rnumKarr,snumKarr = margin_gridvsrandom(l=l,K=K,num=num,mode=mode)
    print margin[0]
    #temp = []
    #for k in range(K):
    #    for j in range(len(margin[k])):
    #        temp.extend([margin[k][j]]*numKarr[k][j])
    #temp = np.array(temp)
    #count0L = []
    #for m in mth:
    #   count0L.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(rmargin[j][0])
    temp = np.array(temp)
    countr1L = []
    for m in mth:
        countr1L.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(rmargin[j][1])
    temp = np.array(temp)
    countr2L = []
    for m in mth:
        countr2L.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(smargin[j][0])
    temp = np.array(temp)
    counts1L = []
    for m in mth:
        counts1L.append(np.sum(temp>=m))
    temp = []
    for j in range(K):
        temp.extend(smargin[j][1])
    temp = np.array(temp)
    counts2L = []
    for m in mth:
        counts2L.append(np.sum(temp>=m))
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
    #ax.legend(loc=2,frameon=False)
    ax.set_xlabel('number of fields ($K$)')
    ax.set_ylabel('realizable fraction')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # B
    ax = fig.add_subplot(324)
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
    ax.plot(mth,np.array(countr1L)/temp,color='m',lw=1.5) #1f77b4
    ax.plot(mth,np.array(counts1L)/temp,color='b',lw=1.5)    #ff7f0e
    if 0 and num > 1: # remove
        ax.plot(mth,np.array(countr2L)/temp,'--',color='#1f77b4')
        ax.plot(mth,np.array(counts2L)/temp,'--',color='#ff7f0e')
    ax.plot([0,1],2*[1],'k--',lw=1)
    ax.set_ylim(0,1)
    ax.set_xlim(0,0.4)
    ax.set_xlabel('margin $\kappa$')
    ax.set_ylabel('CDF')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # CD
    color = ['b','g','r','c','m']
    l = [31,43] #[2,3]
    K = 6
    #inp = [0,30,sum(l),100]  # first entry is zero
    inp = [0,100]  # first entry is zero
    num = 1000
    numr = len(inp)
    u = act_mat_grid_binary(l)
    if 0:
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
            with open('fig4CD_'+str(id)+'.txt','wb') as f:
                pickle.dump((margin_spatial,margin_new),f)
    color = ['b','#0f9b8e','#0cfc73']
    for id in range(2):
        ax = pylab.subplot(3,2,5+id)
        with open('fig4CD_'+str(id)+'.txt','r') as f:
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
            #sns.violinplot(np.append(kmat1,kmat2),np.append(mmat1,mmat2),np.append(nmat1,nmat2),inner=None,linewidth=.4,bw=.2,gridsize=100)
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

    """
    # Not sure whether this will be kept
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
    """
    """
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
    ax.spines['top'].set_visible(False)"""
    fig.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.1,wspace=0.4,hspace=0.5)
    fig.savefig('f4'+'qp'*is_qp+'.svg')
    """
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
    """
    #inset.subplots_adjust(left=0.1,top=0.95,right=0.95,bottom=0.1,wspace=0.3,hspace=0.3)
    #inset.savefig('f4b'+'qp'*is_qp+'.svg')

#def extra_fig4():
if 0:
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


def morefig5b():
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
    pylab.savefig('morefig5b.svg')

def morefig5c():
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
    pylab.savefig('morefig5c.svg')

def morefig5d():
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
    pylab.savefig('morefig5d.svg')

def fieldloc(trace):
    x = np.diff(trace)
    y = pylab.find(x<0)
    z = np.diff(y)
    z = np.append([10],z)
    z = (z > 1)
    peakloc = y*z
    return peakloc[peakloc>0]

def fig6(seed):
    return 0
    
if 0:
    seed = 8
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
    nadd = 1 #4
    nex = 1#0
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
                    """
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
                    print alpha,np.dot(v[iN,:,loc],w1),th
                    ax = pylab.subplot(514) # after learning
                    a1[iN,:] = np.dot(w1,v[iN,:,:])   # a1 is activity after learning
                    pylab.plot(a1[iN,:])
                    pylab.plot([0,R],[th]*2,'k',lw=1)
                    for id in idarr:
                        pylab.plot([id]*2,[np.min(a[iN,:]),np.max(a[iN,:])],'r--',lw=1)
                    for id in loc:
                        pylab.plot([id]*2,[np.min(a1[iN,:]),np.max(a1[iN,:])],'g--',lw=1)
                    ax = pylab.subplot(515) # after learning thresholded
                    ath1 = a1[iN,:]-th
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
                    nfieldstay[K-1][k-1].append(temp)"""
                    #pylab.savefig('./fig6/seed{}pc{}K{}A{}eg{}.svg'.format(seed,iN,K,k,j))
                    #pylab.close('all')
    """
    nK = 4
    nadd = 4
    #with open('./fig6/zfieldchange_seed'+str(seed)+'.txt','wb') as f:
    #    pickle.dump((fieldshift,nfieldchange,nfieldstay),f)"""

#def readfig6():
if 0:
    wperturb = 0
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
        if not wperturb:
            for jj in range(nadd):
                nfieldchange[j].append([])
                nfieldstay[j].append([])
    for seed in range(1,11):
        if wperturb:
            with open('./fig6/wperturb_fieldchange_seed'+str(seed)+'.txt','rb') as f:
                a,b,c = pickle.load(f)
        else:
            with open('./fig6/fieldchange_seed'+str(seed)+'.txt','rb') as f:
                a,b,c = pickle.load(f)
        fieldshift.extend(a)
        for j in range(nK):
            if wperturb:
                nfieldchange[j].extend(b[j])
                nfieldstay[j].extend(c[j])
            else:
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
        if wperturb:
            print 'XXX'
        else:
            for jj in range(nadd):
                meanfieldchange[j,jj] = np.mean(nfieldchange[j][jj])
    """
    color = ['b','g','r','c','m']
    for j in range(nK):
        if wperturb:
            print 'XXX'
        else:
            for jj in range(nadd):
                pylab.plot([jj+1]*len(nfieldchange[j][jj]),nfieldchange[j][jj],'x',c=color[j])
        pylab.plot(range(1,nadd+1),meanfieldchange[j,:],'o-',c=color[j],label='$K$='+str(j+1))
    """
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

def fig6_perturb(seed):
    rng = np.random.RandomState(seed)
    wp1 = 0
    wp2 = 1
    bin = 1
    #l = [31,43,59]
    l = [31,43]
    ori = [0.39,0.46]
    Ng = 600
    Np = 10
    sig = 0.16
    R = l[0]
    for j in range(len(l)-1):
        R *= l[j+1]
    R = 2000
    nK = 4
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
            for j in range(nex):  # total number of examples at difficult target locations
                print 'Example '+str(j)
                if np.mod(j,5) == 0:
                    pylab.figure(figsize=[6,7])
                ax = pylab.subplot(5,1,np.mod(j,5)+1)
                dw = rng.randn(Ng)*0.1*np.std(w[iN,:])
                w1 = w[iN,:] + dw
                w1[w1<0] = 0
                w1 /= np.sum(w1)
                a1 = np.dot(w1,v[iN,:,:])
                ath = a1 - th
                ath[ath<0] = 0
                pylab.plot(ath)
                for id in idarr:
                    pylab.plot([id]*2,[np.min(ath),np.max(ath)],'r--',lw=1)
                if np.mod(j,5) == 4:
                    pylab.savefig('./fig6/wperturb_seed{}pc{}K{}eg{}.svg'.format(seed,iN,K,j))
                pylab.close('all')
                loc1 = fieldloc(ath)
                for id in idarr:
                    fieldshift.extend(loc1-id)
                nfieldchange[K-1].append(loc1.size-np.sum(ath[idarr]>0)+np.sum(ath[idarr]==0))
                temp = 0
                for id in idarr:
                    if np.any(id==loc1):
                        temp += 1
                nfieldstay[K-1].append(temp)
    with open('./fig6/wperturb_fieldchange_seed'+str(seed)+'.txt','wb') as f:
        pickle.dump((fieldshift,nfieldchange,nfieldstay),f)

#from multiprocessing import Pool
#p = Pool(4)
#p.map(fig6,range(100))

def read_pcorr(l):
    print 'Please test!'
    eps = 1e-10
    R = 20
    p = []
    for j in range(np.sum(l)-len(l)+1):
        p.append([1]*(j+1))
    fname = 'pcorr_l'+str(l)+'K4'
    with open(fname+'.txt','rb') as f:
        pcorr = pickle.load(f)
    for X in range(9,R+1):
        p0 = np.copy(pcorr[X-9])
        p0 = list(p0)
        if X > 9:
            fname2 = 'pcorr_'+str(l)+'X'+str(X)
            with open(fname2+'.txt','rb') as f:
                pcorr2 = pickle.load(f)
            p0.extend(pcorr2)
        temp = np.copy(p0)
        temp = list(temp)
        temp.reverse()
        p0.extend(temp[np.mod(X+1,2):])
        p0.append(1)
        p.append(p0)
    psum = []
    pall = []
    for X in range(1,R+1):
        lsp = []
        lspK = []
        nCrsum = 1
        for K in range(1,X+1):
            nlsp = p[X-1][K-1]*nCr(X,K)
            if np.abs(nlsp-np.round(nlsp)) < eps:
                lspK.append(int(np.round(nlsp)))
                nCrsum += nCr(X,K)
            else:
                print 'Please check',X,K,nlsp,np.abs(nlsp-np.round(nlsp))
        lsp.append(lspK)
        psum.append(np.sum(lspK)+1)
        pall.append(psum[-1]/float(nCrsum))
        print nCrsum
    print psum
    return pall

def gridvscover():
    pylab.figure(figsize=[10,6])
    for j in [1]:#range(3):
        if j == 0:
            l = np.array([2,3])
            R = 6
        elif j == 1:
            l = np.array([3,4])
            R = 12
        elif j == 2:
            l = np.array([2,3,5])
            R = 20
        N = len(l)
        u = []
        for iN in range(N):
            for iM in range(l[iN]):
                u.append(grid(np.arange(R),l[iN],float(iM)/l[iN],0,'del'))
        u = np.array(u)
        ax = pylab.subplot(2,3,j+1)
        if j == 2:
            pall = read_pcorr(l)
        else:
            pall,p2,p3,p4 = frac_vs_S(l,R)
        pylab.plot(range(1,R+1),pall,'k',label='grid')
        pcover = []
        prank = []
        for x in range(1,R+1):
            pcover.append(cover(sum(l),x)/float(2**x))
            prank.append(cover(Sc,x)/float(2**x))
        pall = np.array(pall)
        pcover = np.array(pcover)
        prank = np.array(prank)
        pylab.plot(range(1,R+1),pcover,'b',label='random ($N$)')
        pylab.plot(range(1,R+1),prank,'g',label='random(rank)')
        pylab.legend(loc=3,frameon=False)
        pylab.plot([1,R],[0,0],'k--',lw=1)
        pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
        pylab.plot([1,R],[1,1],'k--',lw=1)
        pylab.plot([Sc]*2,[0,1],'k--',lw=1)
        pylab.xticks([1,Sc,np.sum(l),2*np.sum(l)],('1','$S_c$','$N$','2$N$'))
        pylab.yticks(np.arange(0,1.1,0.5))
        pylab.xlim(0.5,R+0.5)
        pylab.ylabel('Fraction')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax = pylab.subplot(2,3,j+4)
        pylab.plot(range(1,R+1),pall/pcover,'b',label='random ($N$)')
        pylab.plot(range(1,R+1),prank/pcover,'g',label='random (rank)')
        pylab.legend(loc=3,frameon=False)
        pylab.plot([1,R],[0,0],'k--',lw=1)
        pylab.plot([1,R],[0.5,0.5],'k--',lw=1)
        pylab.plot([1,R],[1,1],'k--',lw=1)
        pylab.plot([Sc]*2,[0,1],'k--',lw=1)
        pylab.xticks([1,Sc,np.sum(l),2*np.sum(l)],('1','$S_c$','$N$','2$N$'))
        pylab.yticks(np.arange(0,1.1,0.5))
        pylab.xlim(0.5,R+0.5)
        pylab.xlabel('Length of track')
        pylab.ylabel('Ratio')
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

#def margin4diffcode():
if 0:
    rng = np.random.RandomState(4)
    l = np.array([2,3])
    N = len(l)
    Ng = np.sum(l)
    sig = 0.212
    R = l[0]
    for j in range(N-1):
        R = lcm(R,l[j+1])
    gridmodel = 'del'
    u = []
    for iN in range(N):
        for iM in range(l[iN]):
            u.append(grid(np.arange(np.max(R)),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel))
    u = np.array(u)
    mat =[['A','B','C','D','E','F'],['+','-','-','-','-','-'],['+','+','-','-','-','-'],['+','-','+','-','-','-'],['+','-','-','+','-','-'],['+','+','+','-','-','-'],['+','+','-','+','-','-'],['+','+','-','-','+','-'],['+','-','+','-','+','-']]
    for j in range(8):
        pylab.plot([0,R],np.ones(2)-j*1.5,c='k',lw=1)
        pylab.plot([0,R],np.zeros(2)-j*1.5,c='k',lw=1)
        for k in range(R+1):
            pylab.plot([k]*2,np.array([0,1])-j*1.5,c='k',lw=1)
    for j in range(9):
        for k in range(R):
            pylab.text(k+0.32,1.7-j*1.5,mat[j][k],fontsize=15)
    pylab.ylim(-12,2)
    pylab.xlim(-0.1,R+5)
    pylab.axis('off')
    print '#### Binary grid ####'
    print u.shape
    marr,darr,karr = input_margin(u)
    narr = [5,5]
    gridmodel = 'gau'
    for j in range(len(narr)):
        v = []
        if j == 0:
            for iN in range(N):
                for iM in range(l[iN]):
                    v.append(grid(np.arange(R),l[iN],float(iM)/l[iN],phsh=0.,gridmodel=gridmodel,sig=sig))
        else:
            for iN in range(N):
                for iM in range(narr[j]):
                    v.append(grid(np.arange(R),l[iN],float(iM)/narr[j],phsh=0.,gridmodel=gridmodel,sig=sig))
        v = np.array(v)
        print '#### Gaussian grid regular phases - '+str(narr[j])+' ####'
        print v.shape
        marr,darr,karr = input_margin(v)
        v = []
        if j == 0:
            for iN in range(N):
                for iM in range(l[iN]):
                    v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
        else:
            for iN in range(N):
                for iM in range(narr[j]):
                    v.append(grid(np.arange(R),l[iN],rng.rand(),phsh=0.,gridmodel=gridmodel,sig=sig))
        v = np.array(v)
        print '#### Gaussian grid random phases - '+str(narr[j])+' ####'
        print v.shape
        marr,darr,karr = input_margin(v)

def fieldautocorr(l,K=10,Np=1):
    ac = np.zeros(l[0]*l[1]-1)
    ifi = []
    rng = np.random.RandomState(102)
    for j in range(Np):
        # Young diagram
        mat = np.zeros((l[0],l[1]))
        p,temp = np.histogram(rng.rand(K),rng.randint(1,l[1]+1))
        p = list(p[p>0])
        p.sort(reverse=True)
        for k in range(len(p)):
            mat[:p[k],k] = 1
        rng.shuffle(mat)
        mat = mat.T
        rng.shuffle(mat)
        mat = mat.T
        i1 = np.tile(range(l[0]),l[1])
        i2 = np.tile(range(l[1]),l[0])
        Y = mat[i1,i2]
        ifi.extend(np.diff(np.where(Y==1)[0]))
        if 1:
            yloc = np.where(Y==1)[0]
            print yloc,perceptron(act_mat_grid_binary(l),yloc)
        #acorr = pylab.xcorr(Y,Y,normed=False,maxlags=l[0]*l[1]-1)
        #ac += acorr[1][l[0]*l[1]:] # for pylab.xcorr
        acorr = np.correlate(Y,Y,'full')
        ac += acorr[l[0]*l[1]:]  # for np.correlate
    pylab.figure(figsize=[8,4])
    ax = pylab.subplot(121)
    for j in range(1,l[1]):
        pylab.plot(2*[j*l[0]],[0,np.max(ac[1:])],'--',c=color[0],lw=1)
    for j in range(1,l[0]):
        pylab.plot(2*[j*l[1]],[0,np.max(ac[1:])],'--',c=color[1],lw=1)
    pylab.bar(range(1,l[0]*l[1]),ac,color='k',width=1)
    pylab.ylim(0,np.max(ac)+2)
    pylab.xlim(-0.5,np.min([200,l[0]*l[1]]))
    pylab.xlabel('spatial lag')
    pylab.ylabel('autocorrelation')
    pylab.title('l='+str(l)+'; K='+str(K))
    pylab.subplot(122)
    for j in range(1,l[-1]):
        pylab.plot(2*[j*l[0]],[0,np.max(ac[1:])],'--',c=color[0],lw=1)
        pylab.plot(2*[j*l[1]],[0,np.max(ac[1:])],'--',c=color[1],lw=1)
    pylab.hist(ifi,np.arange(0.5,np.max(ifi)+2),color='k')
    pylab.xlabel('IFI')
    pylab.xlim(0,np.min([200,np.max(ifi)+2]))
    #pylab.ylim(0,8)
    #pylab.savefig('acf_ifi_l'+str(l)+'K'+str(K)+'.png')

def peak_matching():
    l = [31,43] #[83,113]
    K = 17
    Np = 10
    u = act_mat_grid_binary(l)
    marr = []
    match1 = []
    match2 = []
    match3 = []
    matchifi = []
    matchifiall = []
    rng = np.random.RandomState(102)
    for j in range(Np):
        # Young diagram
        mat = np.zeros((l[0],l[1]))
        p,temp = np.histogram(rng.rand(K),rng.randint(1,l[1]+1))
        p = list(p[p>0])
        p.sort(reverse=True)
        for k in range(len(p)):
            mat[:p[k],k] = 1
        pylab.figure()
        pylab.imshow(mat,aspect='auto')
        rng.shuffle(mat)
        mat = mat.T
        rng.shuffle(mat)
        mat = mat.T
        i1 = np.tile(range(l[0]),l[1])
        i2 = np.tile(range(l[1]),l[0])
        Y = mat[i1,i2]
        yloc = pylab.find(Y==1)
        ifi = np.diff(yloc)
        count = np.sum(np.mod(ifi,l[0])==0)+np.sum(np.mod(ifi,l[1])==0)
        matchifi.append(count)
        ifiall = list(ifi)
        for k in range(2,K):
            for i in range(K-k):
                ifiall.append(yloc[k+i]-yloc[i])
        count = np.sum(np.mod(ifiall,l[0])==0)+np.sum(np.mod(ifiall,l[1])==0)
        matchifiall.append(count)
        m,w,th = svm_margin(u.T,Y)
        print yloc,perceptron(act_mat_grid_binary(l),yloc),m
        w0 = np.copy(w)
        idx = []
        nmatch = []
        for k in range(np.sum(w>0)):
            idx.append(np.argmax(w0))
            nmatch.append(int(np.sum(u[idx[k]]*Y)))
            w0[np.argmax(w0)] = 0
        marr.append(m)
        match1.append(nmatch[0])
        match2.append(np.sum(nmatch[:2]))
        match3.append(np.sum(nmatch[:3]))
        if 1:
            pylab.figure()
            ax = pylab.subplot(111)
            for k in range(len(idx)):
                pylab.plot(range(l[0]*l[1]),u[idx[k],:]*0.8+k,'b+')
                pylab.plot(range(l[0]*l[1]),Y*0.8+k,'rx')
            pylab.yticks(range(len(idx)),idx)
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
            pylab.xlabel('location')
            pylab.ylabel('grid cell (blue)/place cell (red)')
            pylab.title('margin = '+str(m)+'; '+str(nmatch))
    pylab.figure()
    ax = pylab.subplot(221)
    pylab.plot(marr,match1,'+')
    pylab.ylabel('# matching fields')
    pylab.title('1 neuron with largest w')
    ax = pylab.subplot(222)
    pylab.plot(marr,match2,'+')
    pylab.ylabel('# matching fields')
    pylab.title('2 neurons with largest w')
    ax = pylab.subplot(223)
    pylab.plot(marr,match3,'+')
    pylab.ylabel('# matching fields')
    pylab.xlabel('margin')
    pylab.title('3 neurons with largest w')
    ax = pylab.subplot(224)
    pylab.plot(marr,matchifi,'+')
    pylab.plot(marr,matchifiall,'+')
    pylab.xlabel('margin')
    pylab.ylabel('# matching IFIs')
    pylab.suptitle('l='+str(l)+'; K='+str(K)+'; N='+str(Np))
    pylab.subplots_adjust(wspace=0.4,hspace=0.4)
    pylab.savefig('fieldmatch'+str(l)+'K'+str(K)+'N'+str(Np)+'.png')

#def codesfullrange():
if 0:
    ng = [2,4,8,46,32,1066,4718,41506]  # n=1-5 (no threshold)
    nb = [14,104,1882,94572,15028134,8378070864] # n=2-8 (with threshold so N=3-9)
    nu = [2,4,8,16,32,64,128,256]   # n=1-8 (no threshold)
    pylab.figure()
    ax = pylab.subplot(111)
    pylab.semilogy(range(1,9),ng,'o-',label='grid')
    pylab.semilogy(range(1,9),nu,'o-',label='unary')
    pylab.semilogy(range(3,9),nb,'o-',label='binary')
    pylab.legend(loc=2,frameon=False)
    pylab.xlabel('dimension')
    pylab.ylabel('number of realizable dichotomies')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    pylab.savefig('codesfullrange.svg')
    # Grid
    """
    l_list = [[3,4],[3,5],[4,5]]
    for q in range(len(l_list)):
        l = l_list[q]
        R = l[0]
        for j in range(len(l)-1):
            R = lcm(R,l[j+1])
        count = (1+R)*2   # 1+sum(l)-1
        u = act_mat_grid_binary(l)
        for K in range(2,R/2+1):
            print '----'
            print 'R = '+str(R)+'; K = '+str(K)
            com = [list(temp) for temp in itertools.combinations(range(R),K)]
            for ic in range(len(com)):
                Y = -np.ones(R)
                Y[com[ic]] = 1
                try:
                    m,w = svm_qp(u,Y,is_thre=0,is_wconstrained=0)
                    dec = np.sign(np.dot(w.T,u))  # minus here
                    if np.sum(np.abs(Y-dec))==0:
                        count += 2
                        if K==R/2 and np.mod(R,2) == 0:
                            count -= 1
                    print com[ic],1
                except:
                    print com[ic]
        ng.append(count)
        """
    # Binary
    """
    for Ng in range(4,5):
        R = 2**Ng
        count = (1+R)*2   # 1+sum(l)-1
        bc = np.zeros((Ng,R))
        for j in range(1,R):
            temp = bin(j)[2:].zfill(Ng)
            for n in range(Ng):
                bc[n,j] = int(temp[n])
        for K in range(2,R/2+1):
            print '----'
            print 'R = '+str(R)+'; K = '+str(K)
            com = [list(temp) for temp in itertools.combinations(range(R),K)]
            for ic in range(len(com)):
                Y = -np.ones(R)
                Y[com[ic]] = 1
                try:
                    m,w,b = svm_qp(bc,Y,is_thre=1,is_wconstrained=0)
                    dec = np.sign(np.dot(w.T,bc)-b)  # minus here
                    print Y,dec,np.dot(w.T,bc),b
                    if np.sum(np.abs(Y-dec))==0:
                        count += 2
                        if K==R/2 and np.mod(R,2) == 0:
                            count -= 1
                    print com[ic],1
                except:
                    print com[ic]
        nb.append(count)
        """

if 0:   # 2D # dump
    pylab.figure()
    l = [2,3]
    N = l[0]**2+l[1]**2
    Sc = N-1
    R = 6
    #pall = [1,1,1,1,?,?]
    #pylab.plot(range(1,R+1)**2,pall,'ko-',ms=5,label='grid')
    pcover = []
    for j in range(R):
        x = (j+1)**2
        pcover.append(cover(Sc,x)/float(2**x))
    pylab.plot(np.arange(1,R+1,1)**2,pcover,'o-',c='c',ms=5,label='random')
    pylab.plot([16]*2,[0,1.05],'y')
    pylab.plot([16]*2,[0,1],'k--',lw=1)
    pylab.plot([36]*2,[0,1],'k--',lw=1)
    pylab.legend(loc=6,frameon=False)
    pylab.ylim(0,1.05)
    pylab.xticks([1,4,9,16,25,36])
    pylab.yticks(np.arange(0,1.1,0.5))
    pylab.text(13,0.05,'$R_{copr}$',color='y')
    pylab.text(17,0.05,'$A^*$')
    pylab.text(34,0.05,'$A$')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def Stirling2(n,k):
    num = 0
    for i in range(k+1):
        num += (-1)**i*nCr(k,i)*(k-i)**n
    return num/math.factorial(k)

def PolyBernoulli(n,k):
    num = 0
    for m in range(n+1):
        num += (-1)**(m+n)*math.factorial(m)*Stirling2(n,m)*(m+1)**k
    return num

# def margin_gridrandom():
if 0:
    color = ['b','g','r','c','m']
    l = [31,43] #[2,3]
    K = 8
    num = 10
    mode = 's'   # s,d
    inp = [0,10,36,74,100]  # first entry is zero
    numr = len(inp)
    u = act_mat_grid_binary(l)
    rng = np.random.RandomState(1)
    margin = []
    for n in range(num):
        for r in range(numr): # from no addition to numr additions
            if n == 0:
                margin.append([])
            if n > 0 and r == 0:
                continue
            if inp[r] > 0:
                if mode == 'd':
                    # uniform
                    #randinp = 0.1*rng.rand(inp[r],u.shape[1])
                    # gaussian
                    randinp = 0.1*rng.randn(inp[r],u.shape[1])
                elif mode == 's':
                    randinp = np.zeros((inp[r],u.shape[1]))
                    for j in range(inp[r]):
                        #randinp[j,rng.choice(range(u.shape[1]),10,replace=False)] = 1
                        randinp[j,rng.choice(range(u.shape[1]),2,replace=False)] = 1
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
                    margin[r].append([])
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
                        margin[r][k-1].append(m)
                        dec = np.sign(np.dot(w.T,v)+b)
                        dec[dec<0] = 0
                        print k,p,r,abs(np.sum(np.abs(Y-dec))),m
    #pylab.figure()
    pylab.subplot(224)
    for k in range(1,K+1):
        for r in range(numr):
            if r > 0:
                # mean
                if 1:
                    pn = len(margin[r][k-1])/num
                    for j in range(pn):
                        pylab.plot([k],[np.mean(margin[r][k-1][j::pn])],'x',c=color[r-1],label=str(inp[r])+' random input')
                # all trials
                if 0:
                    pylab.plot([k]*len(margin[r][k-1]),margin[r][k-1],'x',c=color[r-1],label=str(inp[r])+' random input')
            else:
                for mu in margin[0][k-1]:
                    pylab.plot(k+np.array([-0.2,0.2]),2*[mu],'k',label='grid')
        if k == 1:
            pylab.legend(loc=1)
    pylab.xlim(0.5,K+0.5)
    #pylab.title('Dense, weak, uniform, L1 norm')
    #pylab.title('Dense, weak, Gaussian, L2 norm')
    #pylab.title('Sparse, strong, L1 norm')
    pylab.title('Very sparse, strong, L1 norm')
    #pylab.title('$\lambda$='+str(l)+'; grid std={:.2f}; rand std={:.2f}'.format(np.std(u),np.std(v)))
    pylab.xlabel('number of fields $K$')
    pylab.ylabel('margin $\kappa$')

#fig1()
#fig2()
#fig3()
#fig4()
#fig5()
#morefig4b()
#morefig4c()
#morefig4d()
pylab.show()

def number_realizable_asymptotic(l):
    return math.factorial(2*l)/(np.log(2)*np.sqrt(1-np.log(2)))/(2*np.log(2))**(2*l)

def run_number_realizable_full_range():
    lam_arr = np.arange(10,71,10)
    grid_count = []
    rand_count = []
    for lam in lam_arr:
        grid_count.append(number_realizable_asymptotic(lam))
        rand_count.append(cover(2*lam,lam**2))
    pylab.semilogy(lam_arr,grid_count)
    pylab.semilogy(lam_arr,rand_count)
    pylab.semilogy(lam_arr,2.**(lam_arr**2),'k')
