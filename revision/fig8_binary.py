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
#exec(open('gridplacefunc.py').read())
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

def phase(x,period):
    """phase(x,period) returns the phase of location x with respect to the module with spacing period."""
    return np.mod(x/period,1)

def grid(x,period,prefphase,phsh=0.,gridmodel='gau',sig=0.16):
    """grid(x,period,prefphase,phsh) returns the grid cell activity with prefphase at all location x."""
    if gridmodel == 'gau':
        temp_array = np.array((abs(phase(x-phsh*period,period)-prefphase),1-abs(phase(x-phsh*period,period)-prefphase)))
        temp = np.exp(-np.min(temp_array,axis=0)**2/(2*sig**2))
    elif gridmodel == 'del':
        temp = np.zeros(x.size)
        temp[int(np.round((phsh+prefphase)*period)):x.size:period] = 1
    return temp

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

def act_mat_grid_gau(l,R=None):
    if R == None:
        R = l[0]
        for j in range(len(l) - 1):
            R = lcm(R, l[j + 1])
    u = []
    for iN in range(len(l)):
        for iM in range(l[iN]):
            u.append(grid(np.arange(R), l[iN], float(iM) / l[iN], 0))
    return np.array(u)

def margin_gridvsrandom(l,K=6,num=10,mode='ext',is_qp=0): # ext=exact, sX=sample X without replacement
    u = act_mat_grid_binary(l)/len(l)
    #u = act_mat_grid_gau(l)
    #u /= sum(u[:,0])
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
                if is_qp == 0:
                    m,w,b = svm_margin(u.T,Y)
                    dec = np.sign(np.dot(w.T,u)+b)
                    dec[dec<0] = 0
                else:
                    try:
                        Y[Y==0] = -1
                        m,w,b = svm_qp(u,Y,1,1)
                    except:
                        m = np.inf
                        w = np.inf*np.ones(u.shape[0])
                        b = np.inf
                    dec = np.sign(np.dot(w.T,u)+b)
                margin[k-1].append(m)
                denominator = math.factorial(l[0]-p[0])*math.factorial(p[-1])*math.factorial(l[1]-len(p))
                for j in np.diff(p):
                    denominator *= math.factorial(abs(j))
                (chist,temp) = np.histogram(p,np.arange(0.5,p[0]+1))
                chist = chist[chist>0]
                for j in chist:
                    denominator *= math.factorial(j)
                numK = math.factorial(l[0])*math.factorial(l[1])/denominator
                numKarr[k-1].append(numK)
        partfunc.append(len(part))
    print(margin)
    # random
    for j in range(num):
        print('Random '+str(j))
        v = rng.rand(u.shape[0],u.shape[1])
        for jj in range(u.shape[1]):
            v[:,jj] = v[:,jj]/np.sum(v[:,jj])
        for k in range(1,K+1):
            print('Number of fields: '+str(k))
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
                if is_qp == 0:
                    m,w,b = svm_margin(v.T,Y)
                    dec = np.sign(np.dot(w.T,v)+b)
                    dec[dec<0] = 0
                else:
                    try:
                        Y[Y==0] = -1
                        m,w,b = svm_qp(v,Y,1,1)
                    except:
                        m = np.inf
                        w = np.inf*np.ones(v.shape[0])
                        b = np.inf
                    dec = np.sign(np.dot(w.T,v)+b)
                if abs(np.sum(np.abs(Y-dec))) < 1e-6:
                    numK += 1
                    rmargin[k-1][j].append(m)
            rnumKarr[k-1].append(numK)
            if j == num-1:
                print(m)
    # shuffled
    for j in range(num):
        print('Shuffled '+str(j))
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
            print('Number of fields: '+str(k))
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
                if is_qp == 0:
                    m,w,b = svm_margin(v.T,Y)
                    dec = np.sign(np.dot(w.T,v)+b)
                    dec[dec<0] = 0
                else:
                    try:
                        Y[Y==0] = -1
                        m,w,b = svm_qp(v,Y,1,1)
                    except:
                        m = np.inf
                        w = np.inf*np.ones(v.shape[0])
                        b = np.inf
                    dec = np.sign(np.dot(w.T,v)+b)
                if abs(np.sum(np.abs(Y-dec))) < 1e-6:
                    numK += 1
                    smargin[k-1][j].append(m)
            snumKarr[k-1].append(numK)
            if j == num-1:
                print(m)
    with open(datapath+'f8_'+mode+'_qp'*is_qp+'_gau.txt','wb') as f:
        pickle.dump((margin,rmargin,smargin,numKarr,rnumKarr,snumKarr),f)
    return margin,rmargin,smargin,numKarr,rnumKarr,snumKarr

def fig8gau():
    is_qp = 0   # use quadratic programming with the options of constrained weights
    if is_qp:
        print('Using quadratic programming')
    else:
        print('Using sklearn SVM')
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
    u = act_mat_grid_binary(l)/len(l)
    #u = act_mat_grid_gau(l)
    #u /= sum(u[:,0])
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
        print(np.array([0.]+list(np.cumsum(numKarr[k-1]))+[nCr(R,k)]),count,m_inset)
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
    pylab.savefig(figpath+'fig8.pdf')


#fig1_2()
#fig3()
#fig5()
#fig5()
#fig6()
#fig6b()
fig8gau()
#for j in range(1,11):
#    fig7(j)
#readfig7()
