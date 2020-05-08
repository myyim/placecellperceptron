# This code was developed by Man Yi Yim (manyi.yim@gmail.com) under Python 2.7.13.

import numpy as np
import pylab
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import pickle
import math
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

def frac4real():
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

def fieldloc(trace):
    x = np.diff(trace)
    y = pylab.find(x<0)
    z = np.diff(y)
    z = np.append([10],z)
    z = (z > 1)
    peakloc = y*z
    return peakloc[peakloc>0]

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
            if np.abs(nlsp-np.round(nlsp)) < 1e-8:
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

def margin4diffcode():
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

def codesfullrange():
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

def margin_gridrandom():
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

def polar4fig6():
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

def LeeData():
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
    pylab.savefig('ifigroup_width'+str(sig)+'.png')
