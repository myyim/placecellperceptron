import numpy as np
import pylab
import matplotlib as mpl
import scipy.stats
import scipy.optimize
import time
import pickle
import os.path
exec(open('para_v2.py').read())

def phase(x,period):
    """phase(x,period) returns the phase of location x with respect to the module with spacing period. The phase is between 0 and 1."""
    return np.mod(x/float(period),1)

def grid(x,period,prefphase,sig=sig):
    """grid(x,period,prefphase,sig=sig) returns the 1D grid cell activity with prefphase at all location x. sig*period characterizes the std of the gaussian profile."""
    temp_array = np.array((abs(phase(x,period)-prefphase),1-abs(phase(x,period)-prefphase)))
    return np.exp(-np.min(temp_array,axis=0)**2/(2*sig**2))

def grid1d_orient(x,period,ori,xph=0.,yph=0.,sig=sig):
    """grid1d_orient(x,period,ori,xph=0.,yph=0.,sig=sig) returns the 1D slice of grid cell activity with certain period, orientation ori and phase (xph,yph) at all locations x. sig*period characterizes the std of the gaussian profile."""
    r = np.zeros(x.size)
    xv = x*np.cos(ori)
    yv = x*np.sin(ori)
    for n in range(-1,int(np.ceil(2*yv[-1]/(np.sqrt(3)*period)))+3):
        for m in range(-n//2-1,int(np.ceil(xv[-1]/period-n/2)+3)):
            r += np.exp((-(xv-(m-xph+(n-yph)/2.)*period)**2-(yv-((n-yph)*np.sqrt(3)/2)*period)**2)/(2*(sig*period)**2))
    return r

def generate_grid1d(l,M,R,sig):
    """generate_grid1d(l,M,R,sig) prints the 1D grid activity with periods l on a 1D track of length R in a file. Each module has M grid cells with regularly distributed phases. sig*period characterizes the std of the gaussian profile."""
    N = len(l)
    x = np.arange(R)
    v = np.zeros((N*M,R))
    for iN in range(N):
        for iM in range(M):
            v[iN*M+iM,:] = grid(x,l[iN],float(iM)/M,sig=sig)
    with open(datapath+'v_1d_l'+str(l)+'_M'+str(M)+'_R'+str(R)+'_sig'+str(sig)+'.txt','wb') as f1:
        pickle.dump(v,f1)

def generate_grid1d_orient(l,oriarr,M2d,R,sig):
    """generate_grid1d_orient(l,oriarr,M2d,R,sig) prints the 1D slices grid activity with periods l and orientation oriarr on a 1D track of length R in a file. Each module has M2d X M2d grid cells with regularly distributed phases. sig*period characterizes the std of the gaussian profile."""
    N = len(l)
    x = np.arange(R)
    v = np.zeros((N*M2d**2,R))
    for iN in range(N):
        for iMx in range(M2d):
            for iMy in range(M2d):
                v[iN*M+iMx*M2d+iMy,:] = grid1d_orient(x,l[iN],oriarr[iN],float(iMx)/M2d,float(iMy)/M2d,sig=sig)
    with open(datapath+'v_1d_ori_l'+str(l)+'_M'+str(M2d)+'_R'+str(R)+'_sig'+str(sig)+'.txt','wb') as f1:
        pickle.dump(v,f1)

def generate_grid1d_orient_error(l,oriarr,M2d,R,sig,nsig,seed=10):
    """generate_grid1d_orient_error(l,oriarr,M2d,R,sig,nsig) prints the 1D slices grid activity, with integration error due to noise of std nsig, with periods l and orientation oriarr on a 1D track of length R in a file. Each module has M2d X M2d grid cells with regularly distributed phases. sig*period characterizes the std of the gaussian profile."""
    rng = np.random.RandomState(seed)
    N = len(l)
    y = np.ones(R)
    y[0] = 0
    y[1:] += nsig*rng.randn(R-1)
    x = np.cumsum(y)
    v = np.zeros((N*M2d**2,R))
    for iN in range(N):
        for iMx in range(M2d):
            for iMy in range(M2d):
                v[iN*M+iMx*M2d+iMy,:] = grid1d_orient(x,l[iN],oriarr[iN],float(iMx)/M2d,float(iMy)/M2d,sig=sig)
    with open(datapath+'v_1d_ori_error_l'+str(l)+'_M'+str(M2d)+'_R'+str(R)+'_sig'+str(sig)+'_nsig'+str(nsig)+'.txt','wb') as f1:
        pickle.dump(v,f1)

def generate_spatial1d(l,M,R,sig):
    """generate_spatial1d(l,M,R,sig) prints the activity of spatial cells with gaussian profile comparable with grid cells of periods l on a 1D track of length R in a file. Each module has M cells with randomly located peaks. sig*period characterizes the std of the gaussian profile."""
    rng = np.random.RandomState(111)
    N = len(l)
    x = np.arange(R)
    v = np.zeros((N*M,R))
    for iN in range(N):
        for iM in range(M):
            v[iN*M+iM] = conv_gau(R*rng.rand(),sig*l[iN],R)
    with open(datapath+'s_1d_l'+str(l)+'_M'+str(M)+'_R'+str(R)+'_sig'+str(sig)+'.txt','wb') as f1:
        pickle.dump(v,f1)

def conv_gau(mu,sig,R):
    """conv_gau(mu,sig,R) returns a Gaussian centered at mu of amplitude 1 on a track of length R. If sig=0, the kernal is a dirac delta"""
    if sig == 0:
        y = np.zeros(int(R))
        y[int(np.round(mu))] = 1
    else:
        z = np.arange(int(R))
        y = np.exp(-(z-mu)**2/(2.*sig**2))#/np.sqrt(2.*np.pi*sig**2)
    return y

def random_weight(input,c=c,dist='lgn',wp1=wp1,wp2=wp2,conn='RDM',Np=Np,seed=seed,normalized='',wout=0):
    """random_weight(input,c=c,dist='lgn',wp1=0,wp2=1.2,conn='RDM',Np=Np,seed=seed,normalized='',wout=0) returns the sum of input into a population of place cells a. c, between 0 and 1, characterizes the connection probability from input cells to place cells. dist is the distribution of weights, which can be lognormal, uniform and absolute Gaussian distributed, with parameters wp1 and wp2 (see the code for details). conn can be 'RDM' or 'FIX', meaning the connection probability is completely random or fixed for each place cell, respectively, with the former giving rise to a larger variability. Np is the number of place cells. seed is for the random number generator to generate random weights and connectivity. normalized can be '', 'pre' or 'post'. wout=0 returns a only while wout=1 returns both a and weights w."""
    rng = np.random.RandomState(seed)
    N = input.shape[0]
    if dist == 'lgn':
        w = rng.lognormal(wp1,wp2,size=[N,Np])
    elif dist == 'uni':
        w = wp1 + wp2*rng.rand(N,Np)
    elif dist == 'gau':
        w = np.abs(wp1+wp2*rng.randn(N,Np))
    if conn == 'RDM':
        # Completely random
        wmask = np.append(np.zeros(int(np.round(N*Np*(1-c)))),np.ones(int(np.round(N*Np*c))))
        rng.shuffle(wmask)
        wmask = np.reshape(wmask,(N,Np))
    elif conn == 'FIX':
        # Constrained
        wmask = np.zeros((N,Np),dtype='int')
        for iNp in range(Np):
            wmask[rng.choice(range(N),int(np.round(N*c)),replace=0),iNp] = 1
    w *= wmask
    if normalized != '':   # normalization on the postsnaptic side
        if normalized == 'pre':
            for iN in range(N):
                if np.sum(w[iN,:]) > 0:
                    w[iN,:] /= sum(w[iN,:])
        elif normalized == 'post':
            for iNp in range(Np):
                if np.sum(w[:,iNp]) > 0:
                    w[:,iNp] /= sum(w[:,iNp])
    a = np.dot(w.transpose(),input)
    if wout == 0:
        return a
    else:
        return a,w

def learned_weight(input,c=c,conn='RDM',Np=Np,seed=seed*2,normalized='',wout=0):
    """learned_weight(input,c=c,conn='RDM',Np=Np,seed=seed*2,normalized='',wout=0) returns the sum of input into a population of place cells a through a special kind of learning. Regarding learning, weights follow one-shot Hebbian and are set to be the sum of product of pre- and post-syanptic activities. c, between 0 and 1, characterizes the connection probability from input cells to place cells. conn can be 'RDM' or 'FIX', meaning the connection probability is completely random or fixed for each place cell, respectively, with the former giving rise to a larger variability. Np is the number of place cells. seed is for the random number generator to generate connectivity. normalized can be '', 'pre' or 'post'. wout=0 returns a only while wout=1 returns both a and weights w."""
    rng = np.random.RandomState(seed)
    N = input.shape[0]
    R = input.shape[1]
    if conn == 'RDM':
        # Completely random
        wmask = np.append(np.zeros(int(np.round(N*Np*(1-c)))),np.ones(int(np.round(N*Np*c))))
        rng.shuffle(wmask)
        wmask = np.reshape(wmask,(N,Np))
    elif conn == 'FIX':
        # Constrained
        wmask = np.zeros((N,Np),dtype='int')
        for iNp in range(Np):
            wmask[rng.choice(range(N),int(N*c/100),replace=0),iNp] = 1
    cues = np.zeros((Np,R))
    for iNp in range(Np):
        cues[iNp,rng.randint(0,R,1)] = 1
    w = np.dot(input,cues.T)
    w *= wmask
    if normalized != '':   # normalization on the postsnaptic side
        if normalized == 'pre':
            for iN in range(N):
                if np.sum(w[iN,:]) > 0:
                    w[iN,:] /= sum(w[iN,:])
        elif normalized == 'post':
            for iNp in range(Np):
                if np.sum(w[:,iNp]) > 0:
                    w[:,iNp] /= sum(w[:,iNp])
    a = np.dot(w.T,input)
    if wout == 0:
        return a
    else:
        return a,w

def lnlike(params,nfield):
    """lnlike(params,nfield) returns the log likelihood of nfield given params"""
    r,p = params
    sumlog = sum(np.log(scipy.stats.nbinom.pmf(nfield,r,p)))
    #print r,p,-sumlog
    return -sumlog

def argCOM(y):
    """argCOM(y) returns the location of COM of y."""
    idx = np.round(np.sum(y/np.sum(y)*np.arange(len(y))))
    return int(idx)

def detectfield(y,nth,mode='FIX'):   # FIX,NEU
    """detectfield(y,nth,mode='FIX') returns the fields, their widths and the binary field signal given the activity y and number of std above mean as threshold. For mode='FIX', threshold is set to be  mean + nth*std of the entire population y; whereas for mode='NEU', threshold is specific to each neuron."""
    field = []
    width = []
    if mode == 'FIX':
        th_bool = (y>np.mean(y)+nth*np.std(y))*1
    elif mode == 'NEU':
        th = ((np.mean(y,1) + nth*np.std(y,1))*np.ones((y.shape[1],y.shape[0]))).T
        th_bool = (y>th)*1
    else:
        print('Select the right mode')
        return
    for iNp in range(y.shape[0]):
        track = np.diff(th_bool[iNp,:])
        idx = np.where(track==1)[0]+1
        idx1 = np.where(track==-1)[0]+1
        if th_bool[iNp,0] == 1:
            idx = np.append([0],idx)
        nfield = []
        nwidth = []
        for j in range(len(idx)):
            if j == len(idx)-1 and len(idx)!=len(idx1):
                nfield.append(idx[j]+argCOM(y[iNp,idx[j]:]))
                nwidth.append(y.shape[1]-idx[j])
            else:
                nfield.append(idx[j]+argCOM(y[iNp,idx[j]:idx1[j]]))
                nwidth.append(idx1[j]-idx[j])
        field.append(nfield)
        width.append(nwidth)
    return field,width,th_bool

"""
def placefield_stats(y,res=1,kwin=1,more=1):
    #k-winner
    field = []
    width = []
    th_bool = (y<0)*0     # all false
    #first = []
    for j in range(int(R/res)):
        y0 = np.copy(y[:,j*res:j*res+res])
        th0 = th_bool[:,j*res:j*res+res]
        for k in range(kwin):
            ymax = np.max(y0)
            idx = np.where(y0==ymax)
            y0[idx[0]] = 0
            if more == 0:
                th0[idx[0],:] = 1
            elif more == 1: #old
                th0[y0==ymax] = 1
            th_bool[:,j*res:j*res+res] = th0
    for iNp in range(y.shape[0]):
        track = np.diff(th_bool[iNp,:])
        idx = np.where(track==1)[0]+1
        idx1 = np.where(track==-1)[0]+1
        if th_bool[iNp,0] == 1: #np.any(iNp==first):
            idx = np.append([0],idx)
        nfield = []
        nwidth = []
        for j in range(len(idx)):
            if j == len(idx)-1 and len(idx)!=len(idx1):
                nfield.append(idx[j]+np.argmax(y[iNp,idx[j]:]))
                nwidth.append(y.shape[1]-idx[j])
            else:
                nfield.append(idx[j]+np.argmax(y[iNp,idx[j]:idx1[j]]))
                nwidth.append(idx1[j]-idx[j])
        field.append(nfield)
        width.append(nwidth)
    return field,width,th_bool
"""

def constrainfield(allfield,allfieldsize,th_bool,minw=15):
    """constrainfield(allfield,allfieldsize,th_bool,minw=15) returns place fields that satisfy the minimum width minw constraint, together with their widths and the binary signal of fields."""
    field = []
    width = []
    binary_bool = np.copy(th_bool)
    for j in range(len(allfield)):
        idx = np.array(allfieldsize[j])>=minw
        field.append(np.array(allfield[j])[idx].tolist())
        width.append(np.array(allfieldsize[j])[idx].tolist())
        if np.any(np.array(allfieldsize[j])<minw):
            track = np.diff(th_bool[j])
            idx = np.where(track==1)[0]+1
            idx1 = np.where(track==-1)[0]+1
            for k in np.array(allfield[j])[np.array(allfieldsize[j])<minw]:
                i1 = idx[k-idx>=0]
                if i1.size == 0:
                    i1 = 0
                else:
                    i1 = i1[-1]
                i2 = idx1[idx1-k>=0]
                if i2.size == 0:
                    i2 = th_bool[j].size
                else:
                    i2 = i2[0]
                binary_bool[i1:i2] = 0
    return field,width,binary_bool

def printpc(a,nth=nth,ncell=8,celltoprint=None):
    """printpc(a,nth=nth,ncell=8,celltoprint=None) returns a figure on activity of cells in celltoprint or ncell place cells if celltoprint is None. Their place fields are also indicated as well as the threshold for field detection. Currently a global threshold for the place cell population with mean + nth*std is applied."""
    pylab.figure(figsize=[8,ncell+1])
    if celltoprint == None:
        toprint = np.arange(ncell)
    else:
        toprint = celltoprint
    for j in range(len(toprint)):
        ax = pylab.subplot(len(toprint),1,j+1)
        pylab.plot(np.arange(a.shape[1]),a[toprint[j],:])
        allfield,allfieldsize,th_bool = detectfield(a,nth=nth,mode='FIX')
        if constrained:
            field,_,_ = constrainfield(allfield,allfieldsize,th_bool)
        th = np.mean(a) + nth*np.std(a)
        pylab.plot([0,R],[th,th],'k--',lw=1)
        if len(field) > 0:
            field = np.array(field[toprint[j]])
            pylab.plot(field,max(a[toprint[j],:])*np.ones(field.size),'bo',ms=3)
        if j == ncell-1:
            pylab.xlabel('Linearized position (m)')
            pylab.ylabel('Rate')
        else:
            pylab.xticks([])
        pylab.xlim(0,R)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    pylab.subplots_adjust(bottom=0.15)

def fieldstat(allfield,c=c,nth=nth,sort_neuron=1):
    """fieldstat(allfield,c=c,nth=nth,sort_neuron=1) returns the fitted parameters for the gamma-Poisson r_fit and p_fit, and the total number of fields sum(np.array(nfield)). A figure with the analysis as in Rich et al (2014) is also returned. """
    Np = len(allfield)
    rng = np.random.RandomState(1)
    #allfieldsize = [item for sublist in allfieldsize for item in sublist]
    #allfieldsize = np.array(allfieldsize)
    fields = [item for sublist in allfield for item in sublist]
    fields = np.array(fields)
    nfield = [len(sublist) for sublist in allfield]
    medianfield = []
    ifi = []
    allifi = []
    ifimean = []
    nfield6 = []
    cv = []
    recruitpos = []
    for iNp in range(Np):
        if nfield[iNp]>=6:
            medianfield.append(np.median(allfield[iNp]))
            ifi.append(np.median(np.diff(allfield[iNp])))
            ifimean.append(np.mean(np.diff(allfield[iNp])))
            nfield6.append(len(allfield[iNp]))
            temp = allfield[iNp]
            temp.sort()
            cv.append(np.std(np.diff(temp))/np.mean(np.diff(temp)))
        if nfield[iNp]>1:
            allifi.extend(np.diff(allfield[iNp]))
        if nfield[iNp]>0:
            recruitpos.append(allfield[iNp][0])
    if selected:
        global celltoprint
        nfieldmax = np.max(nfield)
        while len(celltoprint) < 8:
            celltoprint.extend(np.where(np.array(nfield)==nfieldmax)[0])
            nfieldmax -= 1
        celltoprint = celltoprint[:8]
    row = np.sum(np.array(nfield)==0)
    if row == Np:
        print('No place fields found')
        return
    pylab.figure(figsize=[7,10])
    ax = pylab.subplot(411)
    if sort_neuron == 1:
        j = 1
        while j < Np:
            pylab.plot([0,R/100.],j*np.ones(2),'0.8')
            j += 5
        for j in range(1,np.max(nfield)+1):
            for iNp in range(Np):
                if nfield[iNp] == j:
                    row += 1
                    pylab.plot(np.array(allfield[iNp])/100.,row*np.ones(j),'k.')
    else:
        for iNp in range(Np):
            if nfield[iNp] > 0:
                pylab.plot(np.array(allfield[iNp])/100.,(iNp+1)*np.ones(nfield[iNp]),'k.')
    pylab.xlim(0,R/100.)
    pylab.ylim(-2,Np+3)
    pylab.yticks([1,Np])
    pylab.xlabel('Linearized position (m)')
    pylab.ylabel('Neuron ID')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    pylab.subplot(437)
    pylab.hist(np.array(medianfield)/100.,np.arange(0,(R+1)/100,R/1000),color='k')
    median_bs = np.array([])
    for j in range(len(medianfield)):
        rn = rng.uniform(0,R/100.,[num_bs,nfield6[j]])
        median_bs = np.append(median_bs,np.median(rn,1))
    yfit = pylab.histogram(median_bs,np.arange(0,R/100.+R/1000.,R/1000.))
    pylab.plot(yfit[1][:-1]+R/2000.,yfit[0]/float(num_bs),'r')
    pylab.xlim(0,R/100.)
    pylab.xticks([0,20,40])
    pylab.xlabel('Median field location (m)')
    pylab.ylabel('Count')
    pylab.subplot(438)
    #pylab.hist((np.array(ifi)-float(R)/np.array(nfield6))/np.array(ifimean),np.arange(-1,1.1,0.1))
    pylab.hist((np.array(ifi)-float(R)*np.log(2.)/np.array(nfield6))/(float(R)/np.array(nfield6)),np.arange(-1,1.1,0.1),color='k')
    ratio_bs = np.array([])
    for j in range(len(ifi)):
        rn = rng.exponential(float(R)/nfield6[j],[num_bs,nfield6[j]-1])
        ratio_bs = np.append(ratio_bs,(np.median(rn,1)-float(R)*np.log(2.)/nfield6[j])/(float(R)/nfield6[j]))
    yfit = pylab.histogram(ratio_bs,np.arange(-1,1.1,0.1))
    pylab.plot(yfit[1][:-1]+0.05,yfit[0]/float(num_bs),'r')
    pylab.xlim(-1,1)
    pylab.xlabel('Interval error')
    pylab.ylabel('Count')
    pylab.subplot(434)
    nfield_hist = pylab.hist(nfield,bins=np.arange(np.max(nfield)+1)-0.5,color='k')
    # Maximum likelihood
    result = scipy.optimize.minimize(lnlike,[1,0.5],args=(nfield,),method='L-BFGS-B',bounds=((1e-6,50),(1e-6,1-1e-6)))
    r_fit,p_fit = result["x"]
    #pylab.text(np.max(nfield)/2.,np.max(nfield_hist[0])*2./3,'r = {:.2f}\np = {:.2f}\nf = {:d}\nn = {:.2f}'.format(r_fit,p_fit,np.sum(nfield),np.sum(np.array(nfield)>0)/float(Np)))
    pylab.text(np.max(nfield)/2.,np.max(nfield_hist[0])*2./3,'r = {:.2f}\np = {:.2f}'.format(r_fit,p_fit))
    pylab.plot(np.arange(0,np.max(nfield)+1,1),len(nfield)*scipy.stats.nbinom.pmf(np.arange(0,np.max(nfield)+1,1),r_fit,p_fit),'r')
    """
        # Nonlinear regression using least squares on cumulative density function
        print nfield_hist[0]
        nfield_cumsum = np.cumsum(nfield_hist[0])
        print nfield_cumsum
        # regularized incomplete beta, CDF of gamma-Poisson process
        def fun_lsq(params,k):
        return 1- scipy.special.betainc(k+1,params[0],params[1])*scipy.special.beta(k+1,params[0])
        res_lsq = scipy.optimize.leastsq(fun_lsq,[0.5,0.5],args=(nfield_cumsum))
        print res_lsq[0]
        pylab.plot(range(np.max(nfield)+1),fun_lsq(res_lsq[0],range(np.max(nfield)+1)),'r')
        """
    # Poisson distribution
    mean_nfield = np.mean(nfield)
    #pylab.plot(np.arange(0,np.max(nfield)+1,1),len(nfield)*scipy.stats.poisson.pmf(np.arange(0,np.max(nfield)+1,1),mean_nfield),'0.5')
    pylab.xlim(-0.5,np.max(nfield)+1)
    pylab.ylim(0,101)
    pylab.xticks(range(0,31,10))
    pylab.xlabel('place fields per cell')
    pylab.ylabel('no. of cells')
    pylab.subplot(435)
    count,width = pylab.histogram(np.array(recruitpos)/100.,bins=np.arange(0.,R/100.,1/100.))
    pylab.plot(width,np.append([0],np.cumsum(count)/float(Np)),'k',label='Sim')
    beta = p_fit/(1.-p_fit)
    pylab.plot(np.arange(0,R/100.,1/100.),1-(beta/(beta+np.arange(0,R/100.,1/100.)/(R/100.)))**r_fit,'r',label='GP')
    pylab.plot(np.arange(0,R/100.,1/100.),1-np.exp(-mean_nfield*np.arange(0,R/100.,1/100.)/(R/100.)),'0.5',label='Poisson')
    pylab.legend(loc=4)
    pylab.xlim(0,R/100.)
    pylab.xticks([0,20,40])
    pylab.yticks([0,0.5,1])
    pylab.ylim(0,1)
    pylab.xlabel('Recruitment position (m)')
    pylab.ylabel('Cumulative proportion')
    pylab.subplot(436)
    count,width = pylab.histogram(fields/100.,bins=np.arange(0.,R/100.,1/100.))
    pylab.plot(width,np.append([0],np.cumsum(count)/float(sum(count))),'k')
    pylab.plot([0,R/100.],[0,1],'r')
    pylab.xlim(0,R/100.)
    pylab.xticks([0,20,40])
    pylab.yticks([0,0.5,1])
    pylab.ylim(0,1)
    pylab.xlabel('Field location (m)')
    pylab.ylabel('Cumulative proportion')
    pylab.subplot(439)
    pylab.hist(cv,bins=np.arange(-0.1,2,0.2),color='k')
    pylab.xlabel('CV')
    pylab.ylabel('Count')
    """
    pylab.hist(allfieldsize,bins=np.arange(min_len-0.5,np.max(allfieldsize)+1,1),color='k')
    pylab.xlim(min_len-0.5,np.max(allfieldsize)+1)
    pylab.xlabel('Place field size (m)')
    pylab.ylabel('Count')
    """
    pylab.subplot(4,3,10)
    for j in range(10):
        allrecruitpos = []
        for iNp in range(Np):
            if len(allfield[iNp]) > 0:
                if np.max(allfield[iNp])>j*R/10.:
                    field = np.array(allfield[iNp])[np.array(allfield[iNp])>j*R/10.]
                    allrecruitpos.append(field[0])
        count,width = pylab.histogram(np.array(allrecruitpos)/100.,bins=np.arange(0.,R/100.,1/100.))
        start = sum(np.cumsum(count)==0)
        pylab.plot(width[start-1:],np.append([0],np.cumsum(count)/float(Np))[start-1:])
    pylab.xticks([0,20,40])
    pylab.yticks([0,0.5,1])
    pylab.ylim(0,1)
    pylab.xlabel('Memoryless recruitment (m)')
    pylab.ylabel('Cumulative proportion')
    pylab.subplot(4,3,11)
    for j in range(10):
        allrecruitpos = []
        for iNp in range(Np):
            if len(allfield[iNp]) > 0:
                if np.max(allfield[iNp])>j*R/10.:
                    field = np.array(allfield[iNp])[np.array(allfield[iNp])>j*R/10.]
                    allrecruitpos.append(field[0])
        count,width = pylab.histogram(np.array(allrecruitpos)/100.,bins=np.arange(0.,R/100.,1/100.))
        start = sum(np.cumsum(count)==0)
        pylab.plot(width[start-1:]-j*R/1000.,np.append([0],np.cumsum(count)/float(Np))[start-1:])
    pylab.xticks([0,20,40])
    pylab.yticks([0,0.5,1])
    pylab.ylim(0,1)
    pylab.xlabel('Shifted (m)')
    pylab.subplot(4,3,12)
    pylab.hist(allifi,bins=np.arange(-0.5,500,1),color='k')
    pylab.xlabel('IFI')
    pylab.ylabel('Count')
    """
    pylab.figure(figsize=[12,6])
    pylab.hist(allifi,bins=np.arange(-0.5,max(allifi)+1,1),color='k')
    pylab.xlabel('IFI')
    pylab.ylabel('Count')
    """
    """
    allrecruitpos = []
    for j in range(10):
        allrecruitpos.append([])
        for iNp in range(Np):
            if len(allfield[iNp])>j:
                allrecruitpos[-1].append(allfield[iNp][j])
        count,width = pylab.histogram(np.array(allrecruitpos[-1])/100.,bins=np.arange(0.,R/100.,1/100.))
        start = sum(np.cumsum(count)==0)
        pylab.plot(width[start-1:],np.append([0],np.cumsum(count)/float(Np))[start-1:])
    pylab.xticks([0,20,40])
    pylab.yticks([0,0.5,1])
    pylab.ylim(0,1)
    pylab.xlabel('$j$-th recruitment (m)')
    pylab.ylabel('Cumulative proportion')
    """
    for j in range(4,13):
        ax = pylab.subplot(4,3,j)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    pylab.subplots_adjust(bottom=0.1,hspace=0.45,wspace=0.45)
    recruitpos1 = []
    for iNp in range(Np):
        if len(allfield[iNp])>0:
            recruitpos1.append(allfield[iNp][0])
    #pylab.subplot(411)
    pylab.suptitle('M={},$\lambda$={},c={},$\sigma$={},th={},#={},f={}'.format(M,l,c,sig,nth,np.sum(np.array(nfield)),np.round(np.sum(np.array(nfield)>0)/float(Np),2)))
    #print 'Total number of place fields = '+str(sum(np.array(nfield)))
    #print 'Total number of silent neurons (out of '+str(a[:,0].size)+') = '+str(sum(np.array(nfield)==0))
    #print 'Mean first recruitment position = '+str(np.mean(recruitpos1)/len(recruitpos1))
    #print 'Fitting of gamma-Poisson: r = '+str(r_fit)+', p = '+str(p_fit) +'; f = '+str(sum(np.array(nfield)>0)/float(Np))
    return r_fit,p_fit,sum(np.array(nfield))#sum(np.array(nfield)>0)/float(Np)

def autocorr(allfield,R=R):
    """autocorr(allfield,R=R) returns the sum of autocorrelation of all field distribution on a track of length R."""
    Np = len(allfield)
    acf = np.zeros(2*R-1)
    for j in range(Np):
        y = np.zeros(R)
        y[allfield[j]] = 1
        acf += np.correlate(y,y,'full')
    return acf

def psddirect(allfield,R=R):
    """psddirect(allfield,R=R) returns the sum of PSDs of field distribution over all neurons in allfield, which is a binary COM field signal, on a track of length R."""
    Np = len(allfield)
    sdsum = np.zeros(R)
    for j in range(Np):
        y = np.zeros(R)
        y[allfield[j]] = 1
        cfft = np.fft.fft(y)
        sdsum += np.abs(cfft)**2
    return sdsum

def psdacf(allfield,R=R):
    """psdacf(allfield,R=R) returns the PSD of the sum of autocorrelation function of field distribution over all neurons in allfield, which is a binary COM field signal, on a track of length R."""
    acf = autocorr(allfield,R=R)
    acf = np.append([0],acf)
    cfft = np.fft.fft(acf)
    return np.abs(cfft)

def plotpsd(allfield,l=l,oriarr=oriarr,mode='direct',R=R):
    """plotpsd(allfield,mode='direct',R=R) returns a figure of PSD of the given field distribution allfield and its shuffled version for comparison."""
    N = len(l)
    rng = np.random.RandomState(234)
    fieldshuf = []
    for j in range(len(allfield)):
        if len(allfield[j]) >= 1:
            fieldcom = rng.rand(len(allfield[j]))*R
            fieldcom = np.around(fieldcom)
            fieldcom = fieldcom.astype(int)
            fieldcom.sort()
            fieldcom = list(fieldcom)
            fieldshuf.append(fieldcom)
    if mode == 'direct':
        sd = psddirect(allfield,R)
        sdshuf = psddirect(fieldshuf,R)
    elif mode == 'viaacf':
        sd = psdacf(allfield,R=R)
        sdshuf = psdacf(fieldshuf,R=R)
    pylab.plot(np.arange(len(sd))/len(sd),sd,'b',label='grid')
    pylab.plot(np.arange(len(sd))/len(sd),sdshuf,'0.5',label='shuffled')
    pylab.legend(loc=2)
    pylab.plot(np.arange(len(sd))/len(sd),sd,'b',label='grid') # for the legend
    # field locations
    pylab.plot(np.array(2*np.sin(oriarr)/(np.sqrt(3)*l)),np.zeros(N),'ro')
    pylab.plot(np.array(1/l*(np.cos(oriarr)-np.sin(oriarr)/np.sqrt(3))),np.zeros(N),'rx')
    pylab.plot(np.array(1/l*(np.cos(oriarr)+np.sin(oriarr)/np.sqrt(3))),np.zeros(N),'r^')
    # harmonics
    #pylab.plot(2*np.array(2*np.sin(oriarr)/(np.sqrt(3)*l)),np.zeros(N),'go')
    #pylab.plot(2*np.array(1/l*(np.cos(oriarr)-np.sin(oriarr)/np.sqrt(3))),np.zeros(N),'gx')
    #pylab.plot(2*np.array(1/l*(np.cos(oriarr)+np.sin(oriarr)/np.sqrt(3))),np.zeros(N),'g^')
    pylab.xlim(0,sd_frange)
    pylab.ylabel('PSD: '+mode)
    pylab.xlabel('spatial frequency (cm$^{-1}$)')
    sdmax = np.max(sd[10:len(sd)//2+1])
    pylab.ylim(-0.05*sdmax,1.1*sdmax)
