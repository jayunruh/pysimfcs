#this module contains functions for analyzing fluorescence fluctuation data
#from tqdm.notebook import tqdm
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
import pandas as pd
import scipy.special as ss
import scipy.signal as ssig
import matplotlib.pyplot as plt

# here are some global variables or some definitions to be set for analysis
psftype='3dgaussian'
psftypes={'3dgaussian':'3D_Gaussian','3dgl2':'3D_Gaussian_Lorentzian_squared',
          '2dgaussian':'2D_Gaussian'}
#here are the 2dpch histogram lengths
pch2dsizes=[10,10]
#here are the rics fit parameters in seconds and microns
ricspixeltime=0.00001
ricspixelsize=0.05
ricslinetime=0.00128
#here are the ics fit parameters
centerx=16
centery=16
fitxpts=32
fitypts=32

##########################################
#here are the correlation functions
##########################################

def autocorr(traj):
    '''
    calcalate the fcs autocorrelation using the fft
    '''
    tfft=fft.rfft(traj)
    temp=tfft.real.flatten()**2+tfft.imag.flatten()**2
    corr=fft.irfft(temp)
    return corr/(len(traj)*(traj.mean()**2))-1

def autocorr2d(img):
    '''
    calcalate the spatial autocorrelation using the fft
    '''
    tfft=fft.rfft2(img)
    temp=tfft.real**2+tfft.imag**2
    corr=fft.irfft2(temp)
    corr=corr/(img.shape[0]*img.shape[1]*(img.mean()**2))-1
    return fft.fftshift(corr)

def carpetstics(carpet):
    '''
    calcalate the spatial autocorrelation using the fft
    undo the fftshift along the temporal axis
    '''
    return fft.fftshift(autocorr2d(carpet),axes=[0])

def paircorrelation(carpet,shift=4):
    '''
    calculate the cross corr from each line to neighbor shift lines away
    '''
    xpts=carpet.shape[-1]
    corrpts=xpts-shift
    #now calculate the pair correlation functions
    pcf=np.array([crosscorr(carpet[:,i],carpet[:,(i+shift)]) for i in range(corrpts)]).T
    return pcf

def crosscorr(traj1,traj2):
    '''
    calcalate the fcs autocorrelation using the fft
    '''
    tfft1=fft.rfft(traj1)
    tfft2=fft.rfft(traj2)
    temp=tfft1*np.conjugate(tfft2)
    corr=fft.irfft(temp)
    return corr/(len(traj1)*(traj1.mean()*traj2.mean()))-1

def crosscorr2d(img1,img2):
    '''
    calcalate the fcs autocorrelation using the fft
    '''
    tfft1=fft.rfft2(img1)
    tfft2=fft.rfft2(img2)
    temp=tfft1*np.conjugate(tfft2)
    corr=fft.irfft2(temp)
    corr=corr/(img1.shape[0]*img1.shape[1]*(img1.mean()*img2.mean()))-1
    return fft.fftshift(corr)

def avgquadrants(img):
    '''
    averages the 4 quadrants of a 2d correlation function
    assume the zero frequency is in the center
    '''
    ur=np.roll(np.fliplr(img.copy()),shift=1,axis=1)
    ll=np.roll(np.flipud(img.copy()),shift=1,axis=0)
    both=np.roll(np.flipud(np.fliplr(img.copy())),shift=(1,1),axis=(0,1))
    avgd=0.25*(img.copy()+ur+ll+both)
    return avgd

def bintraj(traj,binby):
    '''
    bin a trajectory by reshaping and averaging
    '''
    return traj.reshape(len(traj)//binby,binby).mean(axis=1)

def binmultilog(corr,tauwidth=8):
    '''
    do a modified logarithmic binning with bin size doubling every tauwidth bins
    '''
    corr2=corr.copy()
    xvals=np.arange(len(corr))
    pos=tauwidth
    pos2=pos
    binsize=2
    while(pos2<len(corr)//2):
        corr2[pos:(pos+tauwidth)]=bintraj(corr2[pos2:(pos2+tauwidth*binsize)],binsize)
        xvals[pos:(pos+tauwidth)]=np.arange(pos2,(pos2+tauwidth*binsize),binsize)
        pos2+=tauwidth*binsize
        pos+=tauwidth
        binsize*=2
    return corr2[:pos],xvals[:pos]

def carpetbml(carpetcorr):
    '''
    do a multilog binning of a carpet correlation
    '''
    carpetbml1=np.array([binmultilog(carpetcorr[:,i])[0] for i in range(carpetcorr.shape[1])]).T
    binxvals=binmultilog(carpetcorr[:,0])[1]
    return carpetbml1,binxvals

##########################################
#here is the correlation fitting function
##########################################

def fcsfunc(xvals,*params):
    '''
    returns the fcs fitting function
    params are background, zratio, G01, taud1, ...
    '''
    zr=params[1]
    g0s=[params[i] for i in range(2,len(params),2)]
    tauds=[params[i] for i in range(3,len(params),2)]
    if(psftype=='3dgaussian' or psftype=='3dgl2'):
        profile=g0s[0]/((1.0+xvals/tauds[0])*np.sqrt(1.0+xvals/(zr*zr*tauds[0])))
        for i in range(1,len(g0s)):
            tprof=g0s[i]/((1.0+xvals/tauds[i])*np.sqrt(1.0+xvals/(zr*zr*tauds[i])))
            profile+=tprof
    else:
        #here is the 2d gaussian form of things
        profile=g0s[0]/(1.0+xvals/tauds[0])
        for i in range(1,len(g0s)):
            tprof=g0s[i]/(1.0+xvals/tauds[i])
            profile+=tprof
    return profile+params[0]

##########################################
#here are the RICS (raster image correlation) fitting functions
##########################################

def ricsfunc(xvals,*params):
    '''
    returns the rics 1D fitting function for horizontal and vertical directions (combined single side profile skipping the g0)
    the global ricspixeltime, ricslinetime, ricspixelsize parameters control the fit
    xvals should be pixel distances from the center (e.g. 1,2,3,...)
    params are background, w0, zratio, G01, D1, ...
    w0 units are microns and D units are microns squared per second
    '''
    zr=params[2]
    w02=params[1]*params[1]
    z02=w02*zr*zr
    g0s=[params[i] for i in range(3,len(params),2)]
    ds=[params[i] for i in range(4,len(params),2)]
    fitsize=len(xvals)//2
    htauvals=xvals[:fitsize]*ricspixeltime
    hr2vals=(xvals[:fitsize]*ricspixelsize)**2
    vtauvals=xvals[fitsize:]*ricslinetime
    vr2vals=(xvals[:fitsize]*ricspixelsize)**2
    if(psftype=='3dgaussian'):
        tempx=4.0*htauvals*ds[0]
        hprofile=g0s[0]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02)*np.sqrt(1.0+tempx/z02))
        tempy=4.0*vtauvals*ds[0]
        vprofile=g0s[0]*np.exp(-vr2vals/(w02+tempy))/((1.0+tempy/w02)*np.sqrt(1.0+tempy/z02))
        for i in range(1,len(g0s)):
            tempx=4.0*hxvals*ds[i]                              
            thprof=g0s[i]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02)*np.sqrt(1.0+tempx/z02))
            hprofile+=thprof
            tempy=4.0*vxvals*ds[i]
            tvprof=g0s[i]*np.exp(-vr2vals/(w02+tempy))/((1.0+tempy/w02)*np.sqrt(1.0+tempy/z02))
            vprofile+=tvprof
    elif(psftype=='3dgl2'):
        #here is the gaussian lorentzian squared
        tempx=8.0*htauvals*ds[0]
        hprofile=g0s[0]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02)*np.sqrt(1.0+tempx/z02))
        tempy=8.0*vtauvals*ds[0]
        vprofile=g0s[0]*np.exp(-2.0*vr2vals/(w02+tempy))/((1.0+tempy/w02)*np.sqrt(1.0+tempy/z02))
        for i in range(1,len(g0s)):
            tempx=8.0*hxvals*ds[i]                              
            thprof=g0s[i]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02)*np.sqrt(1.0+tempx/z02))
            hprofile+=thprof
            tempy=8.0*vxvals*ds[i]
            tvprof=g0s[i]*np.exp(-2.0*vr2vals/(w02+tempy))/((1.0+tempy/w02)*np.sqrt(1.0+tempy/z02))
            vprofile+=tvprof
    else:
        #here is the 2d gaussian form of things
        tempx=4.0*htauvals*ds[0]
        hprofile=g0s[0]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02))
        tempy=4.0*vtauvals*ds[0]
        vprofile=g0s[0]*np.exp(-vr2vals/(w02+tempy))/((1.0+tempy/w02))
        for i in range(1,len(g0s)):
            tempx=4.0*hxvals*ds[i]                              
            thprof=g0s[i]*np.exp(-hr2vals/(w02+tempx))/((1.0+tempx/w02))
            hprofile+=thprof
            tempy=4.0*vxvals*ds[i]
            tvprof=g0s[i]*np.exp(-vr2vals/(w02+tempy))/((1.0+tempy/w02))
            vprofile+=tvprof
    profile=np.array(list(hprofile)+list(vprofile))
    return profile+params[0]

def getricsfull(xvals,rics):
    '''
    takes the single side rics profiles and make them into symmetrical peak x
    if G(0) is missing, replicate it from the first point
    '''
    fitsize=len(xvals)//2
    txvals=xvals[:fitsize]
    hrics=rics[:fitsize]
    vrics=rics[fitsize:]
    hotherside=np.flip(hrics)
    votherside=np.flip(vrics)
    xother=np.flip(txvals)
    if(xvals[0]==0):
        hprofile=np.array(list(hotherside)+list(hrics))
        vprofile=np.array(list(votherside)+list(vrics))
        xprofile=np.array(list(xother)+list(txvals))
        return hprofile,vprofile,xprofile
    else:
        hprofile=np.array(list(hotherside)+[hrics[0]]+list(hrics))
        vprofile=np.array(list(votherside)+[hrics[0]]+list(vrics))
        xprofile=np.array(list(xother)+[0]+list(txvals))
        return hprofile,vprofile,xprofile

def getricshalf(corr,skipg0=True):
    '''
    turns a symmetric set of rics profiles and turns them into a single side combined profile
    skipg0 omits the central point
    '''
    midpoint=corr.shape[-1]//2
    hprofile=corr[midpoint]
    vprofile=corr[:,midpoint]
    if(skipg0):
        xvals=np.array(list(range(1,midpoint))+list(range(1,midpoint)))
        profile=np.array(list(hprofile[(midpoint+1):])+list(vprofile[(midpoint+1):]))
        return xvals,profile
    else:
        xvals=np.array(list(range(midpoint))+list(range(midpoint)))
        profile=np.array(list(hprofile[midpoint:])+list(vprofile[midpoint:]))
        return xvals,profile

##########################################
#here are the ICS functions (Simple Gaussian)
##########################################

def gausfunc(xvals,*params):
    '''
    returns a centered gaussian for fitting ICS data
    params are baseline,g0,w0
    '''
    fitx,fity=np.arange(fitxpts),np.arange(fitypts)
    ygrid,xgrid=np.meshgrid(fity,fitx)
    r2grid=(ygrid-centery)**2+(xgrid-centerx)**2
    return (params[1]*np.exp(-r2grid/(params[2]**2))+params[0]).flatten()

##########################################
#here are the pch functions
##########################################

def poisson(avg,k):
    '''
    gets the poisson probability of recieving k counts given avg intensity
    '''
    if(avg>0.0):
        return (avg**k)*np.exp(-avg)/ss.factorial(k)
    else:
        if(k==0):
            return 1.0
        else:
            return 0.0

def mygamma(counts,x):
    '''
    my version of the incomplete gamma function
    '''
    gval=np.array([(x**i)/ss.factorial(i) for i in range(counts)]).sum()
    return ss.factorial(counts-1)*(1.0-np.exp(-x)*gval)

def p2GL(k,e):
    '''
    this function uses equation 15 from Chen et al 1999 to calculate the
	probability of receiving k counts from a single particle
	with brightness, i, given a gaussian lorentzian squared point spread
	function
    '''
    dx=0.01
    pi2=np.pi**2
    value=mygamma(k,(4.0*e)/pi2)
    integral=(value*dx)/2.0
    x=0.0
    while(value>(0.0001*integral)):
        x=x+dx
        value=(1+x*x)*mygamma(k,(4.0*e)/(pi2*(1+x*x)*(1+x*x)))
        integral=integral+value*dx
    return integral*(np.pi/(2.0*ss.factorial(k)))

def p3DG(k,e):
    '''
    this function uses equation 16 from Chen et al 1999 to calculate the
    probability of receiving k counts from a single particle
    with brightness, i, given a 3D gaussian point spread function
    '''
    dx=0.01
    value=mygamma(k,e)
    integral=(value*dx)/2.0
    x=0.0
    while(value>(0.0001*integral)):
        x=x+dx
        value=mygamma(k,(e*np.exp((-2.0)*x*x)))
        integral=integral+value*dx
    return integral*(2.0**1.5)/(np.sqrt(np.pi)*ss.factorial(k))

def p2DG(k,i):
    '''
    get the count probability for a single particle in a 2D gaussian focal volume
    since the volume of the psf doesn't matter, we will choose w0=1
    for 2D integration, dx*dy=r*dr*dtheta
    '''
    dx=0.01
    value=dx*ss.poisson(e,k)
    integral=(value*dx)/2.0
    x=dx;
    while(value>(0.0001*integral)):
        x+=dx
        value=x*poisson(e*np.exp(-2.0*x*x),k)
        integral+=value*dx
    return 4.0*integral

def convolve(arr1,arr2):
    '''
    do a non-fft convolution
    '''
    return np.array([np.array([arr1[i-j]*arr2[j] for j in range(i+1)]).sum() for i in range(len(arr1))])

def singlespecies(brightness,number,nlength=20,klength=15,psftype='3dgaussian'):
    '''
    calculates pch for a single species
    nlength should be more than the maximum realistic number of molecules
    klength should be 1.5 times the total pch size
    '''
    if(psftype=='3dgaussian'):
        probsingles1=np.array([0]+[p3DG(i,brightness) for i in range(1,klength+1)])
    else:
        if(psftype=='3dgl2'):
            probsingles1=np.array([0]+[p2GL(i,brightness) for i in range(1,klength+1)])
        else:
            probsingles1=np.array([0]+[p2DG(i,brightness) for i in range(1,klength+1)])
    # now calculate the 1 particle probability for k = 0 using the
    # normalization condition
    probsingles1[0]=1.0-probsingles1.sum()

    # now calculate the average of the n particle distributions for each k value weighted by
    # the poissonian distribution of n in the focal volume given an average n value
    # the zero particle distribution is 1 for k = 0 and 0 for all other k values
    probsingles3=np.zeros(klength+1,dtype=np.double)
    probsingles3[0]=1.0
    #temppch=np.array([1.0*poisson(number,0)]+[0.0]*klength)
    temppch=np.array([probsingles3[j]*poisson(number,0) for j in range(klength+1)])
    for i in range(1,nlength):
        probsingles2=probsingles3.copy()
        probsingles3=convolve(probsingles1,probsingles2);
        for j in range(klength+1):
            temppch[j]+=probsingles3[j]*poisson(number,i)
    return temppch

def getpch(brightnesses,numbers,background,klength=10,psftype='3dgaussian'):
    '''
    gets the overall pch function by convolving background and species with each other
    psf options are 3dgaussian, 3dgl2 (3d gauss lorentz squared), and 2dgaussian
    '''
    pch=singlespecies(brightnesses[0],numbers[0],klength=klength,psftype=psftype)
    for i in range(1,len(brightnesses)):
        tpch=singlespecies(brightnesses[i],numbers[i],klength=klength,psftype=psftype)
        pch=convolve(pch,tpch)
    if(background!=0.0):
        poissonvals=np.array([poisson(background,i) for i in range(klength+1)])
        pch=convolve(pch,poissonvals)
    return pch

def pchfunc(xvals,*params):
    '''
    the pch fitting function
    parameters are background, bright1, n1, ...
    psftype is set through the global module parameter
    '''
    brightnesses=[params[i] for i in range(1,len(params),2)]
    numbers=[params[i] for i in range(2,len(params),2)]
    klength=int(len(xvals)*1.5)+1
    return getpch(brightnesses,numbers,params[0],klength=klength,psftype=psftype)[:len(xvals)]

def binomial(ntrue,trials,trueprob):
    '''
    calculate the binomial function
    '''
    return (ss.factorial(trials)/(ss.factorial(ntrue)*ss.factorial(trials-ntrue)))*(trueprob**ntrue)*((1.0-trueprob)**(trials-ntrue))

def singlespecies2d(brightnessa,brightnessb,number,nlength=20,klengtha=15,klengthb=15,psftype='3dgaussian'):
    '''
    calculates 2d pch for a single species
    nlength should be more than the maximum realistic number of molecules
    klength should be 1.5 times the total pch size in each dimension
    '''
    #temppch=np.array([klengtha+1,klengthb+1],dtype=np.float64)
    #pchsingle=getpch(brightnessa+brightnessb,number,klength=(klengtha+klengthb))
    pchsingle=singlespecies(brightnessa+brightnessb,number,klength=(klengtha+klengthb),nlength=nlength)
    fbrighta=brightnessa/(brightnessa+brightnessb)
    temppch=[[binomial(i,i+j,fbrighta)*pchsingle[i+j] for j in range(klengthb+1)] for i in range(klengtha+1)]
    return np.array(temppch)

def convolve2d(mat1,mat2):
    '''
    do a non-fft 2d convolution
    assume matrices are the same shape
    '''
    retmat=[[np.array([mat1[k,l]*mat2[i-k,j-l] for l in range(j+1) for k in range(i+1)]).sum() for j in range(mat1.shape[1])] for i in range(mat1.shape[0])]
    return np.array(retmat)

def getpch2d(brightnessesa,brightnessesb,numbers,backgrounda,backgroundb,klengtha=10,klengthb=10,psftype='3dgaussian'):
    '''
    get the two channel pch function
    '''
    #first repeatedly convolve to get the multi-species 2d pch
    pch1=singlespecies2d(brightnessesa[0],brightnessesb[0],numbers[0],klengtha=klengtha,klengthb=klengthb,psftype=psftype)
    for i in range(1,len(brightnessesa)):
        if(numbers[i]!=0):
            temppch2=singlespecies2d(brightnessesa[i],brightnessesb[i],numbers[i],klengtha=klengtha,klengthb=klengthb,psftype=psftype)
            pch1=convolve2d(pch1,temppch2)
    #now convolve in the backgrounds
    if(backgrounda!=0.0):
        poissondist=np.zeros(pch1.shape,dtype=np.float64)
        poissondist[:,0]=np.array([poisson(backgrounda,i) for i in range(klengtha+1)])
        pch1=convolve2d(pch1,poissondist)
    if(backgroundb!=0.0):
        poissondist=np.zeros(pch1.shape,dtype=np.float64)
        poissondist[0,:]=np.array([poisson(backgroundb,i) for i in range(klengthb+1)])
        pch1=convolve2d(pch1,poissondist)
    return np.array(pch1)

def pchfunc2d(xvals,*params):
    '''
    the 2dpch fitting function for a 3d Gaussian psf
    parameters are backgrounda, backgroundb, bright1a, bright1b, n1, ...
    psftype is set through the global module parameter
    xvals is a simple range object and ignored
    '''
    nspecies=(len(params)-2)//3
    brightnessesa=[params[3*i+2] for i in range(nspecies)]
    brightnessesb=[params[3*i+3] for i in range(nspecies)]
    numbers=[params[3*i+4] for i in range(nspecies)]
    klengtha=int(pch2dsizes[0]*1.5)+1
    klengthb=int(pch2dsizes[1]*1.5)+1
    pch2d=getpch2d(brightnessesa,brightnessesb,numbers,params[0],params[1],
                    klengtha=klengtha,klengthb=klengthb,psftype=psftype)
    return pch2d[:pch2dsizes[0],:pch2dsizes[1]].flatten()

def getpchweights(pch):
    '''
    returns the pch weights (inverse of sigma) which are pch/(pchnorm*(1-pchnorm))
    '''
    pchnorm=pch.astype(float)/pch.sum()
    return np.array([pch[i]/(pchnorm[i]*(1.0-pchnorm[i])) if pch[i]>0 else 1.0 for i in range(len(pch))])

##########################################
#here are some N and B functions
##########################################

def var(intensity):
    '''
    returns the variance (<I^2>-<I>^2)
    assumes the first axis is the 
    '''
    return np.mean(intensity**2,axis=0)-np.mean(intensity,axis=0)**2

def covar(inta,intb):
    '''
    returns the covariance (<Ia*Ib>-<Ia><Ib>)
    '''
    return np.mean(inta*intb,axis=0)-np.mean(inta,axis=0)*np.mean(intb,axis=0)

def gaussFilterNaN(arr,sigma):
    '''
    applies a gaussian filter to an array with nan values
    convert the nans to zeros and then convert back to nan after smoothing
    '''
    sm=ndi.gaussian_filter(np.nan_to_num(arr),sigma)
    sm[np.isnan(arr)]=np.nan
    return sm

def polyContains(polygon,points):
    '''
    copy of the contains code from ImageJ FloatPolygon: https://imagej.nih.gov/ij/developer/source/ij/process/FloatPolygon.java.html
    polygon and points are numpy arrays of npts x [x,y] coordinates
    '''
    inside=np.zeros(len(points),dtype=bool)
    r1=np.roll(np.arange(polygon.shape[0]),-1)
    x=points[:,1]
    y=points[:,0]
    for i in range(polygon.shape[0]):
        mask=((polygon[i,0]>=y)!=(polygon[r1[i],0]>=y)) & (x>(polygon[r1[i],1]-polygon[i,1])*(y-polygon[i,0])/(polygon[r1[i],0]-polygon[i,0])+polygon[i,1])
        inside[mask]=~inside[mask]
    return inside

##########################################
#here are a some linear detrending functions
##########################################

def detrendStackLinearSeg(stack,segments,maintain_intensity=True):
    '''
    fits every pixel of a stack to a series of lines and subtracts it from the stack
    if maintain intensity is true the average intensity of each line is added back
    assumes a single channel image
    '''
    avgint=stack.mean(axis=0)
    #divide the stack into segments in time for detrending
    seglen=len(stack)//segments
    subs=[]
    for i in range(segments):
        start=i*seglen
        end=start+seglen
        if(i==(segments-1)):
            end=len(stack)
        slopes,intercepts=getStackTrends(stack[start:end])
        xvals=np.array(range(0,end-start))
        trendstack=[[slopes[i,j]*xvals+intercepts[i,j] for j in range(stack.shape[2])] for i in range(stack.shape[1])]
        sub=stack[start:end]-np.moveaxis(np.array(trendstack),2,0)
        subs.append(sub)
    if(segments>1):
        sub=np.concatenate(subs,axis=0)
    else:
        sub=subs[0]
    if(maintain_intensity):
        sub+=stack.mean(axis=0)
    return sub

def getStackTrends(stack):
    '''
    fits every pixel of a stack to a line and returns the coefficients
    note that intecept is (xsqsum*ysum-xsum*xysum)/(xsqsum*length-xsum*xsum)
    and slope is (xysum*length-xsum*ysum)/(xsqsum*length-xsum*xsum)
    returns slope and intercept arrays
    '''
    xvals=np.array(range(len(stack))).astype(float)
    xsum=xvals.sum()
    xsqsum=(xvals*xvals).sum()
    ysums=stack.sum(axis=0).astype(float)
    xysums=(stack*xvals.reshape(-1,1,1)).sum(axis=0).astype(float)
    nslices=len(stack)
    divisor=xsqsum*nslices-xsum*xsum
    slopes=(xysums*nslices-ysums*xsum)/divisor
    intercepts=(ysums*xsqsum-xysums*xsum)/divisor
    return slopes,intercepts

##########################################
#here are a few colormap functions
##########################################

def getncmap():
    '''
    #get the jet colormap with the under values set to white
    '''
    ncmap=plt.colormaps['jet']
    ncmap.set_under([1,1,1,1])
    return ncmap

def getsinglecmap(color='green'):
    '''
    gets a single colormap as a linear segmented colormap by name
    '''
    colors={'blue':[1,0,0],'cyan':[1,1,0],
            'green':[0,1,0],'yellow':[0,1,1],
            'red':[0,0,1],'magenta':[1,0,1]}
    if(color not in colors):
        print('color not found')
        return None
    from matplotlib.colors import ListedColormap
    cmax=colors[color]
    ramp=np.arange(256)/256.0
    clist=np.array([ramp*cmax[2],ramp*cmax[1],ramp*cmax[0]]).T
    return ListedColormap(clist,name=color)