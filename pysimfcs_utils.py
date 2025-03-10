from tqdm.notebook import tqdm
import numpy as np
import scipy.fft as fft
import scipy.ndimage as ndi
import pandas as pd
import scipy.special as ss
import scipy.signal as ssig

#here are some global parameters for the simulation
noisemodes={'none':'no_noise','poisson':'poisson_noise','analog':'analog_noise'}
#analog noise settings
analoggain=10.0
analogoffset=100.0
analogreadstdev=5.0

def scaleSimSettings(simsettings):
    '''
    scale the simulation settings (w0 and background)
    '''
    ssimsettings=simsettings.copy()
    psize=simsettings['boxsize']/simsettings['boxpixels']
    ftime=simsettings['frametime']*1.0e-6
    ssimsettings['w0']=simsettings['w0']/psize
    ssimsettings['background']=[bval*ftime for bval in simsettings['background']]
    return ssimsettings

def scaleSimSpecies(simsettings,simspecies):
    '''
    scale the simspecies dictionary list to pixel and frame units
    '''
    scaless=[]
    psize=simsettings['boxsize']/simsettings['boxpixels']
    ftime=simsettings['frametime']*1.0e-6
    for i in range(len(simspecies)):
        scaless.append({'N':simspecies[i]['N']})
        scaless[i]['D']=simspecies[i]['D']*ftime/(psize*psize)
        scaless[i]['B']=[bval*ftime for bval in simspecies[i]['B']]
    return scaless

def initSim(ssimsettings,ssimspecies):
    '''
    initializes the coordinates for the simulation
    '''
    coords=[]
    for i in range(len(ssimspecies)):
        scoords=np.array([np.random.uniform(0,ssimsettings['boxpixels'],ssimspecies[i]['N']),
                          np.random.uniform(0,ssimsettings['boxpixels'],ssimspecies[i]['N']),
                          np.random.uniform(0,ssimsettings['boxpixels'],ssimspecies[i]['N'])]).T
        coords.append(scoords)
    return coords

def movemolecules2(coords,ssimsettings,ssimspecies):
    '''
    moves molecules according to scaled simsettings and simspecies
    '''
    coords2=[]
    for i in range(len(coords)):
        tcoords=movemolecules(coords[i],ssimspecies[i]['D'],ssimsettings['dims'],
                              [ssimsettings['boxpixels']]*3,ssimsettings['boundarytype'])
        coords2.append(tcoords)
    return coords2

def movemolecules(coords,dscaled,diffdimensions=[True,True,True],
                  boxpixels=[128,128,128],boundarytype='periodic'):
    '''
    update the coordinates according to the scaled diffusion coefficient (pixels squared per frame)
    boundary can be periodic or reflective
    '''
    coords2=coords.copy()
    for i in range(len(diffdimensions)):
        if(diffdimensions[i]):
            coords2[:,i]=np.random.normal(loc=coords[:,i],scale=np.sqrt(2.0*dscaled))
    fboxpixels=np.array(boxpixels).astype(float)
    over0=(coords2[:,0]>=fboxpixels[0])
    over1=(coords2[:,1]>=fboxpixels[1])
    over2=(coords2[:,2]>=fboxpixels[2])
    under0=(coords2[:,0]<0.0)
    under1=(coords2[:,1]<0.0)
    under2=(coords2[:,2]<0.0)
    if(boundarytype=='periodic'):
        coords2[over0,0]-=fboxpixels[0]
        coords2[under0,0]+=fboxpixels[0]
        coords2[over1,1]-=fboxpixels[1]
        coords2[under1,1]+=fboxpixels[1]
        coords2[over2,2]-=fboxpixels[2]
        coords2[under2,2]+=fboxpixels[2]
    elif(boundarytype=='reflective'):
        coords2[over0,0]=2.0*fboxpixels[0]-coords2[over0,0]
        coords2[under0,0]=-coords2[under0,0]
        coords2[over1,1]=2.0*fboxpixels[1]-coords2[over1,1]
        coords2[under1,1]=-coords2[under1,1]
        coords2[over2,2]=2.0*fboxpixels[2]-coords2[over2,2]
        coords2[under2,2]=-coords2[under2,2]
    return coords2

#def gausfunc(rvals,gausref=None):
#    '''
#    returns the interpolated gaussian function (sigma = 1)
#    this doesn't speed things up over numpy array exponential function and loses accuracy
#    '''
#    if(gausref is None):
#        gausref=np.exp(-(np.linspace(0.0,100.0,1000)**2)/2.0)
#    rscale=np.array(rvals)*10.0
#    rp=np.floor(rscale).astype(int)
#    rem=rscale-rp
#    closevals=rp<999
#    retvals=np.zeros(len(rvals),dtype=float)
#    temp=gausref[rp[closevals]]
#    temp2=gausref[rp[closevals]+1]
#    retvals[closevals]=rem[closevals]*(temp2-temp)+temp
#    return retvals

def addNoise(intensities,noisemode='poisson'):
    '''
    adds non, poisson, or analog noise to an intensity or image
    '''
    if(noisemode=='none'):
        return intensities
    elif(noisemode=='poisson'):
        return np.random.poisson(intensities)
    else:
        #analog mode: exponential gain for each photon plus gaussian read noise at offset level
        #use the global offset, readstdev, and gain settings
        photons=np.random.poisson(intensities)
        gainfunc=lambda x:np.sum([np.random.exponential(analoggain) for i in range(x)])
        gainsig=np.vectorize(gainfunc)(photons)
        offsig=np.random.normal(analogoffset,analogreadstdev,intensities.shape)
        return offsig+gainsig

def getIntensity(coords,ssimsettings,ssimspecies):
    '''
    get the intensity from a set of species and coordinates
    '''
    return getSimIntensity(coords,[ssimsettings['boxpixels']/2]*3,
                           ssimsettings,ssimspecies,ssimsettings['noisemode'])

def getSimIntensity(coords,pos,ssimsettings,ssimspecies,noisemode='poisson'):
    '''
    get the intensity of all populations with noise added
    see getPopIntensity for details
    '''
    #get the intensity for all species (species x channels list)
    intensity=np.array([getPopIntensity(coords[i],pos,ssimspecies[i]['B'], \
                                        ssimsettings['w0'],ssimsettings['zratio']) \
                          for i in range(len(coords))])
    #now sum over the populations and add noise
    return addNoise(intensity.sum(axis=0)+np.array(ssimsettings['background']),noisemode=noisemode)

def getPopIntensity(coords,pos,brightness,w0pixels,zratio,maxdist=None):
    '''
    get the single population intensity at the focal point with a 3D gaussian focal volume
    w0 is 2*sigma
    note that brightness is a list for the channels
    return an integrated intensity
    '''
    if(maxdist):
        maxdist2=maxdist*maxdist
    else:
        maxdist2=2.0*2.0*w0pixels*w0pixels
    dist2=((coords[:,0]-pos[0])/zratio)**2+(coords[:,1]-pos[1])**2+(coords[:,2]-pos[2])**2
    #filter out the distances beyond maxdist
    dist2filt=dist2[dist2<=maxdist2]
    #now calculate the intensities
    intensity=np.exp(-2.0*(dist2filt)/(w0pixels*w0pixels)).sum()
    tint=np.array(brightness)*intensity
    return tint

def runPointSimulation(ssimsettings,ssimspecies):
    '''
    run a point simulation
    return a pandas dataframe with columms for each active channel
    '''
    coords=initSim(ssimsettings,ssimspecies)
    simtraj=[getIntensity(coords,ssimsettings,ssimspecies)]
    for i in tqdm(range(1,ssimsettings['nframes'])):
        coords=movemolecules2(coords,ssimsettings,ssimspecies)
        simtraj.append(getIntensity(coords,ssimsettings,ssimspecies))
    simtraj=np.array(simtraj)
    tottime=ssimsettings['nframes']*ssimsettings['frametime']*1.0e-6
    xvals=np.linspace(0.0,tottime,ssimsettings['nframes'])
    nchan=len(coords)
    df=pd.DataFrame({'x':xvals})
    for i in range(len(coords)):
        df['ch'+str(i+1)]=simtraj[:,i]
    return df

def runRasterSimulation(ssimsettings,ssimspecies):
    '''
    run a raster scanning simulation
    return an image stack (for raster_image) or carpet stack (for raster_line)
    '''
    coords=initSim(ssimsettings,ssimspecies)
    if(ssimsettings['mode']=='raster_line'):
        carpet=[]
        ypos=zpos=ssimsettings['boxpixels']//2
        for i in tqdm(range(ssimsettings['nframes'])):
            line=[]
            for j in range(ssimsettings['boxpixels']):
                intensity=getSimIntensity(coords,[zpos,ypos,j],ssimsettings,ssimspecies,ssimsettings['noisemode'])
                line.append(intensity)
                coords=movemolecules2(coords,ssimsettings,ssimspecies)
            carpet.append(line)
        #carpet shape at this point should be line x xpos x channel
        #shift to an array of channels
        return np.moveaxis(np.array(carpet),-1,0)
    elif(ssimsettings['mode']=='raster_image'):
        stack=[]
        zpos=ssimsettings['boxpixels']//2
        for k in tqdm(range(ssimsettings['nframes'])):
            image=[]
            for i in range(ssimsettings['boxpixels']):
                line=[]
                for j in range(ssimsettings['boxpixels']):
                    intensity=getSimIntensity(coords,[zpos,i,j],ssimsettings,ssimspecies,ssimsettings['noisemode'])
                    line.append(intensity)
                    coords=movemolecules2(coords,ssimsettings,ssimspecies)
                image.append(line)
            stack.append(image)
        #stack shape at this point should be frame x line x xpos x channel
        #shift to an array of channels
        return np.moveaxis(np.array(stack),-1,0)
    else:
        #for now only support raster_line or raster_image
        return None

def getPopImage(coords,zpos,ssimsettings,ssimspecies,maxdist=None):
    '''
    get an image of a population of molecules (e.g. from a spinning disk)
    note Im ignoring the background here
    '''
    if(maxdist):
        maxdist2=maxdist*maxdist
    else:
        maxdist2=2.0*2.0*ssimsettings['w0']*ssimsettings['w0']
    #start by placing points on an image
    #the point intensity should be the brightness modulated by the z dist
    zdist2=((coords[:,0]-zpos)/ssimsettings['zratio'])**2
    infocus=(zdist2<=maxdist2)
    coordsfilt=coords[infocus]
    zbright=np.exp(-(zdist2[infocus])/2.0)
    nchan=len(ssimspecies['B'])
    boxpixels=ssimsettings['boxpixels']
    #img=[np.full([boxpixels,boxpixels],fill_value=back) for back in background]
    img=[np.zeros([boxpixels,boxpixels],dtype=float) for i in range(nchan)]
    xfilt=np.round(coordsfilt[:,2]).astype(int)
    #wrap the out of bounds pixels to the other side
    xfilt[xfilt<0]+=boxpixels
    xfilt[xfilt>=boxpixels]-=boxpixels
    yfilt=np.round(coordsfilt[:,1]).astype(int)
    yfilt[yfilt<0]+=boxpixels
    yfilt[yfilt>=boxpixels]-=boxpixels
    #add the points to the image
    for i in range(nchan):
        img[i][yfilt,xfilt]+=ssimspecies['B'][i]*zbright
        #gaussian blur according to the psf
        img[i]=ndi.gaussian_filter(img[i],sigma=ssimsettings['w0']/2)
    return img

def runImageSimulation(ssimsettings,ssimspecies):
    '''
    run an image simulation
    return a multichannel stack
    '''
    coords=initSim(ssimsettings,ssimspecies)
    stack=[]
    zpos=ssimsettings['boxpixels']//2
    for k in tqdm(range(ssimsettings['nframes'])):
        popstack=[]
        for j in range(len(coords)):
            img=getPopImage(coords[j],zpos,ssimsettings,ssimspecies[j])
            popstack.append(img)
        allstack=np.array(popstack).sum(axis=0)
        for j in range(len(ssimsettings['background'])):
            allstack+=ssimsettings['background'][j]
        stack.append(addNoise(allstack,ssimsettings['noisemode']))
        coords=movemolecules2(coords,ssimsettings,ssimspecies)
    #stack shape at this point should be frame x channel x line x xpos
    return np.array(stack)