import numpy as np
import pylab
import matplotlib as mpl
# edit text size in figures and colormap
font_size = 12
mpl.rcParams['axes.titlesize'] = font_size+2
mpl.rcParams['xtick.labelsize'] = font_size
mpl.rcParams['ytick.labelsize'] = font_size
mpl.rcParams['axes.labelsize'] = font_size
mpl.rcParams['legend.fontsize'] = font_size-4
mpl.rcParams['font.size'] = font_size
new_rc_params = {'text.usetex': False,"svg.fonttype": 'none'}
mpl.rcParams.update(new_rc_params)
mpl.rcParams['image.cmap'] = 'jet'

### random number generator RNG
rng = np.random.RandomState(10) # the argument is the seed for RNG and can be any integer

### parameters for grid
l = np.array([31,43])
oriarr = [0.39,0.46] # [0.39,0.46,0.03,0.07,0.19,0.28] # < 0.5235987755982988
N = len(l)     # num of grid modules
M2d = 30
M = M2d**2   # num of neurons in a module
#Ng = N*M
#c = 10
Np = 253 #320 # num of place cells
#sNr = 253
#res = 5
sig = 0.16   # std of grid cell response 0.066 / 0.106 (Fiete 2012), 0.283 earlier
nsig = 0.2
R = 4800   # length of track (cm)
x = np.arange(R)
#X = 300    # size of box
#dx = 1
#num2print = 8
num_bs = 1000
#scale = 0.

### random number generators
seed = 1
#seed_bs = 10

### parameters for 1D model
constrained = 1
selected = 0
conn = 'RDM'
dist = 'lgn'
mode = 'FIX'
normalized = 'post'
wp1 = 0
wp2 = 0.8 #1.2
c = 0.2 # N*M*c ~ 1000
nth = 1.794
sd_frange = 0.04

### path
figpath = './figures/'
if not os.path.exists(figpath):
    os.makedirs(figpath)
datapath = './data/'
if not os.path.exists(datapath):
    os.makedirs(datapath)
