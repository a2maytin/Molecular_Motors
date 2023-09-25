#
import lm
import pyLM
import pyLM.units
from pyLM.units import *
import pyLM.RDME
from pyLM.RDME import RDMESimulation
import numpy as np
import random
import math
import scipy as sc
from scipy.spatial.distance import cdist
from scipy.spatial.distance import pdist
# Arguments

outputFile = "Infection.lm"

# Domain
latticeSpacing = 64 # nm
sim = RDMESimulation(dimensions=micron(16.384, 16.384, 16.384), spacing=nm(latticeSpacing), defaultRegion='extra') # 
# should be size/latticeSpacing=0 ; size/32=0

species = ['L','M','S','X','preC','capsid', 'CNPC','dimer','rcDNA', 'cccDNA1','cccDNA2','cccDNA3','cccDNA4' ,'cccDNA5','RNApol','cccDNApol1','cccDNApol2','cccDNApol3','cccDNApol4','cccDNApol5','pgRNA','LSmRNA','SmRNA','preCmRNA','XmRNA', 'capsidE', 'capsidR', 'viralP', 'RNP' ] 

# Filled virion
species2 = ['LC%d']
species3 = ['LCE%d']
species4 = ['LCR%d']

sim.defineSpecies(species)
for i in range (1,21):
	for s in species2:
		sim.defineSpecies([s%i])

for i in range (1,21):
	for s in species3:
		sim.defineSpecies([s%i])

for i in range (1,21):
	for s in species4:
		sim.defineSpecies([s%i])


sim.addRegion('CellWall')
sim.addRegion('Cytoplasm')
sim.addRegion('NucEnv')
sim.addRegion('Nucleus')
sim.addRegion('Nucleoli')
sim.addRegion('Cajal')
sim.addRegion('Speckle')
sim.addRegion('NPC')
sim.addRegion('Mito')
sim.addRegion('Golgi')
sim.addRegion('ER')
sim.addRegion('MT')
sim.addRegion('MTOC')
#sim.addRegion('Boundary')


############################################ CELL GEOMETRY ######################################################################

cellCenter = lm.point(*micron(6.144,6.144,8))
nuclCenter = lm.point(*micron(6.144,6.144, 6.144))
lengthX = micron(12.288)
lengthY = micron(12.288)   
lengthZ = micron(16) 
cellRadius = micron(8.9)
membraneThickness= micron(0.128)
nuclSize= micron(3.5)
nucleoliRadius = micron(0.9)
speckleRadius=micron(0.35)
cajalRadius=micron(0.5)
poreRadius=micron(0.08)
n_mtoc = 4 
n_mts = 1600 
n_cajals=4
n_speckles=20
n_NPCs=1078
n_mito= 1650
n_nucleoli = 2


####################### Create cell components ##########################

Cytoplasm = lm.Cuboid(cellCenter, lengthX-membraneThickness, lengthY-membraneThickness, lengthZ-membraneThickness, sim.siteTypes['Cytoplasm'])             
Cytoplasm.thisown = 0


####################### Golgi Apparatus  #################################################
if True:
    m1 = lm.Sphere(nuclCenter , micron(3.5), sim.siteTypes['Golgi'])             
    m11 = lm.Sphere(nuclCenter, micron(3.308), sim.siteTypes['Golgi'])             
    m12 = lm.Sphere(nuclCenter, micron(3.116), sim.siteTypes['Golgi'])             
    m13 = lm.Sphere(nuclCenter, micron(2.924), sim.siteTypes['Golgi'])             
    m14 = lm.Sphere(nuclCenter, micron(2.732), sim.siteTypes['Golgi'])             

    m2 = lm.Cone(nuclCenter, micron(2), micron(6.5), sim.siteTypes['Golgi'], lm.vector(1.0,1.0,1.0))
    m22 = lm.Cone(nuclCenter, micron(2.5), micron(8), sim.siteTypes['Golgi'], lm.vector(1.0,1.0,1.0))
    m23 = lm.Cone(nuclCenter, micron(3), micron(9.5), sim.siteTypes['Golgi'], lm.vector(1.0,1.0,1.0))
    m24 = lm.Cone(nuclCenter, micron(3.5), micron(10.5), sim.siteTypes['Golgi'], lm.vector(1.0,1.0,1.0))


    Gol1= lm.Difference(m1, m11, sim.siteTypes['Golgi'])
    Gol2= lm.Difference(m11, m12, sim.siteTypes['Golgi'])
    Gol3= lm.Difference(m12, m13, sim.siteTypes['Golgi'])
    Gol4= lm.Difference(m13, m14, sim.siteTypes['Golgi'])

    G1 = lm.Intersection(Gol1, m2, sim.siteTypes['Golgi'])
    G2 = lm.Intersection(Gol2, m22, sim.siteTypes['Golgi'])
    G3 = lm.Intersection(Gol3, m23, sim.siteTypes['Golgi'])
    G4 = lm.Intersection(Gol4, m24, sim.siteTypes['Golgi'])

    G11 = lm.Union(G1, G2, sim.siteTypes['Golgi'])
    G22 = lm.Union(G3, G4, sim.siteTypes['Golgi'])
    Golgi = lm.Union(G11, G22, sim.siteTypes['Golgi'])
    Golgi.thisown = 0
    print("Golgi done!")

######################## Mitochondria with number of "n_mito" ###############################3
if True: 
    mit = []
    pp1 = []
    pp2 = []
    Mito = lm.UnionSet(sim.siteTypes['Mito'])
    Mito.thisown=0
    i = 0
    while i < n_mito :
        x,y,z = np.random.uniform (-0.93, 0.93, 3)
        xm = x*lengthX*0.5 + cellCenter.x 
        ym = y*lengthY*0.5 + cellCenter.y 
        zm = z*lengthZ*0.5 + cellCenter.z 
        r = math.sqrt((xm-cellCenter.x)**2 + (ym-cellCenter.y)**2 + (zm-cellCenter.z)**2)
        if r >  micron(5.5) : 
            i += 1
            if i < 550 :
                point1 = [xm - micron(0.125), ym-micron(0.125) , zm]
                point2 = [xm + micron(0.125), ym+micron(0.125) , zm]
            elif  i >= 550 and i < 1100 :
                point1 = [xm, ym - micron(0.125), zm- micron(0.125)]
                point2 = [xm, ym + micron(0.125), zm+ micron(0.125)]
            elif  i >= 1100 :
                point1 = [xm - micron(0.125), ym , zm - micron(0.125)]
                point2 = [xm + micron(0.125), ym , zm + micron(0.125)]
                
            pp1.append(point1)
            pp2.append(point2)
    mp1 = np.array(pp1)
    mp2 = np.array(pp2)

    for h in range(0, n_mito):
        m = lm.Capsule(lm.point(mp1[h][0], mp1[h][1], mp1[h][2]),
                       lm.point(mp2[h][0], mp2[h][1], mp2[h][2]),
                       micron(0.25),
                       sim.siteTypes['Mito'])
        m.thisown = 0
        mit.append(m)
        Mito.addShape(m)
    print("Mito done!")

    ################# MTOC: currently 2 of them  #########################
    m_mtoc = []
    P_MTOC = np.zeros((n_mtoc, 3))
    MTOC = lm.UnionSet(sim.siteTypes['MTOC'])
    MTOC.thisown=0
    taken = 0
    counter = 0

    P_MTOC[0] = [nuclCenter.x + micron(5.5),nuclCenter.y, nuclCenter.z] 
    P_MTOC[1] = [nuclCenter.x - micron(5.5),nuclCenter.y, nuclCenter.z ] 
    P_MTOC[2] = [nuclCenter.x, nuclCenter.y + micron(5.5), nuclCenter.z] 
    P_MTOC[3] = [nuclCenter.x,  nuclCenter.y - micron(5.5), nuclCenter.z] 

    for h in range(0, n_mtoc):
        if h == 0:
            m_mtoc = lm.Capsule(lm.point(P_MTOC[h][0], P_MTOC[h][1], P_MTOC[h][2]),
                           lm.point(P_MTOC[h][0]- micron(0.3), P_MTOC[h][1], P_MTOC[h][2]),
                           micron(0.15),
                           sim.siteTypes['MTOC'])
            m_mtoc.thisown = 0
            MTOC.addShape(m_mtoc)

        if h == 1:
            m_mtoc = lm.Capsule(lm.point(P_MTOC[h][0], P_MTOC[h][1], P_MTOC[h][2]),
                           lm.point(P_MTOC[h][0]+ micron(0.3), P_MTOC[h][1], P_MTOC[h][2]),
                           micron(0.15),
                           sim.siteTypes['MTOC'])
            m_mtoc.thisown = 0
            MTOC.addShape(m_mtoc)

        if h == 2:
            m_mtoc = lm.Capsule(lm.point(P_MTOC[h][0], P_MTOC[h][1], P_MTOC[h][2]),
                           lm.point(P_MTOC[h][0], P_MTOC[h][1]- micron(0.3), P_MTOC[h][2]),
                           micron(0.15),
                           sim.siteTypes['MTOC'])
            m_mtoc.thisown = 0
            MTOC.addShape(m_mtoc)

        if h == 3:
            m_mtoc = lm.Capsule(lm.point(P_MTOC[h][0], P_MTOC[h][1], P_MTOC[h][2]),
                           lm.point(P_MTOC[h][0]- micron(0.3), P_MTOC[h][1], P_MTOC[h][2]+ micron(0.3)),
                           micron(0.15),
                           sim.siteTypes['MTOC'])
            m_mtoc.thisown = 0
            MTOC.addShape(m_mtoc)
    print("MTOC done!")


    ################# MTs: with the number of "n_mts" #########################
    mt = []
    m_mt = []
    P_MT = []
    sec_point = []
    rad = []
    acc = 0
    MT_length = micron(7)
    MT = lm.UnionSet(sim.siteTypes['MT'])
    MT.thisown=0
    taken = 0

    for i in range (0, n_mts):
        x,y,z = np.random.uniform (-0.95, 0.95, 3)
        xmt = x*lengthX*0.5 + cellCenter.x 
        ymt = y*lengthY*0.5 + cellCenter.y 
        zmt = z*lengthZ*0.5 + cellCenter.z 
        r = math.sqrt((xmt-cellCenter.x)**2 + (ymt-cellCenter.y)**2 + (zmt-cellCenter.z)**2)
        mt_center = [xmt, ymt, zmt]
        sec_point.append(mt_center)


    for m in range (0, n_mtoc):
        for h in range(m*400, (m+1)*400):
                m_mt = lm.Capsule(lm.point(P_MTOC[m][0], P_MTOC[m][1], P_MTOC[m][2]),
                                  lm.point( sec_point[h][0], sec_point[h][1], sec_point[h][2]),
                                  micron(0.064),
                                  sim.siteTypes['MT'])
                m_mt.thisown = 0
                MT.addShape(m_mt)

    print("MTs done!")

    #################  Nucleus, Envelope and Nucleolus #############################
CellWall  = lm.Cuboid(cellCenter, lengthX ,lengthY, lengthZ, sim.siteTypes['CellWall'])
NucEnv    = lm.Sphere( lm.point(*micron(6.144,6.144, 6.144)) , nuclSize, sim.siteTypes['NucEnv'])
Nucleus   = lm.Sphere(lm.point(*micron(6.144,6.144, 6.144)), nuclSize-membraneThickness, sim.siteTypes['Nucleus'])
Nucleoli   = lm.Sphere( nuclCenter, nucleoliRadius, sim.siteTypes['Nucleoli'])
CellWall.thisown = 0
NucEnv.thisown = 0
Nucleus.thisown = 0
Nucleoli.thisown = 0

################ Cajal body: "n_cajal" with radius "cajalRadius" #############################
if True: 
    dist = [] ; listdist = []
    Cajalpre = []
    accepted = []
    np_accepted = []
    cc = 1 ; point = 0
    while cc < n_cajals + 1:
        u = np.random.uniform(-1, 1)
        thet = np.random.uniform(0, 2.0*math.pi)
        rad = np.random.uniform( nucleoliRadius*1e6 + cajalRadius*1e6 + 0.1 , nuclSize*1e6 - (cajalRadius*1e6 + 0.1))
        xp = micron(rad*math.sqrt(1-u**2)*np.cos(thet)) + nuclCenter.x
        yp = micron(rad*math.sqrt(1-u**2)*np.sin(thet)) + nuclCenter.y
        zp = micron(rad*u) + nuclCenter.z
        caj = [xp, yp, zp]
        listdist = []
        point += 1
        if (point == 1 ):
            accepted.append(caj)
            np_accepted = np.array(accepted)
        for i in range (0, cc):
            dist =  np.sqrt(np.sum((np.array(caj)-np_accepted[i])**2))
            listdist.append(dist) 
        np_dist = np.array(listdist)
        if(np.all(np_dist > 2*cajalRadius + micron(0.05))):
            accepted.append(caj)
            np_accepted = np.array(accepted)
            listdist = []
            cc += 1
    NCajal = np.array(accepted)

    cc = [] 
    Cajal = lm.UnionSet(sim.siteTypes['Cajal'])
    Cajal.thisown = 0
    for k in range(0, n_cajals):
        w = lm.Sphere(lm.point(NCajal[k][0],
                               NCajal[k][1],
                               NCajal[k][2]),
                      cajalRadius,
                      sim.siteTypes['Cajal'])
        w.thisown=0
        cc.append(w)
        Cajal.addShape(w)
    print("Cajal done!")

################ Nuclear Speckles: "n_speckles" with radius "speckleRadius" ##########################
if True:
    counter = 0
    taken = 0 
    listdist = [] ; np_acceptspeck = [] ; acceptspeck = [] 
    while counter < n_speckles + 1:
        u = np.random.uniform(-1, 1)
        thet = np.random.uniform(0, 2.0*math.pi)
        rad = np.random.uniform( nucleoliRadius*1e6 + speckleRadius*1e6 + 0.1 , nuclSize*1e6 - (speckleRadius*1e6 + 0.1))
        xm = micron(rad*math.sqrt(1-u**2)*np.cos(thet)) + nuclCenter.x
        ym = micron(rad*math.sqrt(1-u**2)*np.sin(thet)) + nuclCenter.y
        zm = micron(rad*u) + nuclCenter.z
        point = [xm, ym, zm]
        listdist = [] ; dist = 0 ; np_dist = []
        taken += 1
        if ( taken == 1 ):
            acceptspeck.append(point)
            np_acceptspeck = np.array(acceptspeck)
        for i in range (0, counter):
            dist =  np.sqrt(np.sum((point-np_acceptspeck[i])**2))
            listdist.append(dist)
        np_dist = np.array(listdist)
        tt = 0
        if(np.all(np_dist > 2*speckleRadius + micron(0.1))):
            for k in range (0, n_cajals):
                if (np.sqrt(np.sum((np.array(point)-NCajal[k])**2)) > speckleRadius + cajalRadius + micron(0.1)):
                    tt += 1
            if (tt == 4):
                acceptspeck.append(point)
                np_acceptspeck = np.array(acceptspeck)
                counter += 1
                listdist = []
    NSpeck = np.array(acceptspeck)

    collection = []
    j = 0
    Speckle = lm.UnionSet(sim.siteTypes['Speckle'])
    Speckle.thisown = 0
    for k in range(0, n_speckles):
        q = lm.Sphere(lm.point(NSpeck[k][j],
                               NSpeck[k][j+1],
                               NSpeck[k][j+2]),
                      speckleRadius,
                      sim.siteTypes['Speckle'])
        q.thisown=0
        collection.append(q)
        Speckle.addShape(q)
    print("Speckles done!")

################# NPC: with the number of "n_NPCs" #########################
if True:
    NPCpre = []
    taken = 0

    ldist = [] ; npcdist = 0 ; np_npcdist = []
    acceptnpc = []
    npc = 0

    while npc < n_NPCs + 1:   
        x,y,z = np.random.uniform(-0.5,0.5,3)
        r = math.sqrt(x**2 + y**2 + z**2) 
        Rp = (nuclSize - sim.latticeSpacing*0.5)/r
        xpp = x*Rp + nuclCenter.x
        ypp = y*Rp + nuclCenter.y
        zpp = z*Rp + nuclCenter.z
        pores = [xpp,ypp,zpp]
        NPCpre.append(pores)
        ldist = [] ; np_npcdist = []
        taken += 1
        if (taken == 1 ):
            acceptnpc.append(pores)
            np_acceptnpc = np.array(acceptnpc)
        for i in range (0, npc):
            npcdist =  np.sqrt(np.sum((np.array(pores)-np_acceptnpc[i])**2))
            ldist.append(npcdist)
        np_npcdist = np.array(ldist)
        if(np.all(np_npcdist > 0.18e-6)):
            acceptnpc.append(pores)
            np_acceptnpc = np.array(acceptnpc)
            ldist = []
            npc += 1

    NP = np.array(acceptnpc)

    collect = []
    j = 0
    NPC = lm.UnionSet(sim.siteTypes['NPC'])
    NPC.thisown = 0
    for k in range(0, n_NPCs):
        s = lm.Sphere(lm.point(NP[k][j],
                               NP[k][j+1],
                               NP[k][j+2]),
                      poreRadius,
                      sim.siteTypes['NPC'])
        s.thisown=0
        collect.append(s)
        NPC.addShape(s)

    print("NPCs done!")

########################################################################

sim.lm_builder.addRegion(CellWall)
sim.lm_builder.addRegion(Cytoplasm)
sim.lm_builder.addRegion(Mito)
sim.lm_builder.addRegion(MT)
sim.lm_builder.addRegion(MTOC)
sim.lm_builder.addRegion(Golgi)
sim.lm_builder.addRegion(NucEnv)
sim.lm_builder.addRegion(Nucleus)
sim.lm_builder.addRegion(Cajal)
sim.lm_builder.addRegion(Speckle)
sim.lm_builder.addRegion(Nucleoli)
sim.lm_builder.addRegion(NPC)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PARTICLE COUNTS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       

sim.addParticles(species='capsid', region='Cytoplasm', count = 52 )  
sim.addParticles(species='capsidR', region='Cytoplasm', count = 260 ) 
sim.addParticles(species='capsidE', region='Cytoplasm', count = 520  ) 

sim.addParticles(species='dimer', region='Cytoplasm', count =4)  

#sim.addParticles(species='rcDNA', region='NPC', count = 0 )
sim.addParticles(species='RNApol', region='Nucleus', count = 680 )
sim.addParticles(species='cccDNA1', region='Nucleus', count = 5 ) # Starting from an already infected cell
sim.addParticles(species='cccDNA2', region='Nucleus', count = 5 ) # Starting from an already infected cell
sim.addParticles(species='cccDNA3', region='Nucleus', count = 5 ) # Starting from an already infected cell
sim.addParticles(species='cccDNA4', region='Nucleus', count = 5 ) # Starting from an already infected cell
sim.addParticles(species='cccDNA5', region='Nucleus', count = 5 ) # Starting from an already infected cell
sim.addParticles(species='pgRNA', region='Nucleus', count =  59 )
sim.addParticles(species='LSmRNA', region='Nucleus', count = 0)
sim.addParticles(species='SmRNA', region='Nucleus', count = 0)
sim.addParticles(species='XmRNA', region='Nucleus', count = 0)
sim.addParticles(species='preCmRNA', region='Nucleus', count = 0)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% REACTIONS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Vvoxel=0.000262144e-15 # 0.064^3 in Liter
scale = 157863.12  # = 6.022e23 * 0.064^3 e-15 in Liters 

#Kb =4*3.1415*7.4e-12*45e-10*1000/Vvoxel  # 4*pi*(D1+D2)*(R1+R2)*1000*NA/(NA*V)
 #K1 = 0.0071 ; K2 = 0.0104 ; K3 = 0.0119; K4 = 0.0357 ; K5 = 0.0071
Kdis =1.667e-3 # This could be of the order of tens of minutes but here I assume 10 mins based on the literature

Kbpol =0.33e6/scale  
Kunb =1e-3  

K1 =0.02 ; K2 =0.0208 ; K3 =0.0238 ; K4 =0.0714 ; K5 =0.0784  # Based on ave RNAP speed of 50 b/s 

# Complete and Empty capaids dissociate in NPCs
sim.modifyRegion('NPC').addReaction(reactant= 'capsid', product = ('rcDNA', 'dimer','dimer'), rate=Kdis ) #complete capsid
sim.modifyRegion('NPC').addReaction(reactant= 'capsidE', product = (''), rate=Kdis ) #Empty capsid

sim.modifyRegion('Nucleus').addReaction(reactant= 'rcDNA', product = ('cccDNA1','cccDNA2','cccDNA3','cccDNA4','cccDNA5'), rate =0.0033) # changed based on DNA repair rate of 12-24 b/4 mins for 1.6 kb of HBV genome  

sim.modifyRegion('Nucleus').addReaction(reactant=('cccDNA1', 'RNApol'), product='cccDNApol1', rate=Kbpol )
sim.modifyRegion('Nucleus').addReaction(reactant='cccDNApol1', product=( 'cccDNA1', 'RNApol'), rate=Kunb)

sim.modifyRegion('Nucleus').addReaction(reactant=('cccDNA2', 'RNApol'), product='cccDNApol2', rate=Kbpol )
sim.modifyRegion('Nucleus').addReaction(reactant='cccDNApol2', product=( 'cccDNA2', 'RNApol'), rate=Kunb)

sim.modifyRegion('Nucleus').addReaction(reactant=('cccDNA3', 'RNApol'), product='cccDNApol3', rate=Kbpol )
sim.modifyRegion('Nucleus').addReaction(reactant='cccDNApol3', product=( 'cccDNA3', 'RNApol'), rate=Kunb)

sim.modifyRegion('Nucleus').addReaction(reactant=('cccDNA4', 'RNApol'), product='cccDNApol4', rate=Kbpol )
sim.modifyRegion('Nucleus').addReaction(reactant='cccDNApol4', product=( 'cccDNA4', 'RNApol'), rate=Kunb)

sim.modifyRegion('Nucleus').addReaction(reactant=('cccDNA5', 'RNApol'), product='cccDNApol5', rate=Kbpol )
sim.modifyRegion('Nucleus').addReaction(reactant='cccDNApol5', product=( 'cccDNA5', 'RNApol'), rate=Kunb)

sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNApol1', product = ('pgRNA', 'RNApol', 'cccDNA1'), rate=K1) 
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNApol2', product = ('LSmRNA', 'RNApol', 'cccDNA2'), rate=K2)
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNApol3', product = ('SmRNA', 'RNApol', 'cccDNA3'), rate=K3)
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNApol4', product = ('XmRNA', 'RNApol', 'cccDNA4'), rate=K4)
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNApol5', product = ('preCmRNA', 'RNApol', 'cccDNA5'), rate=K5)

# Translation of proteins by Not explicitly binding to the ribosomes
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'pgRNA', product = ('pgRNA', 'dimer'), rate=0.013) # translation of RNAs
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'pgRNA', product = ('pgRNA', 'viralP'), rate=0.013) # translation of RNAs
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'LSmRNA', product = ('LSmRNA', 'L'), rate=0.0139) # translation of RNAs
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'SmRNA', product = ('LSmRNA', 'M'), rate=0.0156) # translation of RNAs
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'SmRNA', product = ('SmRNA', 'S'), rate=0.0156) # translation of RNAs 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'XmRNA', product = ('XmRNA', 'X'), rate=0.0476) # translation of RNAs  
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'preCmRNA', product = ('preCmRNA', 'preC'), rate=0.0526) # preC protein generates the HBceAg or E protein

sim.modifyRegion('Cytoplasm').addReaction(reactant=('pgRNA', 'viralP'), product='RNP', rate=Kbpol ) # approximated this with the human RNA polymerase
sim.modifyRegion('Cytoplasm').addReaction(reactant='RNP', product= ('pgRNA', 'viralP'), rate=Kunb )
kassemb =1.667e-3 # Assembly of empty capsid is 10 m according to Tresset
sim.modifyRegion('Cytoplasm').addReaction(reactant=('RNP'), product='capsidR', rate=0.2*kassemb)

kassemb =1.667e-3 # Assembly of empty capsid is 10 m according to Tresset
sim.modifyRegion('Cytoplasm').addReaction(reactant=('dimer','dimer'), product='capsidE', rate=kassemb )

#sim.modifyRegion('Cytoplasm').addReaction(reactant= 'dimer', product='', rate= 1) # Because of the change above I'll need the dimers 

# capsid maturation from RNA containing to rcDNA
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'capsidR', product='capsid', rate= math.log(2)/(12*3600)  ) 

Kb =4*3.1415*7.5e-12*45e-10*1000/Vvoxel  # 4*pi*(D1+D2)*(R1+R2)*1000*NA/(NA*V)
Kub = 0.041
####### COMPLETE VIRION FORMATION ######################

sim.modifyRegion('ER').addReaction(reactant=('L', 'capsid'), product='LC1', rate=Kb)
sim.modifyRegion('ER').addReaction(reactant= 'LC1', product = ('L', 'capsid'), rate=Kub)

for i in range (2, 21):
	j = i - 1
	sim.modifyRegion('ER').addReaction(reactant=('LC%d'%j, 'L'), product='LC%d'%i, rate=Kb)
	sim.modifyRegion('ER').addReaction(reactant= 'LC%d'%i, product = ('L', 'LC%d'%j), rate=Kub)

########## EMPTY VIRION FORMATION ###################

KbE =Kb*10  
KubE = Kub
sim.modifyRegion('ER').addReaction(reactant=('L', 'capsidE'), product='LCE1', rate=KbE)
sim.modifyRegion('ER').addReaction(reactant= 'LCE1', product = ('L', 'capsidE'), rate=KubE)

for i in range (2, 21):
	j = i - 1
	sim.modifyRegion('ER').addReaction(reactant=('LCE%d'%j, 'L'), product='LCE%d'%i, rate=KbE)
	sim.modifyRegion('ER').addReaction(reactant= 'LCE%d'%i, product = ('L', 'LCE%d'%j), rate=KubE)


########## RNA VIRION FORMATION ###################

KbR= Kb/50000
KubR = Kub

sim.modifyRegion('ER').addReaction(reactant=('L', 'capsidR'), product='LCR1', rate=KbR)
sim.modifyRegion('ER').addReaction(reactant= 'LCR1', product = ('L', 'capsidR'), rate=KubR)

for i in range (2, 21):
	j = i - 1
	sim.modifyRegion('ER').addReaction(reactant=('LCR%d'%j, 'L'), product='LCR%d'%i, rate=KbR)
	sim.modifyRegion('ER').addReaction(reactant= 'LCR%d'%i, product = ('L', 'LCR%d'%j), rate=KubR)


###### DEGRADATION REACTIONS ##########################

sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNA1', product = (''), rate = 1.6e-7 )
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNA2', product = (''), rate = 1.6e-7 )
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNA3', product = (''), rate = 1.6e-7 )
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNA4', product = (''), rate = 1.6e-7 )
sim.modifyRegion('Nucleus').addReaction(reactant= 'cccDNA5', product = (''), rate = 1.6e-7 )
sim.modifyRegion('Nucleus').addReaction(reactant= 'rcDNA', product = (''), rate = 1.6e-7 )


sim.modifyRegion('Cytoplasm').addReaction(reactant= 'pgRNA', product = (''), rate= 3.8e-5 ) 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'LSmRNA', product = (''), rate=  6.4e-5) 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'SmRNA', product = (''), rate= 6.4e-5 ) 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'XmRNA', product = (''), rate= 1.9e-4) 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'preCmRNA', product = (''), rate= 3.8e-5) 

degX = 2.9e-4  # half life of proteinX = 40 mins
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'dimer', product = (''), rate= degX ) # UPDATE ALL!! 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'viralP', product = (''), rate= degX )
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'L', product = (''), rate =  2.9e-4 )
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'S', product = (''), rate= degX  )
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'M', product = (''), rate=degX   )
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'X', product = (''), rate= degX ) 
sim.modifyRegion('Cytoplasm').addReaction(reactant= 'preC', product = (''), rate= degX )  

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  DIFFUSIONS %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
d_L = 22e-12 # scaled using Dill et al., based on D_GFP
d_capsid_C = 7.4e-12 
d_capsid_NPC = 16e-12 
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='capsid', rate= d_capsid_C )
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='capsidE', rate= d_capsid_C )
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='capsidR', rate= d_capsid_C )

# diffusions of capsids to NPCs
sim.setTwoWayTransitionRate(species='capsidR', one='Cytoplasm', two='NPC', rate= d_capsid_C )  # RNA-containing capsids 
sim.setTransitionRate(species='capsid', via='Cytoplasm', to='NPC', rate= d_capsid_C) # one way diffusion for complete capsid
sim.setTransitionRate(species='capsidE', via='Cytoplasm', to='NPC', rate= d_capsid_C) # one way diffusion for empty capsid

sim.modifyRegion('NPC').setDiffusionRate(species='capsid', rate= d_capsid_NPC)
sim.modifyRegion('NPC').setDiffusionRate(species='capsidE', rate= d_capsid_NPC)
sim.modifyRegion('NPC').setDiffusionRate(species='capsidR', rate= d_capsid_NPC)

d_RNA_NPC = 0.07e-12
sim.modifyRegion('NPC').setDiffusionRate(species='rcDNA', rate= d_RNA_NPC )
sim.modifyRegion('NPC').setDiffusionRate(species='dimer', rate= d_RNA_NPC )
sim.setTransitionRate(species='dimer', via='NPC', to='Cytoplasm', rate= d_RNA_NPC) # dimer has to go back to cytoplasm to make empty capsids 

d_DNA_N = 0.61e-12
sim.setTransitionRate(species='rcDNA', via='NPC', to='Nucleus', rate= d_DNA_N) 

sim.modifyRegion('Nucleus').setDiffusionRate(species='rcDNA', rate= d_DNA_N)

sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNA1', rate= d_DNA_N)
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNA2', rate= d_DNA_N)
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNA3', rate= d_DNA_N)
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNA4', rate= d_DNA_N)
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNA5', rate= d_DNA_N)

d_RNA_pol = 6.1e-13   # Kim et al., NAR, 2007
sim.modifyRegion('Nucleus').setDiffusionRate(species='RNApol', rate= d_RNA_pol)
 
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNApol1', rate= d_RNA_pol) 
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNApol2', rate= d_RNA_pol) 
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNApol3', rate= d_RNA_pol) 
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNApol4', rate= d_RNA_pol) 
sim.modifyRegion('Nucleus').setDiffusionRate(species='cccDNApol5', rate= d_RNA_pol) 

d_RNA_N = 0.61e-12
sim.modifyRegion('Nucleus').setDiffusionRate(species='pgRNA', rate= d_RNA_N )
sim.modifyRegion('Nucleus').setDiffusionRate(species='LSmRNA', rate= d_RNA_N )
sim.modifyRegion('Nucleus').setDiffusionRate(species='SmRNA', rate= d_RNA_N )
sim.modifyRegion('Nucleus').setDiffusionRate(species='XmRNA', rate= d_RNA_N )
sim.modifyRegion('Nucleus').setDiffusionRate(species='preCmRNA', rate= d_RNA_N )

d_RNA_C = 2.24e-12
# The RNAs get to the cytoplasm
sim.setTransitionRate(species='pgRNA', via='Nucleus', to='NPC', rate= d_RNA_N) 
sim.modifyRegion('NPC').setDiffusionRate(species='pgRNA', rate= d_RNA_NPC )
sim.setTransitionRate(species='pgRNA', via='NPC', to='Cytoplasm', rate= d_RNA_C ) 

sim.setTransitionRate(species='LSmRNA', via='Nucleus', to='NPC', rate= d_RNA_N ) 
sim.modifyRegion('NPC').setDiffusionRate(species='LSmRNA', rate= d_RNA_NPC )
sim.setTransitionRate(species='LSmRNA', via='NPC', to='Cytoplasm', rate= d_RNA_C ) 

sim.setTransitionRate(species='SmRNA', via='Nucleus', to='NPC', rate= d_RNA_N ) 
sim.modifyRegion('NPC').setDiffusionRate(species='SmRNA', rate= d_RNA_NPC)
sim.setTransitionRate(species='SmRNA', via='NPC', to='Cytoplasm', rate= d_RNA_C ) 

sim.setTransitionRate(species='XmRNA', via='Nucleus', to='NPC', rate= d_RNA_N ) 
sim.modifyRegion('NPC').setDiffusionRate(species='XmRNA', rate= d_RNA_NPC )
sim.setTransitionRate(species='XmRNA', via='NPC', to='Cytoplasm', rate= d_RNA_C ) 

sim.setTransitionRate(species='preCmRNA', via='Nucleus', to='NPC', rate= d_RNA_N ) 
sim.modifyRegion('NPC').setDiffusionRate(species='preCmRNA', rate= d_RNA_NPC )
sim.setTransitionRate(species='preCmRNA', via='NPC', to='Cytoplasm', rate= d_RNA_C ) 

sim.modifyRegion('Cytoplasm').setDiffusionRate(species='pgRNA', rate= d_RNA_C)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='LSmRNA', rate= d_RNA_C)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='SmRNA', rate= d_RNA_C)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='XmRNA', rate= d_RNA_C)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='preCmRNA', rate= d_RNA_C)

# diffusion of proteins

sim.modifyRegion('Cytoplasm').setDiffusionRate(species='dimer', rate= d_L ) 
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='viralP', rate= d_L ) 

sim.modifyRegion('Cytoplasm').setDiffusionRate(species='L', rate= d_L)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='M', rate= d_L)
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='S', rate= d_L)
# localization of newly-formed envelope proteins to ER
sim.setTransitionRate(species='L', via='Cytoplasm', to='ER', rate= d_L )
sim.setTransitionRate(species='M', via='Cytoplasm', to='ER', rate= d_L)
sim.setTransitionRate(species='S', via='Cytoplasm', to='ER', rate= d_L)
sim.modifyRegion('ER').setDiffusionRate(species='L', rate= d_L)
sim.modifyRegion('ER').setDiffusionRate(species='M', rate= d_L)
sim.modifyRegion('ER').setDiffusionRate(species='S', rate= d_L)

sim.modifyRegion('Cytoplasm').setDiffusionRate(species='X', rate= d_L) 
sim.modifyRegion('Cytoplasm').setDiffusionRate(species='preC', rate= d_L) 

sim.modifyRegion('Cytoplasm').setDiffusionRate(species='RNP', rate= d_RNA_C )

sim.modifyRegion('ER').setDiffusionRate(species='L', rate= d_L)
#sim.setTransitionRate(species='capsid', via='Cytoplasm', to='ER', rate= d_L)
sim.setTwoWayTransitionRate(species='capsid', one='Cytoplasm', two='ER', rate= d_capsid_C)
sim.setTwoWayTransitionRate(species='capsidE', one='Cytoplasm', two='ER', rate= d_capsid_C)

sim.modifyRegion('ER').setDiffusionRate(species='capsid', rate= d_capsid_C)
sim.modifyRegion('ER').setDiffusionRate(species='capsidE', rate= d_capsid_C)
sim.modifyRegion('ER').setDiffusionRate(species='capsidR', rate= d_capsid_C)

for i in range (1, 21):
        sim.modifyRegion('ER').setDiffusionRate(species='LC%d'%i, rate= d_L)
        sim.modifyRegion('Cytoplasm').setDiffusionRate(species='LC%d'%i, rate= d_L)
        sim.setTwoWayTransitionRate(species='LC%d'%i, one='Cytoplasm', two='ER', rate= d_L)
 # Empty virion
        sim.modifyRegion('ER').setDiffusionRate(species='LCE%d'%i, rate= d_L)
        sim.modifyRegion('Cytoplasm').setDiffusionRate(species='LCE%d'%i, rate= d_L)
        sim.setTwoWayTransitionRate(species='LCE%d'%i, one='Cytoplasm', two='ER', rate= d_L)
 # RNA virion
        sim.modifyRegion('ER').setDiffusionRate(species='LCR%d'%i, rate= d_L)
        sim.modifyRegion('Cytoplasm').setDiffusionRate(species='LCR%d'%i, rate= d_L)
        sim.setTwoWayTransitionRate(species='LCR%d'%i, one='ER', two='Cytoplasm', rate= d_L)

e=0
# Set up the ER
print("Loading ER")
ER = np.load ("ER.npy")
print("Discretizing")
lattice = sim.getLattice() # This just discretizes everything to the lattice
print("Setting ER")
for x in range(ER.shape[0]):
    for y in range(ER.shape[1]):
        for z in range(ER.shape[2]):
            idx = (x,y,z)
            if ER[x,y,z]:
                if sim.lattice.getSiteType(x,y,z) == 2:
                    sim.setLatticeSite(idx, "ER")
                    e +=1
                    sim.addParticleAtIdx(idx, 'L')


print("ER Done")

# Set simulation Parameters
dt = (nm(latticeSpacing)**2)/(2*22e-12)
print('dt =', dt)
sim.setTimestep(dt)
sim.setWriteInterval(10*dt)
sim.setLatticeWriteInterval(15000*dt)
sim.setSimulationTime(600)

sim.save(outputFile)

