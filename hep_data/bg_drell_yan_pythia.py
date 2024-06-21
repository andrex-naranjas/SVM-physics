import sys
import numpy as np
import matplotlib.pyplot as plt
import math
import pythia8
import pandas as pd

# usage: python3 bg_drell_yan_pythia.py ZW 100000

process = str(sys.argv[1])
n_events = int(sys.argv[2])
# start Pythia
pythia = pythia8.Pythia()

if process=="Wq": # not working atm
    pythia.readString("WeakBosonAndParton:qg2Wq = on")  # Wq process
elif process=="ZW":
    pythia.readString("WeakDoubleBoson:ffbar2ZW = on")  # ZW process
elif process=="WW":
    pythia.readString("WeakDoubleBoson:ffbar2WW = on")  # WW process
elif process=="ALL":
    pythia.readString("WeakDoubleBoson:all = on")
elif process=="TTBAR":
    pythia.readString("Top:gg2ttbar = on")  # ttbar process OK
elif process=="SOFTQCD": # not working atm
    pythia.readString("SoftQCD:inelastic = on")  # Inelastic background process
else:
    sys.exit("proccess not supported")
    
pythia.init()

print("Simulating events for the "+process+"  background...")

mass_Z = []
cos_theta_Z = []
phi_Z = []
# electron
cos_theta_e = []
phi_e = []
pT_e = []
eta_e = []
pX_e = []
pY_e = []
pZ_e = []
e_e = []
# positron
cos_theta_pos = []
phi_pos = []
pT_pos = []
eta_pos = []
pX_pos = []
pY_pos = []
pZ_pos = []
e_pos = []

z_eta_og = []
reconstructed_Z_masses = []
z_mass_og = []
z_pt_og = []

pZ, pz, mass_z, eta_z, pt_z = 0, 0, 0, 0, 0

# generate events
#n_events = 10000
selected_events = 0
for i in range(n_events):
    if not pythia.next():
        continue
    wz_boson_found = False
    for particle in pythia.event:
        if particle.id() == 23 or abs(particle.id()) == 24:  # WZ boson
            # check if the Z boson mass is within a certain range
            z_mass = particle.m()
            if abs(z_mass - 91.1876) < 20.0:  # select Z bosons near the Z boson mass (within 10 GeV)
                wz_boson_found = True
                pZ = particle.p()
                pz = np.array([pZ[0], pZ[1], pZ[2]])
                mass_z = pZ.mCalc()
                eta_z = pZ.eta()
                pt_z = pZ.pT()
                # cos_theta_Z.append(pZ[3] / np.linalg.norm(pz))
                # phi_Z.append(math.atan2(pZ[1], pZ[0]))
                break
    if not wz_boson_found:
        continue
    electrons_found = []
    positrons_found = []
    for i in range(pythia.event.size()):
        particle = pythia.event[i]
        if particle.id() == 11:  # electron
            mother = particle.mother1()
            if pythia.event[mother].id() == 23 or abs(pythia.event[mother].id()) == 24 :  # require electron from W boson decay
                electrons_found.append(particle)
        elif particle.id() == -11:  # positron
            mother = particle.mother1()
            if pythia.event[mother].id() == 23 or abs(pythia.event[mother].id()) == 24:  # require positron from Z boson decay
                positrons_found.append(particle)
    # match electrons with positrons from the same Z boson decay
    for electron in electrons_found:
        for positron in positrons_found:
            if electron.mother1() == positron.mother1() or True:
                pElectron = electron.p()
                pPositron = positron.p()
                # get the reconstructed Z boson mass
                reconstructed_Z_mass = (pElectron + pPositron).mCalc()
                reconstructed_Z_masses.append(reconstructed_Z_mass)
                # get electron kinematics
                cos_theta_e.append(pElectron[3] / pElectron.pAbs())
                phi_e.append(math.atan2(pElectron[2], pElectron[1]))
                pT_e.append(pElectron.pT())
                eta_e.append(pElectron.eta())
                pX_e.append(pElectron[0])
                pY_e.append(pElectron[1])
                pZ_e.append(pElectron[2])
                e_e.append(pElectron.e())
                # get positron kinematics
                cos_theta_pos.append(pPositron[3] / pPositron.pAbs())
                phi_pos.append(math.atan2(pPositron[2], pPositron[1]))
                pT_pos.append(pPositron.pT())
                eta_pos.append(pPositron.eta())
                pX_pos.append(pPositron[0])
                pY_pos.append(pPositron[1])
                pZ_pos.append(pPositron[2])
                e_pos.append(pPositron.e())
                # Z boson kinematics
                cos_theta_Z.append(pZ[3] / np.linalg.norm(pz))
                phi_Z.append(math.atan2(pZ[1], pZ[0]))
                z_mass_og.append(mass_z)
                z_eta_og.append(eta_z)
                z_pt_og.append((pElectron + pPositron).pT())
                # z_pt_og.append(pt_z)
                
                selected_events += 1
                if selected_events == 10000:
                    break  # Stop after selecting a fixed number of events for each variable
        if selected_events == 10000:
            break  # Stop after selecting a fixed number of events for each variable
    if selected_events == 10000:
        break  # Stop after selecting a fixed number of events for each variable

# Create a single figure for all histograms
fig, axs = plt.subplots(3, 7, figsize=(20, 15), subplot_kw={'aspect': 'auto'})


# Plot Cosine of Theta for Z Boson
hist_mass_Z, bins_mass_Z, _ = axs[0, 0].hist(z_mass_og, bins=50, range=(60, 120))
axs[0, 0].set_title('Z Boson mass')
axs[0, 0].set_xlabel('mass')
axs[0, 0].set_ylabel('Counts')
axs[0, 0].text(0.05, 0.95, f'Integral: {np.sum(hist_mass_Z)}', ha='left', va='top', transform=axs[0, 0].transAxes)

# Plot Cosine of Theta for Z Boson
hist_Z, bins_Z, _ = axs[0, 1].hist(cos_theta_Z, bins=50, range=(-1, 1))
axs[0, 1].set_title('Cosine of Theta for Z Boson')
axs[0, 1].set_xlabel('cos(theta)')
axs[0, 1].set_ylabel('Counts')
axs[0, 1].text(0.05, 0.95, f'Integral: {np.sum(hist_Z)}', ha='left', va='top', transform=axs[0, 1].transAxes)

# Plot Phi for Z Boson
hist_phi_Z, bins_phi_Z, _ = axs[0, 2].hist(phi_Z, bins=50, range=(-math.pi, math.pi))
axs[0, 2].set_title('Phi for Z Boson')
axs[0, 2].set_xlabel('phi')
axs[0, 2].set_ylabel('Counts')
axs[0, 2].text(0.05, 0.95, f'Integral: {np.sum(hist_phi_Z)}', ha='left', va='top', transform=axs[0, 2].transAxes)

# Plot histogram of invariant masses
hist_eta_z, bins_eta_z, _ = axs[0, 3].hist(z_eta_og, bins=50)#, range=(60, 120))
axs[0, 3].set_xlabel('eta')
axs[0, 3].set_ylabel('Counts')
axs[0, 3].set_title('Z eta')
axs[0, 3].text(0.05, 0.95, f'Integral: {np.sum(hist_eta_z)}', ha='left', va='top', transform=axs[0, 3].transAxes)

# Plot histogram of reconstructed Z boson mass
hist_reco_Z_mass, bins_reco_Z_mass, _ = axs[0, 4].hist(reconstructed_Z_masses, bins=50, range=(60, 120)) #, histtype='step')
axs[0, 4].set_xlabel('Reco Z Mass')
axs[0, 4].set_ylabel('Counts')
axs[0, 4].set_title('Reco Z Boson Mass')
axs[0, 4].text(0.05, 0.95, f'Integral: {np.sum(hist_reco_Z_mass)}', ha='left', va='top', transform=axs[0, 4].transAxes)

# Plot histogram of z pt
hist_pt_z, bins_pt_z, _ = axs[0, 5].hist(z_pt_og, bins=50)#, range=(60, 120))
axs[0, 5].set_xlabel('pT')
axs[0, 5].set_ylabel('Counts')
axs[0, 5].set_title('Z pT')
axs[0, 5].text(0.05, 0.95, f'Integral: {np.sum(hist_eta_z)}', ha='left', va='top', transform=axs[0, 5].transAxes)


# electron variables
# cosine of Theta for electron
hist_e, bins_e, _ = axs[1, 0].hist(cos_theta_e, bins=50, range=(-1, 1))
axs[1, 0].set_title('Cosine of Theta for Electron')
axs[1, 0].set_xlabel('cos(theta)')
axs[1, 0].set_ylabel('Counts')
axs[1, 0].text(0.05, 0.95, f'Integral: {np.sum(hist_e)}', ha='left', va='top', transform=axs[1, 0].transAxes)

# Plot Phi for Electron
hist_phi_e, bins_phi_e, _ = axs[1, 1].hist(phi_e, bins=50, range=(-math.pi, math.pi))
axs[1, 1].set_title('Phi for Electron')
axs[1, 1].set_xlabel('phi')
axs[1, 1].set_ylabel('Counts')
axs[1, 1].text(0.05, 0.95, f'Integral: {np.sum(hist_phi_e)}', ha='left', va='top', transform=axs[1, 1].transAxes)

# Plot Eta for Electron
hist_eta_e, bins_eta_e, _ = axs[1, 2].hist(eta_e, bins=50)#, range=(-5, 5))
axs[1, 2].set_title('Eta for Electron')
axs[1, 2].set_xlabel('eta')
axs[1, 2].set_ylabel('Counts')
axs[1, 2].text(0.05, 0.95, f'Integral: {np.sum(hist_eta_e)}', ha='left', va='top', transform=axs[1, 2].transAxes)

# Plot pX for Electron
hist_px_e, bins_px_e, _ = axs[1, 3].hist(pX_e, bins=50)
axs[1, 3].set_title('pX for Electron')
axs[1, 3].set_xlabel('pX')
axs[1, 3].set_ylabel('Counts')
axs[1, 3].text(0.05, 0.95, f'Integral: {np.sum(hist_px_e)}', ha='left', va='top', transform=axs[1, 3].transAxes)

# Plot pY for Electron
hist_py_e, bins_py_e, _ = axs[1, 4].hist(pY_e, bins=50)
axs[1, 4].set_title('pY for Electron')
axs[1, 4].set_xlabel('pY')
axs[1, 4].set_ylabel('Counts')
axs[1, 4].text(0.05, 0.95, f'Integral: {np.sum(hist_py_e)}', ha='left', va='top', transform=axs[1, 4].transAxes)

# Plot pZ for Electron
hist_pz_e, bins_pz_e, _ = axs[1, 5].hist(pZ_e, bins=50)
axs[1, 5].set_title('pZ for Electron')
axs[1, 5].set_xlabel('pZ')
axs[1, 5].set_ylabel('Counts')
axs[1, 5].text(0.05, 0.95, f'Integral: {np.sum(hist_pz_e)}', ha='left', va='top', transform=axs[1, 5].transAxes)

# Plot pZ for Electron
hist_pt_e, bins_pt_e, _ = axs[1, 6].hist(pT_e, bins=50)
axs[1, 6].set_title('pT for Electron')
axs[1, 6].set_xlabel('pT')
axs[1, 6].set_ylabel('Counts')
axs[1, 6].text(0.05, 0.95, f'Integral: {np.sum(hist_pt_e)}', ha='left', va='top', transform=axs[1, 6].transAxes)


# positron plots
# cosine of Theta for positron
hist_pos, bins_pos, _ = axs[2, 0].hist(cos_theta_pos, bins=50, range=(-1, 1))
axs[2, 0].set_title('Cosine of Theta for positron')
axs[2, 0].set_xlabel('cos(theta)')
axs[2, 0].set_ylabel('Counts')
axs[2, 0].text(0.05, 0.95, f'Integral: {np.sum(hist_pos)}', ha='left', va='top', transform=axs[2, 0].transAxes)

# Plot Phi for positron
hist_phi_pos, bins_phi_pos, _ = axs[2, 1].hist(phi_pos, bins=50, range=(-math.pi, math.pi))
axs[2, 1].set_title('Phi for positron')
axs[2, 1].set_xlabel('phi')
axs[2, 1].set_ylabel('Counts')
axs[2, 1].text(0.05, 0.95, f'Integral: {np.sum(hist_phi_pos)}', ha='left', va='top', transform=axs[2, 1].transAxes)

# Plot Eta for positron
hist_eta_pos, bins_eta_pos, _ = axs[2, 2].hist(eta_pos, bins=50) #, range=(-5, 5))
axs[2, 2].set_title('Eta for positron')
axs[2, 2].set_xlabel('eta')
axs[2, 2].set_ylabel('Counts')
axs[2, 2].text(0.05, 0.95, f'Integral: {np.sum(hist_eta_pos)}', ha='left', va='top', transform=axs[2, 2].transAxes)

# Plot pX for positron
hist_px_pos, bins_px_pos, _ = axs[2, 3].hist(pX_pos, bins=50)
axs[2, 3].set_title('pX for positron')
axs[2, 3].set_xlabel('pX')
axs[2, 3].set_ylabel('Counts')
axs[2, 3].text(0.05, 0.95, f'Integral: {np.sum(hist_px_pos)}', ha='left', va='top', transform=axs[2, 3].transAxes)

# Plot pY for positron
hist_py_pos, bins_py_pos, _ = axs[2, 4].hist(pY_pos, bins=50)
axs[2, 4].set_title('pY for positron')
axs[2, 4].set_xlabel('pY')
axs[2, 4].set_ylabel('Counts')
axs[2, 4].text(0.05, 0.95, f'Integral: {np.sum(hist_py_pos)}', ha='left', va='top', transform=axs[2, 4].transAxes)

# Plot pZ for positron
hist_pz_pos, bins_pz_pos, _ = axs[2, 5].hist(pZ_pos, bins=50)
axs[2, 5].set_title('pZ for positron')
axs[2, 5].set_xlabel('pZ')
axs[2, 5].set_ylabel('Counts')
axs[2, 5].text(0.05, 0.95, f'Integral: {np.sum(hist_pz_pos)}', ha='left', va='top', transform=axs[2, 5].transAxes)

# Plot pZ for positron
hist_pt_pos, bins_pt_pos, _ = axs[2, 6].hist(pT_pos, bins=50)
axs[2, 6].set_title('pT for positron')
axs[2, 6].set_xlabel('pT')
axs[2, 6].set_ylabel('Counts')
axs[2, 6].text(0.05, 0.95, f'Integral: {np.sum(hist_pt_e)}', ha='left', va='top', transform=axs[2, 6].transAxes)

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)
plt.savefig("./plots/dy_bg_"+process+".pdf")

# Create a dictionary to store the data
data = {
    # 'cos_theta_Z': cos_theta_Z,
    # 'phi_Z': phi_Z,
    'cos_theta_e': cos_theta_e,
    'phi_e': phi_e,
    'pT_e': pT_e,
    'eta_e': eta_e,
    'pX_e': pX_e,
    'pY_e': pY_e,
    'pZ_e': pZ_e,
    'e_e': e_e,
    'cos_theta_pos': cos_theta_pos,
    'phi_pos': phi_pos,
    'pT_pos': pT_pos,
    'eta_pos': eta_pos,    
    'pX_pos': pX_pos,
    'pY_pos': pY_pos,
    'pZ_pos': pZ_pos,
    'e_pos': e_pos,
    'z_eta_og': z_eta_og,
    'z_mass_og': z_mass_og,
    'z_pt_og': z_pt_og,
    'reco_Z_masses': reconstructed_Z_masses
}

# Create a DataFrame from the dictionary
df = pd.DataFrame(data)
print(df.head())

# save the DataFrame to a CSV file
df.to_csv('./data/events_data_bkg_'+process+'.csv', index=False)

# end of the Pythia instance
pythia.stat()
