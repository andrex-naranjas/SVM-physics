import pythia8
import matplotlib.pyplot as plt
import numpy as np

# Create a Pythia instance
pythia = pythia8.Pythia()

# Set up the Pythia configuration
pythia.readString("Beams:eCM = 13000.")  # Set the center-of-mass energy to 13 TeV
pythia.readString("HardQCD:all = on")    # Enable hard QCD processes

# Enable charmed baryon decays
pythia.readString("4332:mayDecay = on")  # Enable Omega_c0 decay
pythia.readString("4132:mayDecay = on")  # Enable Xi_c0 decay
pythia.readString("4232:mayDecay = on")  # Enable Xi_c+ decay

# Initialize Pythia
pythia.init()

# Lists to store kinematics of final states
pt_list_omega_c = []
eta_list_omega_c = []
phi_list_omega_c = []

pt_list_xic = []
eta_list_xic = []
phi_list_xic = []

pt_list_kaon = []
eta_list_kaon = []
phi_list_kaon = []

# Lists to store kinematics of Xi_c decay products
pt_list_proton = []
eta_list_proton = []
phi_list_proton = []

pt_list_xic_kaon = []
eta_list_xic_kaon = []
phi_list_xic_kaon = []

pt_list_pion = []
eta_list_pion = []
phi_list_pion = []

# List to store invariant masses
invariant_mass_list = []

# Generate events
for iEvent in range(10000000):
    # Generate an event
    if not pythia.next():
        continue
    
    # Analyze the event
    for i in range(pythia.event.size()):
        particle = pythia.event[i]
        if particle.id() == 4332:  # Check for Omega_c0 baryons
            pt_list_omega_c.append(particle.pT())
            eta_list_omega_c.append(particle.eta())
            phi_list_omega_c.append(particle.phi())
            for daughter_index in particle.daughterList():
                daughter = pythia.event[daughter_index]
                if abs(daughter.id()) in [4132, 4232]:  # Xi_c baryons
                    pt_list_xic.append(daughter.pT())
                    eta_list_xic.append(daughter.eta())
                    phi_list_xic.append(daughter.phi())
                    # Analyze Xi_c decay products
                    protons = []
                    kaons = []
                    pions = []
                    for grand_daughter_index in daughter.daughterList():
                        grand_daughter = pythia.event[grand_daughter_index]
                        if abs(grand_daughter.id()) == 2212:  # Protons
                            pt_list_proton.append(grand_daughter.pT())
                            eta_list_proton.append(grand_daughter.eta())
                            phi_list_proton.append(grand_daughter.phi())
                            protons.append(grand_daughter)
                        elif abs(grand_daughter.id()) in [311, 321]:  # Kaons
                            pt_list_xic_kaon.append(grand_daughter.pT())
                            eta_list_xic_kaon.append(grand_daughter.eta())
                            phi_list_xic_kaon.append(grand_daughter.phi())
                            kaons.append(grand_daughter)
                        elif abs(grand_daughter.id()) in [211, 111]:  # Pions
                            pt_list_pion.append(grand_daughter.pT())
                            eta_list_pion.append(grand_daughter.eta())
                            phi_list_pion.append(grand_daughter.phi())
                            pions.append(grand_daughter)
                    
                    # Calculate invariant mass if we have at least one proton, one kaon, and one pion
                    if protons and kaons and pions:
                        for proton in protons:
                            for kaon in kaons:
                                for pion in pions:
                                    e_total = proton.e() + kaon.e() + pion.e()
                                    px_total = proton.px() + kaon.px() + pion.px()
                                    py_total = proton.py() + kaon.py() + pion.py()
                                    pz_total = proton.pz() + kaon.pz() + pion.pz()
                                    invariant_mass = np.sqrt(e_total**2 - px_total**2 - py_total**2 - pz_total**2)
                                    invariant_mass_list.append(invariant_mass)

# Plot the kinematics for Omega_c
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Omega_c^0', fontsize=16)

# Transverse Momentum (pT) plot for Omega_c
axs[0].hist(pt_list_omega_c, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Omega_c^0')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Omega_c
axs[1].hist(eta_list_omega_c, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Omega_c^0')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Omega_c
axs[2].hist(phi_list_omega_c, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Omega_c^0')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('omega_c_kinematics.png')

# Plot the kinematics for Xi_c
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Xi_c Decay Products', fontsize=16)

# Transverse Momentum (pT) plot for Xi_c
axs[0].hist(pt_list_xic, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Xi_c Decay Products')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Xi_c
axs[1].hist(eta_list_xic, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Xi_c Decay Products')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Xi_c
axs[2].hist(phi_list_xic, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Xi_c Decay Products')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('xic_decay_products.png')

# Plot the kinematics for Kaons
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Kaon Decay Products', fontsize=16)

# Transverse Momentum (pT) plot for Kaons
axs[0].hist(pt_list_kaon, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Kaon Decay Products')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Kaons
axs[1].hist(eta_list_kaon, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Kaon Decay Products')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Kaons
axs[2].hist(phi_list_kaon, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Kaon Decay Products')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('kaon_decay_products.png')

# Plot the kinematics for Xi_c decay products: Protons
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Xi_c Decay Products: Protons', fontsize=16)

# Transverse Momentum (pT) plot for Protons
axs[0].hist(pt_list_proton, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Protons')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Protons
axs[1].hist(eta_list_proton, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Protons')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Protons
axs[2].hist(phi_list_proton, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Protons')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('proton_decay_products.png')

# Plot the kinematics for Xi_c decay products: Kaons
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Xi_c Decay Products: Kaons', fontsize=16)

# Transverse Momentum (pT) plot for Kaons
axs[0].hist(pt_list_xic_kaon, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Kaons')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Kaons
axs[1].hist(eta_list_xic_kaon, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Kaons')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Kaons
axs[2].hist(phi_list_xic_kaon, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Kaons')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('xic_kaon_decay_products.png')

# Plot the kinematics for Xi_c decay products: Pions
fig, axs = plt.subplots(3, 1, figsize=(10, 15))
fig.suptitle('Kinematics of Xi_c Decay Products: Pions', fontsize=16)

# Transverse Momentum (pT) plot for Pions
axs[0].hist(pt_list_pion, bins=50, color='blue', alpha=0.7)
axs[0].set_title('Transverse Momentum (pT) of Pions')
axs[0].set_xlabel('pT [GeV]')
axs[0].set_ylabel('Counts')

# Pseudorapidity (eta) plot for Pions
axs[1].hist(eta_list_pion, bins=50, color='green', alpha=0.7)
axs[1].set_title('Pseudorapidity (eta) of Pions')
axs[1].set_xlabel('eta')
axs[1].set_ylabel('Counts')

# Azimuthal Angle (phi) plot for Pions
axs[2].hist(phi_list_pion, bins=50, color='red', alpha=0.7)
axs[2].set_title('Azimuthal Angle (phi) of Pions')
axs[2].set_xlabel('phi [rad]')
axs[2].set_ylabel('Counts')

# Adjust layout and save the figure
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('pion_decay_products.png')

# Plot the invariant mass of the grand daughters
plt.figure(figsize=(10, 6))
plt.hist(invariant_mass_list, bins=50, color='purple', alpha=0.7)
plt.title('Invariant Mass of Proton, Kaon, and Pion from Xi_c Decays')
plt.xlabel('Invariant Mass [GeV]')
plt.ylabel('Counts')
plt.savefig('invariant_mass.png')

# Print statistics
pythia.stat()
