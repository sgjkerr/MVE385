
import numpy as np  # summation
import re
import os
from scipy.signal import find_peaks  # peak detection
import shutil
import time
import pickle
import pandas as pd
import csv
from rdkit import Chem
from rdkit.Chem.rdmolops import RemoveAllHs
from rdkit.Chem.rdmolfiles import MolFromPDBFile, MolToSmiles
from rdkit.Chem.inchi import InchiReadWriteError


w_nm = 10.0 # w = line width for broadening - nm, FWHM

def read_dftb_output(spectrum_file, line_start=5, line_end=55):
    energylist = list()  # energy cm-1
    intenslist = list()  # fosc

    count_line = 0
    for line in spectrum_file:
        line = line.decode('utf-8')
        # only recognize lines that start with number
        # split line into 3 lists mode, energy, intensities
        # line should start with a number
        if line_start <= count_line <= line_end:
            if re.search("\d\s{1,}\d", line):
                energylist.append(float(line.strip().split()[0]))
                intenslist.append(float(line.strip().split()[1]))
        else:
            pass
        count_line = count_line + 1
    return energylist, intenslist

def energy_to_wavelength(eVlist, intenslist):
    # convert wave number to nm for nm plot
    valuelist = [convert_ev_in_nm(value) for value in eVlist]
    # Combine the two lists into a list of tuples
    combined = list(zip(valuelist, intenslist))
    # Sort the list of tuples based on the values in list1
    sorted_combined = sorted(combined, key=lambda x: x[0])
    # Separate the two lists
    valuelist, intenslist = zip(*sorted_combined)
    return valuelist, intenslist

def convert_ev_in_nm(ev_value):
    planck_constant = 4.1357 * 1e-15  # eV s
    light_speed = 299792458  # m / s
    meter_to_nanometer_conversion = 1e+9
    return 1 / ev_value * planck_constant * light_speed * meter_to_nanometer_conversion
    
def smooth_spectrum(valuelist, intenslist, encoder):
    w = w_nm  # use line width for nm axis
    data = bins_to_spectrum(valuelist, w, intenslist, encoder)
    return data

def gauss(a, m, x, w):
    # calculation of the Gaussian line shape
    # a = amplitude (max y, intensity)
    # x = position
    # m = maximum/median (stick position in x, wave number)
    # w = line width, FWHM
    return a * np.exp(-(np.log(2) * ((m - x) / w) ** 2))

def bins_to_spectrum(valuelist, w, intenslist, encoder):
    spectrum_discretization_step = encoder['resolution']
    xmin_spectrum = 0
    xmax_spectrum = encoder['max_wavelength'] + spectrum_discretization_step
    xmax_spectrum_tmp = xmax_spectrum*2

    gauss_sum = list()  # list for the sum of single gaussian spectra = the convoluted spectrum

    # plotrange must start at 0 for peak detection
    x = np.arange(xmin_spectrum, xmax_spectrum_tmp, spectrum_discretization_step)

    # plot single gauss function for every frequency freq
    # generate summation of single gauss functions
    for index, wn in enumerate(valuelist):
        # sum of gauss functions
        gauss_sum.append(gauss(intenslist[index], x, wn, w))

    # y values of the gauss summation
    gauss_sum_y = np.sum(gauss_sum, axis=0)
    gauss_sum_y = gauss_sum_y / np.sum(gauss_sum_y)
    # Compute center of mass
    center_of_mass = np.sum(x * gauss_sum_y) / np.sum(gauss_sum_y)
    
    # normalize
    # find the index of x = encoder['min_wavelength'] in x
    index_min = int(encoder['min_wavelength']/spectrum_discretization_step)
    
    x = x[index_min:int(len(x)/2)]
    gauss_sum_y = gauss_sum_y[index_min:int(len(gauss_sum_y)/2)]
    #gauss_sum_y = gauss_sum_y / np.max(gauss_sum_y)

    xdata = x
    ydata = gauss_sum_y
    xlimits = [np.min(xdata), np.max(xdata)]

    y = []
    for elements in range(len(xdata)):
        if xlimits[0] <= xdata[elements] <= xlimits[1]:
            y.append(ydata[elements])
    return y, center_of_mass

def get_bar_spectra(valuelist, intenslist, encoder):
    spectrum_discretization_step = encoder['resolution']
    xmin_spectrum = encoder['min_wavelength']
    xmax_spectrum = encoder['max_wavelength'] + spectrum_discretization_step
    
    breaks = int((xmax_spectrum - xmin_spectrum) / spectrum_discretization_step)
    y = np.zeros(breaks)
    for ind, value in enumerate(valuelist):
        if xmin_spectrum <= value <= xmax_spectrum:
            y[int((value-xmin_spectrum)/spectrum_discretization_step)] += intenslist[ind]
    
    #y = y/np.max(y)
    return list(y)

def get_mol_spectra(source_file, member, tar, encoder):
    try:
        with tar.extractfile(member) as f:
            eVlist, intenslist = read_dftb_output(f)
            min_wavelength = convert_ev_in_nm(min(eVlist))
            max_wavelength = convert_ev_in_nm(max(eVlist))
            valuelist, intenslist =  energy_to_wavelength(eVlist, intenslist)
            data_bar = get_bar_spectra(valuelist, intenslist, encoder)
            data, center_of_mass = smooth_spectrum(valuelist, intenslist, encoder)    
    except FileNotFoundError:
        print(f"'EXC.DAT' not found in {member.name}")
        data = None
    return data, data_bar, center_of_mass

def get_gross_charge(source_file, member, tar, encoder):
    try:
        with tar.extractfile(member) as f:
            # Read lines 15 to empty line in file f
            lines = f.readlines()
            fourth_from_bottom = lines[-4]
            dipole_moment_str = fourth_from_bottom.split()[2:5]
            features = [float(value) for value in dipole_moment_str]
            feature_inds = [-12, -14, -15, -21, -22]
            for feature_ind in feature_inds:
                feature = lines[feature_ind]
                feature = feature.split()[-4]
                feature = float(feature)
                if feature_ind in [-15, -21]:
                    feature = np.round(feature/100.0, 10)
                features += [feature]            
            
    except FileNotFoundError:
        print(f"'detailed.out' not found in {member.name}")
        features = None
    return features

def get_xyz_from_gen_file(data, member, tar, encoder):
    if encoder['conf_type'] == 'ornl':
        try:
            pos = []
            with tar.extractfile(member) as f:
                for ind, line in enumerate(f):
                    # Split the line by whitespace
                    line = line.decode('utf-8').split()
                    if ind == 0:
                        num_atoms = int(line[0])
                    elif ind == 1:
                        atom_type = line
                        
                    else:
                        xyz_pos_atom = [float(value) for value in line[2::]]
                        pos.append(xyz_pos_atom) 
                xyz_atoms = np.array(pos)
                
                # center the x,y,z-coordinates
                centroid = np.mean(xyz_atoms, axis=0)
                xyz_atoms -= centroid
                
                # concatenate with atom attributes
                xyz_atoms = xyz_atoms.tolist()

                for ind, xyz_atom in enumerate(xyz_atoms):
                #    xyz_atom += atom_type_one_hots[ind]
                    data.append(xyz_atom)
                    
                    
        except FileNotFoundError:
            print(f"'geo_end.gen' not found in {member.name}")
            data = None
    return data


def get_mol_attributes_file(data, member, tar, encoder):
    try:
        tar.extract(member, path='./datasets/tmp/')
        # Load smiles from pdb file
        path = './datasets/tmp/'+member.name
        mol = MolFromPDBFile(path, sanitize=False, removeHs=False, proximityBonding=True)
        split_path = path.split('/')[:-1]
        path = '/'.join(split_path)
        shutil.rmtree(path)
        for i, atom in enumerate(mol.GetAtoms()):
            attribute_tmp = []
            attribute_tmp += encoder['atom_type'][atom.GetSymbol()]
            attribute_tmp.append(atom.GetDegree())
            attribute_tmp.append(atom.GetExplicitValence())
            attribute_tmp.append(atom.GetMass()/100)
            attribute_tmp.append(atom.GetFormalCharge())
            #attribute_tmp.append(atom.GetNumImplicitHs())
            attribute_tmp.append(int(atom.GetIsAromatic()))
            attribute_tmp.append(int(atom.IsInRing()))
            data.append(attribute_tmp)
        

    except FileNotFoundError:
        print(f"'smiles.pdb' not found in {member.name}")
        data = None
    
    return data

def read_smiles_from_csv_w_pandas(mol, df):
    smiles = None
    inchi = None
    inchikey = None
    # Search for the row where the first column matches the search string
    matching_row = df[df.iloc[:, 0] == mol]

    # Check if a matching row was found
    if not matching_row.empty:
        # Save the string from the second column of the matching row
        smiles = matching_row.iloc[0, 1]
        try:
            mol = Chem.MolFromSmiles(smiles)
            inchi = Chem.MolToInchi(mol, options='-SNon')
            inchikey = Chem.InchiToInchiKey(inchi)
        except:
            print(f"An error occurred:")
            return None, None, None
        
    else:
        print('No matching string found in the first column.')
    
    return smiles, inchi, inchikey
    