from rdkit import Chem
from rdkit import RDLogger
import os
import glob

RDLogger.DisableLog('rdApp.*')
def create_msp_file(mol_directory, filename):
    # Get the list of all 'smiles.pdb' files in the directory
    mol_files = sorted(glob.glob(os.path.join(mol_directory, 'mol_*', 'smiles.pdb')))

    #Select filders within range x:y
    #mol_files = mol_files[x:y]

    with open(filename, 'w') as f:
        for mol_file in mol_files:
            mol = Chem.MolFromPDBFile(mol_file)
            #Check for faulty molecules and skip (Less than 0.01%)
            if mol is None:
                print(f"Failed to read molecule from file: {mol_file}")
                continue
            inchi = Chem.MolToInchi(mol)
            inchikey = Chem.InchiToInchiKey(inchi)
            exc_data_path = os.path.join(os.path.dirname(mol_file), 'EXC.DAT')
            smiles = Chem.MolToSmiles(mol)

            with open(exc_data_path, 'r') as exc_file:
                lines = exc_file.readlines()

            peaks = []
            for line in lines:
                if not line.strip() or line.startswith('='):
                    continue

                parts = line.split()
                try:
                    #Convert to nm
                    mass = 1240/float(parts[0])
                    intensity = float(parts[1])
                    peaks.append({'mass': mass, 'intensity': intensity})
                except ValueError:
                    continue

            peaks.sort(key=lambda peak: peak['mass'])

            #Write up all the names in an .msp file according to MoNA format
            f.write('Name: ' + os.path.basename(os.path.dirname(mol_file)) + '\n')
            f.write('InChIKey: ' + inchikey + '\n')
            f.write('Precursor_type: [M+H]\n') #Optional
            f.write('Spectrum_type: MS2\n') #Optional
            f.write('PrecursorMZ: 285\n') #Optional
            f.write('Ion_mode: P\n') #Optional
            f.write('MW: 285 \n')
            f.write('ExactMass: 285 \n')
            f.write('Comments: "computed SMILES=' + smiles + '" "InChI=' + inchi + '"\n')
            f.write('Num Peaks: ' + str(len(peaks)) + '\n')
            for peak in peaks:
                f.write(str(peak['mass']) + ' ' + str(peak['intensity']) + '\n')
            f.write('\n')  # Separate each molecule with an empty line

create_msp_file('Insert your directory', 'all_data.msp') #Path
