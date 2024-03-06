# Local
source $GEANT4_myPath/bin/geant4.sh

# CPlab
# source /Disk/opt/ubuntu/geant4.10.06.p02/bin/geant4.sh

#HEPMC
export HEPMC_DIR=$PWD/HEPmc/install
if [[ ! -e "${HEPMC_DIR}" ]]; then
    cd HEPmc
    source HEPMCinstall.sh
    cd ..
fi

#Conda
conda activate daml
