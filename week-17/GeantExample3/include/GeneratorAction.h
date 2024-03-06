#ifndef GeneratorAction_h
#define GeneratorAction_h 1

#include "G4VUserPrimaryGeneratorAction.hh"
//#include "HepMC/GenEvent.h"
#include "HepMC/IO_GenEvent.h"

// Simplified version of Geant4 example
// extended/eventgenerator/HepMC/HepMCEx01/include/HepMCG4Interface.hh
class GeneratorAction : public G4VUserPrimaryGeneratorAction
{
  public:
    GeneratorAction();
    ~GeneratorAction() override;

    // Read a particle from the input file
    void GeneratePrimaries( G4Event* ) override;

  private:
    // HepMC ascii file reader
    HepMC::IO_GenEvent* m_asciiInput = nullptr;
};

#endif
