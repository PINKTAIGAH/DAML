#include "GeneratorAction.h"

#include "G4RunManager.hh"
#include "g4csv.hh"
#include "G4LorentzVector.hh"
#include "G4PrimaryParticle.hh"
#include "G4PrimaryVertex.hh"
#include "G4TransportationManager.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"


// Simplified version of Geant4 example
// extended/eventgenerator/HepMC/HepMCEx01/src/HepMCG4Interface.cc
// My only addition is the truth information output at the end
GeneratorAction::GeneratorAction()
{
  // Load my input file
  m_asciiInput = new HepMC::IO_GenEvent( "data/1000.mumu.dat", std::ios::in );
  // m_asciiInput = new HepMC::IO_GenEvent( "data/1000.ee.dat", std::ios::in );
}

GeneratorAction::~GeneratorAction()
{
  delete m_asciiInput;
}

void GeneratorAction::GeneratePrimaries( G4Event* anEvent )
{
  // Load next event from file
  HepMC::GenEvent* hepmcEvent = m_asciiInput->read_next_event();
  if ( !hepmcEvent )
  {
    G4cout << "HepMCInterface: no generated particles. Run terminated..." << G4endl;
    G4RunManager::GetRunManager()->AbortRun();
    return;
  }

  // Loop over all event vertices
  for( auto vitr = hepmcEvent->vertices_begin(); vitr != hepmcEvent->vertices_end(); ++vitr )
  {
    // Check that the vertex has valid particles
    G4bool valid = false;
    for ( auto pitr= (*vitr)->particles_begin( HepMC::children ); pitr != (*vitr)->particles_end( HepMC::children ); ++pitr)
    {
      if ( !(*pitr)->end_vertex() && (*pitr)->status() == 1 )
      {
        valid = true;
        break;
      }
    }
    if ( !valid ) continue; // skip to next vertex in loop

    // Find vertex position 
    HepMC::FourVector pos = (*vitr)->position();
    G4LorentzVector xvtx( pos.x(), pos.y(), pos.z(), pos.t() );

    // Check the vertex is inside the world
    G4Navigator* navigator = G4TransportationManager::GetTransportationManager()->GetNavigatorForTracking();
    G4VPhysicalVolume* world = navigator->GetWorldVolume();
    G4VSolid* solid = world->GetLogicalVolume()->GetSolid();
    EInside qinside = solid->Inside( xvtx.vect()*mm );
    if ( qinside != kInside ) continue; // skip to next vertex in loop

    // Make the vertex (input lengths in mm)
    G4PrimaryVertex* g4vtx = new G4PrimaryVertex( xvtx.x()*mm,
                                                  xvtx.y()*mm,
                                                  xvtx.z()*mm,
                                                  xvtx.t()*mm / c_light );

    // Loop over all particles for this vertex
    for ( auto vpitr = (*vitr)->particles_begin( HepMC::children ); vpitr != (*vitr)->particles_end( HepMC::children ); ++vpitr )
    {
      if ( (*vpitr)->status() != 1 ) continue; // skip to next particle in loop

      // Make the particle (input energies in MeV)
      G4int pdgcode = (*vpitr)->pdg_id();
      HepMC::FourVector mom = (*vpitr)->momentum();
      G4PrimaryParticle* g4prim = new G4PrimaryParticle( pdgcode, mom.px()*MeV, mom.py()*MeV, mom.pz()*MeV );

      // Output truth information for muons ( pdg 13 ) or electrons ( pdg 11 )
      if ( pdgcode == 13 || pdgcode == -13 )
      //if ( pdgcode == 11 || pdgcode == -11 )
      {
        // Get analysis manager
        auto analysisManager = G4AnalysisManager::Instance();

        // Fill ntuple
        analysisManager->FillNtupleIColumn( 0, 0, anEvent->GetEventID() );
        analysisManager->FillNtupleDColumn( 0, 1, g4prim->GetMomentum().phi() );
        analysisManager->FillNtupleDColumn( 0, 2, g4prim->GetMomentum().theta() );
        // Divide by the unit when outputting
        // see http://geant4.web.cern.ch/sites/geant4.web.cern.ch/files/geant4/collaboration/working_groups/electromagnetic/gallery/units/SystemOfUnits.html
        analysisManager->FillNtupleDColumn( 0, 3, g4prim->GetMomentum().mag()/GeV );
        analysisManager->AddNtupleRow( 0 );
      }

      // Add my particle to the vertex
      g4vtx->SetPrimary( g4prim );
    }

    // Add my vertex to the event
    anEvent->AddPrimaryVertex( g4vtx );
  }
}
