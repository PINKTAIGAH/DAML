#include "EventAction.h"

#include "g4csv.hh"
#include "G4VProcess.hh"

EventAction::EventAction()
{
}

EventAction::~EventAction()
{
}

void EventAction::BeginOfEventAction( const G4Event* )
{
  m_graphEdges.clear();
  m_bremPhotons.clear();
  m_allParticles.clear();
  m_conversionPositrons.clear();
}

void EventAction::EndOfEventAction( const G4Event* )
{
  // Finish the ntuple row
  auto analysisManager = G4AnalysisManager::Instance();
  analysisManager->AddNtupleRow( 0 );

  // Print track history graph - very large!
  //for ( auto edge : m_graphEdges )
  //{
  //  std::cout << edge.first << "," << edge.second << std::endl;
  //}

  // Print particle totals
  std::cout << "\t##### New Event #####\t" << std::endl;
  std::cout << "Total particles in the event: " << m_allParticles.size() << std::endl;
  std::cout << "Photons produced from electron Bremsstrahlung: " << m_bremPhotons.size() << std::endl;
  std::cout << "Number of positrons produced from photon conversion: " << m_conversionPositrons.size() << std::endl;
  std::cout << std::endl;
}

void EventAction::RecordParticleData( const G4Step* step )
{
  // Get the track that this step belongs to
  G4Track * track = step->GetTrack();

  // Get the type of particle that made this track
  std::string particle = track->GetParticleDefinition()->GetParticleName();

  // Find out how the track was created
  // have to check if the pointer is valid: it doesn't exist for intermediate steps
  if ( track->GetCreatorProcess() )
  {
    std::string creator = track->GetCreatorProcess()->GetProcessName();

    if ( creator == "eBrem" && particle == "gamma" )
    {
      // Use a set so that track IDs are only stored once
      m_bremPhotons.insert( track->GetTrackID() );
    }
  }

  // Check if the creation process of the track is due to conversion and is a positron
  if (track->GetCreatorProcess()){
    // Get string with creation process
    std::string creator = track->GetCreatorProcess()->GetProcessName();

    // Append trackID if creator is conversion and particle is positron
    if (creator=="conv" && particle=="e+"){
      // Apprnd track ID to set
      m_conversionPositrons.insert(track->GetTrackID());
    } 
  }

  // Collect all unique particle IDs
  m_allParticles.insert( track->GetTrackID() );

  // Make a graph edge connecting a new track ID to its parent
  std::pair< int, int > edge = std::make_pair( track->GetParentID(), track->GetTrackID() );
  m_graphEdges.insert( edge );
}
