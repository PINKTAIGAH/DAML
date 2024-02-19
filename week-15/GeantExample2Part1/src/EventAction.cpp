#include "EventAction.h"

// #include "g4csv.hh"
#include "G4CsvAnalysisManager.hh"

EventAction::EventAction()
{
}

EventAction::~EventAction()
{
}

void EventAction::BeginOfEventAction( const G4Event* )
{
}

void EventAction::EndOfEventAction( const G4Event* )
{
  // Finish the ntuple row
  auto analysisManager = G4CsvAnalysisManager::Instance();
  analysisManager->AddNtupleRow( 0 );
}
