#include "RunAction.h"

#include "g4csv.hh"

RunAction::RunAction() : G4UserRunAction()
{
  // Create analysis manager
  auto analysisManager = G4AnalysisManager::Instance();

  // Add an ntuple for truth (ntuple id 0)
  analysisManager->CreateNtuple( "Truth", "Truth information" );
  analysisManager->CreateNtupleIColumn( "EventNumber" );
  analysisManager->CreateNtupleDColumn( "Phi" );
  analysisManager->CreateNtupleDColumn( "Theta" );
  analysisManager->CreateNtupleDColumn( "Momentum" );
  analysisManager->FinishNtuple();

  // Add an ntuple for tracker layer 1 (ntuple id 1)
  analysisManager->CreateNtuple( "Tracker1", "Tracker 1 coordinates" );
  analysisManager->CreateNtupleIColumn( "EventNumber" );
  analysisManager->CreateNtupleDColumn( "Phi" );
  analysisManager->CreateNtupleDColumn( "Theta" );
  analysisManager->FinishNtuple();

  // Add an ntuple for tracker layer 2 (ntuple id 2)
  analysisManager->CreateNtuple( "Tracker2", "Tracker 2 coordinates" );
  analysisManager->CreateNtupleIColumn( "EventNumber" );
  analysisManager->CreateNtupleDColumn( "Phi" );
  analysisManager->CreateNtupleDColumn( "Theta" );
  analysisManager->FinishNtuple();
}

RunAction::~RunAction()
{
  // Delete analysis manager
  delete G4AnalysisManager::Instance();
}

void RunAction::BeginOfRunAction( const G4Run* )
{
  // Get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();

  // Open an output file
  analysisManager->OpenFile( "output.csv" );
}
 
void RunAction::EndOfRunAction( const G4Run* )
{
  // Get analysis manager
  auto analysisManager = G4AnalysisManager::Instance();

  // Save output data
  analysisManager->Write();
  analysisManager->CloseFile();
}
