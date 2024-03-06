#include "DetectorConstruction.h"
#include "PositionFinder.h"

#include "G4Material.hh"
#include "G4NistManager.hh"
#include "G4Box.hh"
#include "G4Tubs.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4AutoDelete.hh"
#include "G4GeometryManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4SDManager.hh"

G4ThreadLocal
G4GlobalMagFieldMessenger* DetectorConstruction::m_magneticFieldMessenger = 0;

DetectorConstruction::DetectorConstruction() : G4VUserDetectorConstruction()
{
}

DetectorConstruction::~DetectorConstruction()
{
}

// Here we define the actual experiment that we want to perform
G4VPhysicalVolume* DetectorConstruction::Construct()
{
  // Materials
  // http://geant4-userdoc.web.cern.ch/geant4-userdoc/UsersGuides/ForApplicationDeveloper/html/Appendix/materialNames.html
  G4NistManager* nistManager = G4NistManager::Instance();
  G4Material* air = nistManager->FindOrBuildMaterial( "G4_AIR" );
  G4Material* silicon = nistManager->FindOrBuildMaterial( "G4_Si" );

  // Sizes of the principal geometrical components (solids)
  G4double tracker1Length = 400.0*cm;
  G4double tracker1InnerRadius = 399.8*cm;
  G4double tracker1OuterRadius = 400.0*cm;
  G4double tracker2Length = 800.0*cm;
  G4double tracker2InnerRadius = 799.8*cm;
  G4double tracker2OuterRadius = 800.0*cm;
  G4double worldLength = 1000.0*cm;

  // Definitions of Solids, Logical Volumes, Physical Volumes

  // WORLD: Solid (cube)
  G4GeometryManager::GetInstance()->SetWorldMaximumExtent( worldLength );
  G4Box* worldS = new G4Box(
                 "World",         // its name
                 worldLength,
                 worldLength,
                 worldLength );   // its size (in half-lengths)

  // WORLD: Logical volume (how to treat it)
  G4LogicalVolume* worldLV = new G4LogicalVolume(
                 worldS,          // its solid
                 air,             // its material
                 "World" );       // its name

  // WORLD: Physical volume (where is it)
  // Must place the World Physical volume unrotated at (0,0,0).
  G4VPhysicalVolume* worldPV = new G4PVPlacement(
                 0,               // no rotation
                 G4ThreeVector(0.0, 0.0, 0.0), // in the centre
                 worldLV,         // its logical volume
                 "World",         // its name
                 0,               // its mother volume
                 false,           // no boolean operations
                 0,               // copy number
                 true );          // checking overlaps

  // TRACKER1: Solid (tube)
  G4Tubs* tracker1S = new G4Tubs(
                 "Tracker1",        // its name
                 tracker1InnerRadius, // Hollow cylinder
                 tracker1OuterRadius,
                 tracker1Length,    // how far along the beam axis it extends
                 0.0*deg,           // starting angle
                 360.0*deg );       // ending angle (i.e. it's a full circle)

  // TRACKER1: Logical volume (how to treat it)
  G4LogicalVolume* tracker1LV = new G4LogicalVolume(
                 tracker1S,         // its solid
                 silicon,           // its material
                 "Tracker1",        // its name
                 0, 0, 0 );         // Modifiers we don't use

  // TRACKER1: Physical volume (where is it)
  G4VPhysicalVolume* tracker1PV = new G4PVPlacement(
                 0,                 // no rotation
                 G4ThreeVector(0.0, 0.0, 0.0), // in the centre
                 tracker1LV,        // its logical volume
                 "Tracker1",        // its name
                 worldLV,           // its mother volume
                 false,             // no boolean operations
                 0,                 // copy number
                 true );            // checking overlaps

  // TRACKER1: Quit if there's an overlap
  if ( tracker1PV->CheckOverlaps() ) std::cerr << "WARNING: your simulated objects overlap" << std::endl;

  // TRACKER2: Solid (tube)
  G4Tubs* tracker2S = new G4Tubs(
                 "Tracker2",        // its name
                 tracker2InnerRadius, // Hollow cylinder
                 tracker2OuterRadius,
                 tracker2Length,    // how far along the beam axis it extends
                 0.0*deg,           // starting angle
                 360.0*deg );       // ending angle (i.e. it's a full circle)

  // TRACKER2: Logical volume (how to treat it)
  G4LogicalVolume* tracker2LV = new G4LogicalVolume(
                 tracker2S,         // its solid
                 silicon,           // its material
                 "Tracker2",        // its name
                 0, 0, 0 );         // Modifiers we don't use

  // TRACKER2: Physical volume (where is it)
  G4VPhysicalVolume* tracker2PV = new G4PVPlacement(
                 0,                 // no rotation
                 G4ThreeVector(0.0, 0.0, 0.0), // in the centre
                 tracker2LV,        // its logical volume
                 "Tracker2",        // its name
                 worldLV,           // its mother volume
                 false,             // no boolean operations
                 0,                 // copy number
                 true );            // checking overlaps

  // TRACKER2: Quit if there's an overlap
  if ( tracker2PV->CheckOverlaps() ) std::cerr << "WARNING: your simulated objects overlap" << std::endl;

  // Always return the physical world
  return worldPV;
}

// Set up the magnetic field
void DetectorConstruction::ConstructSDandField()
{
  // Create global magnetic field messenger.
  // Uniform magnetic field is then created automatically if
  // the field value is not zero.
  G4ThreeVector fieldValue = G4ThreeVector();
  m_magneticFieldMessenger = new G4GlobalMagFieldMessenger( fieldValue );

  // Register the field messenger for deleting
  G4AutoDelete::Register( m_magneticFieldMessenger );

  // The silicon layers are for tracking
  auto tracker1 = new PositionFinder( "Tracker1", 1 );
  G4SDManager::GetSDMpointer()->AddNewDetector( tracker1 );
  this->SetSensitiveDetector( "Tracker1", tracker1 );
  auto tracker2 = new PositionFinder( "Tracker2", 2 );
  G4SDManager::GetSDMpointer()->AddNewDetector( tracker2 );
  this->SetSensitiveDetector( "Tracker2", tracker2 );
}
