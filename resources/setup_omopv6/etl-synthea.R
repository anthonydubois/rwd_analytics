 devtools::install_github("OHDSI/ETL-Synthea")

 library(ETLSyntheaBuilder)

 # We are loading into a local PostgreSQL database called "synthetic_omop_v6".  
 # The schema to load the Synthea tables is "native".
 # The schema to load the Vocabulary and CDM tables is "cdm_synthetic_omop_v6".  
 # The username and pw are "postgres" and "lollipop".
 # The Synthea and Vocabulary CSV files are located in /tmp/synthea/output/csv and /tmp/Vocabulary_20181119, respectively.
 
 cd <- DatabaseConnector::createConnectionDetails(
  dbms     = "postgresql", 
  server   = "localhost/synthetic_omop_v6", 
  user     = "postgres", 
  password = "lollipop", 
  port     = 5432
)

ETLSyntheaBuilder::DropVocabTables(connectionDetails = cd,
                                   cdmDatabaseSchema = "cdm_synthetic_omop_v6")

ETLSyntheaBuilder::DropEventTables(connectionDetails = cd,
                                   cdmDatabaseSchema = "cdm_synthetic_omop_v6")
                                   
ETLSyntheaBuilder::DropSyntheaTables(connectionDetails = cd, 
                                     syntheaDatabaseSchema = "native")
                                     
ETLSyntheaBuilder::DropMapAndRollupTables (connectionDetails = cd, 
                                           cdmDatabaseSchema = "cdm_synthetic_omop_v6")
                                           
ETLSyntheaBuilder::CreateVocabTables(connectionDetails = cd, 
                                     vocabDatabaseSchema = "cdm_synthetic_omop_v6")
                                     
ETLSyntheaBuilder::CreateEventTables(connectionDetails = cd, 
                                     cdmDatabaseSchema = "cdm_synthetic_omop_v6")
                                     
ETLSyntheaBuilder::CreateSyntheaTables(connectionDetails = cd, 
                                       syntheaDatabaseSchema = "native")
                                       
ETLSyntheaBuilder::LoadSyntheaTables(connectionDetails = cd, 
                                     syntheaDatabaseSchema = "native", 
                                     syntheaFileLoc = "/tmp/synthea/output/csv")
                                     
ETLSyntheaBuilder::LoadVocabFromCsv(connectionDetails = cd, 
                                    vocabDatabaseSchema = "cdm_synthetic_omop_v6", 
                                    vocabFileLoc = "/tmp/Vocabulary_20181119")
                                    
ETLSyntheaBuilder::CreateVocabMapTables(connectionDetails = cd, 
                                        cdmDatabaseSchema = "cdm_synthetic_omop_v6")
                                        
ETLSyntheaBuilder::CreateVisitRollupTables(connectionDetails = cd, 
                                           cdmDatabaseSchema = "cdm_synthetic_omop_v6", 
                                           syntheaDatabaseSchema = "native")

ETLSyntheaBuilder::LoadEventTables(connectionDetails = cd, 
                                   cdmDatabaseSchema = "cdm_synthetic_omop_v6", 
                                   syntheaDatabaseSchema = "native")