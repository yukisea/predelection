import pandas as pd

def print_data_metadata(data, name):
    """Print shape and columns of a given dataset
    
    Arguments:
        data {Panda DataFrame} -- dataset
        name {name of dataset} -- name of dataset
    """

    print(name, "table: there are ", data.shape[0], "results with ", data.shape[1], "properties.")
    print(data.columns)
    print()

def load_data(datadir, verbose=True):
    """Load data from a given directory
    
    Arguments:
        datadir {string} -- directory that saves data
    
    Keyword Arguments:
        verbose {bool} -- print data metadata or not (default: {True})
    
    Returns:
        a tuple of datasets in Panda DataFrame
    """

    results = pd.read_csv(datadir + 'ResultsPipe.txt', sep = '|',low_memory=False)
    limited_results = pd.read_csv(datadir + 'LimitedResultsPipe.txt', sep = '|')
    limited_results2 = pd.read_csv(datadir + 'LimitedResults2Pipe.txt', sep = '|')
    demo = pd.read_csv(datadir + 'PatientDemographicsPipe.txt',sep='|')
    encounters = pd.read_csv(datadir + 'EncountersPipe.txt',sep='|',low_memory=False)
    ptype = pd.read_csv(datadir + 'PatientTypeOnsetPipe.txt',sep='|')
    dkaset = pd.read_csv(datadir + 'DKAPipe.txt',sep='|')
    estab = pd.read_csv(datadir + 'ClinicEstablishedPipe.txt',sep='|')

    if verbose:
        print_data_metadata(results, 'Results')
        print_data_metadata(limited_results, 'Limited Results')
        print_data_metadata(limited_results2, 'Limited Results 2')
        print_data_metadata(demo, 'Demography')
        print_data_metadata(encounters, 'Encounters')
        print_data_metadata(ptype, 'OnsetType')
        print_data_metadata(dkaset, 'DKA')
        print_data_metadata(estab, 'Clinic established patient')

    return results, limited_results, limited_results2, demo, encounters, ptype, dkaset, estab

def keep_patients_after(ptype, year=2011):
    """Filter OnsetType table to only keep patients diagnosed after a given year
    
    Arguments:
        ptype {Panda DataFrame} -- OnsetType table
    
    Keyword Arguments:
        year {int} -- year of diagnosed (default: {2011})
    
    Returns:
        PandaDataFrame -- Filtered onset table
    """

    ptype.TypeOnsetDTS = pd.to_datetime(ptype.TypeOnsetDTS)
    ptype_after_year = ptype[ptype.TypeOnsetDTS.dt.year >= year]
    print("shape of patients diagnosed after", year, ":" , ptype_after_year.shape)

    return ptype_after_year
