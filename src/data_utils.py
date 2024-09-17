import pandas as pd
import numpy as np

import datetime


# import project utils
import sys
sys.path.append('../src')


class Config:
    RANDOM_STATE = 42

    # DATA & SAMPLE FILES
    CURR_RAW_FILE = '../data/Police_Department_Incident_Reports__2018_to_Present_20240910.csv'
    CURR_CLEAN_FILE = '../data/incidents_clean.csv'
    

def generate_clean_csv(infile, outfile, debug=False):
    """
    :param infile: File to read with raw data
    :param outfile: CSV file to write once the data has been cleaned and reformatted
    """

    # Timestamp the report
    print(f'Start Time: {datetime.datetime.now().strftime("%D %H:%M:%S")}\n')

    # Read the file
    print('Reading file: {} ... '.format(infile), end='')
    raw_df = pd.read_csv(infile)
    print('Done: {:,d} rows, {:,d} columns'.format(raw_df.shape[0], raw_df.shape[1]))
    if debug:
        raw_df.columns    

    # Convert column names with spaces and upper-case to underscore and lower-case format
    print('... Formatting column names ... ', end='')
    raw_df.columns = raw_df.columns.str.replace(' ', '_').str.lower()
    if debug:
        raw_df.columns    
    print('Done')

    # Create two new datetime and date columns with clean dates
    # raw_df['datetime'] = pd.to_datetime(df.incident_datetime, format='%Y/%m/%d %I:%M:%S %p')
    print('... Creating timeseries columns: datetime ... ')
    raw_df['datetime'] = pd.to_datetime(raw_df.incident_datetime, format='mixed')
    # raw_df['date'] = pd.to_datetime(raw_df.incident_date, format='mixed')
    print('...... Number of rows where the datetime conversion failed: {:,d}'.format(raw_df.datetime.isnull().sum()))
    # print('...... Number of rows where the date conversion failed: {:,d}'.format(raw_df.date.isnull().sum()))
    print('...... Timespan: {} - {}'.format(raw_df.datetime.min(), raw_df.datetime.max()))    
    print('... Done')

    # set datetime as index to create the timeseries
    print('... Setting index to datetime ... ', end='')
    raw_df = raw_df.set_index('datetime')
    print('Done')
    
    print(f'\nCurrent Time: {datetime.datetime.now().strftime("%D %H:%M:%S")}\n')

    if debug:
        raw_df.info()
    
    # Write the file
    print('... Writing file: {} {} ... '.format(outfile, raw_df.shape), end='')
    raw_df.to_csv(outfile)
    print('Done')

    print('Done')
    print(f'\nEnd Time: {datetime.datetime.now().strftime("%D %H:%M:%S")}\n')


def generate_sample_files(infile, random_state=42):
    """
    Generates an output file containg samples from the input based on frac and random_state
    
    :param infile: CSV file to read data from
    :param outfile_prefix: CSV file to write once the data has been cleaned and reformatted
    :param frac: Fraction of the file to include in sampling
    :param random_state: Control the randomization
    :return: Returns DataFrame with index set to the datetime column
    """

    # Timestamp the report
    print(f'Start Time: {datetime.datetime.now().strftime("%D %H:%M:%S")}\n')
    
    # Read the file
    print('Reading file: {} ... '.format(infile), end='')
    in_df = pd.read_csv(infile)
    print('Done: {:,d} rows, {:,d} columns'.format(in_df.shape[0], in_df.shape[1]))

    outfile_prefix = infile.replace('.csv', '')
    for pct in [10, 25, 50, 75]:
        outfile = f'{outfile_prefix}_{pct}_pct.csv'
    
        out_df = in_df.sample(frac=pct/100, random_state=random_state)
        
        # Write the file
        print('... Writing {}% sample file: {} {} ... '.format(pct, outfile, out_df.shape), end='')
        out_df.to_csv(outfile)
        print('Done')

    # Timestamp the report
    print(f'End Time: {datetime.datetime.now().strftime("%D %H:%M:%S")}\n')


def generate_sample_file(infile, outfile_prefix, frac=0.1, random_state=42):
    """
    Generates an output file containg samples from the input based on frac and random_state
    
    :param infile: CSV file to read data from
    :param outfile_prefix: CSV file to write once the data has been cleaned and reformatted
    :param frac: Fraction of the file to include in sampling
    :param random_state: Control the randomization
    :return: Returns DataFrame with index set to the datetime column
    """
    # Read the file
    print('Reading file: {} ... '.format(infile), end='')
    in_df = pd.read_csv(infile)
    print('Done: {:,d} rows, {:,d} columns'.format(in_df.shape[0], in_df.shape[1]))

    out_df = in_df.sample(frac=frac, random_state=random_state)
    
    # Write the file
    print('Writing file: {} {} ... '.format(outfile, out_df.shape), end='')
    out_df.to_csv(outfile)
    print('Done')



# TODO: This needs to be cleaned out
current_raw_file = '../data/Police_Department_Incident_Reports__2018_to_Present_20240910.csv'
current_clean_file = '../data/incidents_clean.csv'

def select_sample_csv_file(pct=None):
    """
    Given a sample size percentage, selects the appropriate pre-sampled CSV file with randomized data per project_random_state

    :param pct: Sample percentage (integer), defaults to 10% if sample size is not available. Pass "100" to get original file as DF
    :returns: Returns path to selected sample file
    """
    # Which dataset to work from?
    sample_file_size_pct = 10
    if pct is not None:
        pct = 100 if pct > 100 else pct
        pct = 10 if pct < 10 else pct
        sample_file_size_pct = pct

    match(sample_file_size_pct):
        case 100:
            sample_file = current_clean_file
        case 25 | 50 | 75:
            file_prefix = current_clean_file.replace('.csv', '')
            sample_file = f'{file_prefix}_{sample_file_size_pct}_pct.csv'
        case _:
            sample_file = '../data/incidents_clean_10_pct.csv'
    
    return sample_file



def preprocess_drop_cols(df, drop_cols):
    """
    Drops the columns from DF inplace

    :param df: DF to drop columns from
    :param cols: Columns to drop
    :returns: Returns the DF to allow pipelining, but the columns are dropped inplace
    """
    if len(drop_cols) <= 0 or df is None:
        return df

    for col in drop_cols:
        try:
            df.drop(columns=col, inplace=True)
            print(f'... preprocess_drop_cols: Column {col} dropped')
        except:
            print(f'... preprocess_drop_cols: Column {col} not dropped: {repr(sys.exception())}')

    return df



def get_clean_data_from_csv(infile):
    """
    :param infile: CSV file to read data from
    :return: Returns 2 DataFrames raw_df and clean_df with index set to the datetime column
    """
    # Read the file
    print('Reading file: {} ... '.format(infile), end='')
    raw_df = pd.read_csv(infile)
    print('Done: {:,d} rows, {:,d} columns'.format(raw_df.shape[0], raw_df.shape[1]))
    
    # Converting datetime and date to timeseries ...
    print('... Converting datetime to timeseries ... ', end='')
    raw_df.datetime = pd.to_datetime(raw_df.datetime)
    # raw_df.date = pd.to_datetime(raw_df.date)
    print('Done')

    # set datetime as index to create the timeseries
    print('... Setting index to datetime ... ', end='')
    raw_df = raw_df.set_index('datetime')
    print('Done')

    # make a full copy
    clean_df = raw_df.copy()
    
    print('Done')

    return raw_df, clean_df


def preprocess_data(df, drop_cols=None):
    """
    Apply the preprocess steps identified durng EDA

    :param df: DF to run pre-processing steps - done inplace
    :returns: Returns pre-processed DF for chaining
    """

    drop_cols_unwanted = ['Unnamed: 0', 
                          'esncag_-_boundary_file', 'central_market/tenderloin_boundary_polygon_-_updated',  
                          'civic_center_harm_reduction_project_boundary','hsoc_zones_as_of_2018-06-05',
                          'invest_in_neighborhoods_(iin)_areas',
                          'report_type_code', 'report_type_description', 'filed_online',
                          'intersection', 'cnn', 'point',
                          'supervisor_district', 'supervisor_district_2012', 'current_supervisor_districts',
                         ]
    drop_cols_incident = ['incident_datetime', 'report_datetime', 
                          'incident_id', 'incident_code', 'row_id', 'incident_number', 'cad_number',
                          'incident_subcategory', 'incident_description'
                         ]
    drop_cols_pd = ['current_police_districts']
    drop_cols_neighborhoods = ['neighborhoods']

    drop_cols_all = drop_cols_unwanted + drop_cols_incident + drop_cols_pd + drop_cols_neighborhoods
    
    # Preprocessing steps
    print('Pre-processing ... ')

    print('... Dropping unwanted columns ... ')
    if (drop_cols is not None):
        preprocess_drop_cols(df, drop_cols)
        print('... Done')
        return df

    # OK, we're pro-processing the raw data in full
    raw_shape = df.shape
    
    # Drop all unwanted columns
    preprocess_drop_cols(df, drop_cols_all)
    print('... Done')

    # Remove the undesired data found in EDA
    print('... Removing resolution types: "Unfounded", "Exceptional Adult" ... ')
    df = df.query('resolution != "Unfounded" and resolution != "Exceptional Adult"')
    print('... Removing police_district types: "Out of SF" ... ')
    df = df.query('police_district != "Out of SF"')

    # Rename columns
    print('... Renaming column: "analysis_neighborhood" -> "neighborhood" ... ')
    df = df.rename(columns={'analysis_neighborhood':'neighborhood'})
    print('... Renaming columns: Dropping "incident_*" from column names ... ')
    df = df.rename(columns={'incident_date':'date',
                            'incident_time':'time',
                            'incident_year':'year',
                            'incident_day_of_week':'day_of_week',
                            'incident_category':'category'
                           })
    
    # Drop nulls
    print('... Removing rows with nulls (dropna) ... ')
    df = df.dropna()
    print('... Done')        

    reduction = raw_shape[0] - df.shape[0]
    reduction_p = (reduction / raw_shape[0]) * 100
    print(f'Done: Start: {raw_shape}, End: {df.shape} -> Rows removed: {reduction:,d} rows ({reduction_p:,.2f}%)')
    return df
