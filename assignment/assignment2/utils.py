"""
utils.py
This file contains helper functions for Assignment 2.
Author: Sebastian D. Goodfellow
"""

# 3rd party imports
import os
import time
import requests
import numpy as np


def download_noaa_file(filename):
    """This function will download from the NOAA database text file corresponding to the filename and url
    input by the user and save to '/data'"""
    """Get HTML object from url"""
    # Set NCAT url
    url = 'https://www.glerl.noaa.gov/emf/glcfs/gridded_fields/NCAST/'

    # Create data directory
    os.makedirs('./data/', exist_ok=True)

    print('Downloading NOAA file {}'.format(filename))
    while True:
        try:
            # Send file request to server and download
            start_time = time.time()
            response = requests.get(url + filename, verify=False, timeout=120)
            print('Download complete: {} minutes'.format(np.round((time.time() - start_time) / 60., 4)))

            # Save text file
            with open('data/{}'.format(filename), 'w') as file:
                file.write(response.text)
            print('NOAA file {} was saved\n'.format(filename))
            break

        except Exception:
            print('Connection Error, retrying...')
            time.sleep(1)
            pass


def haversine(lat1, lon1, lat2, lon2):
    """Defines a basic Haversine distance formula."""
    MILES = 3959
    lat1, lon1, lat2, lon2 = map(np.deg2rad, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    total_miles = MILES * c
    return total_miles * 1.60934
