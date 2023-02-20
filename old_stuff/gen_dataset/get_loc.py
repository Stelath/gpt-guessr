import requests
import numpy as np
import pandas as pd
import os
from shapely.geometry import Point

def get_data(minv, maxv, coords, key, world):
    url = 'https://maps.googleapis.com/maps/api/streetview/metadata'
    

    file_names = os.listdir('data/')[minv:maxv]
    save_file_names = []
    countries = []
    img_coords = []
    i = 0
    while i < maxv - minv:
        con = True
        if not file_names[i][-5].isnumeric() or int(file_names[i][-5]) != 0:
            i += 1
            con = False
        
        if con:
            file_name = file_names[i][:-6]
            idx = int(file_name[-6:])
            
            lat, lon = coords[idx].y, coords[idx].x

            params = {
                'key': key,
                'size': '640x640',
                'location': f'{lat:.3f},{lon:.3f}',
                'heading': 90,
                'pitch': '20',
                'fov': '90',
                'radius': 1000100
                }

            response = requests.get(url, params)
            if response.json()['status'] == "OK":
                lat, lon = response.json()['location']['lat'], response.json()['location']['lng']
            else:
                lat, lon = 0, 0
            
            did = False
            for j, country in enumerate(world.geometry):
                buffer = country.buffer(0.4)
                if Point(lon, lat).within(buffer):
                    save_file_names.append(file_name)
                    countries.append(j)
                    img_coords.append([lat, lon])
            
            i += 3
        
        print(f"Image {i} Percent: {(i) / (maxv - minv) * 100:.2f}%", end='\r')
        
    pd.DataFrame({'file_name': save_file_names, 'country': countries, 'coords': img_coords}).to_pickle(f'df_{minv}_{maxv}.df')
    # Save the coordinates to the output file
