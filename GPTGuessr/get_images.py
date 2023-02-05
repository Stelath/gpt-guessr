import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
import numpy as np
from csv import writer
import geopandas as gpd
import pandas as pd
# Consider using https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_geo.sample_points for street gps coords
# Stack Overflow on how to get the coordinates: https://stackoverflow.com/questions/68367074/how-to-generate-random-lat-long-points-within-geographical-boundaries

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", help="The latitude and longitude points to download images from", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the images will be stored, (defaults to: data/)", default='data/', type=str)
    parser.add_argument("--start-count", help="The starting count for the images (defaults to 0)", default=0, type=int)
    parser.add_argument("--icount", help="The amount of images to pull (defaults to 25,000)", default=25000, type=int)
    parser.add_argument("--key", help="Your Google Street View API Key", type=str, required=True)
    return parser.parse_args()

args = get_args()
url = 'https://maps.googleapis.com/maps/api/streetview'
metadata_url = 'https://maps.googleapis.com/maps/api/streetview/metadata'

def main():
    print('Enter your Google Street View API Key: ')
    key = input()
    # Open and create all the necessary files & folders
    os.makedirs(args.output, exist_ok=True)
    
    points = pd.read_pickle(args.points)
    coords = points.geometry
    
    print(f"Downloading off {len(coords)} points")
    
    coord_output_file = open(os.path.join(args.output, 'picture_coords.csv'), 'w', newline='')
    csv_writer = writer(coord_output_file)
    
    for i in tqdm(range(args.start_count, args.icount + args.start_count)):
        # Set the parameters for the API call to Google Street View
        lat, lon = coords[i].y + np.random.randint(10) / 1000, coords[i].x + np.random.randint(10) / 1000
        for j in range(3):
            params = {
                'key': key,
                'size': '640x640',
                'location': f'{lat:.3f},{lon:.3f}',
                'heading': str((np.random.randint(0, 3) * 90) + np.random.randint(-15, 15)),
                'pitch': '20',
                'fov': '90',
                'radius': 10000000
            }
            
            response = requests.get(url, params)
            filename = f'street_view_{str(i).zfill(6)}_{j}.jpg'
            # Save the image to the output folder
            with open(os.path.join(args.output, filename), "wb") as file:
                file.write(response.content)
            
            response = requests.get(url, params)
            lat, lon = response.json()['location']['lat'], response.json()['location']['lng']
            
            # Save the coordinates to the output file
            csv_writer.writerow([filename, lat, lon])

    coord_output_file.close()

if __name__ == '__main__':
    main()