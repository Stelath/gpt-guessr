import requests
from tqdm import tqdm
import os
import json
from random import randint
import argparse
from csv import writer
# Consider using https://osmnx.readthedocs.io/en/stable/osmnx.html#osmnx.utils_geo.sample_points for street gps coords
# Stack Overflow on how to get the coordinates: https://stackoverflow.com/questions/68367074/how-to-generate-random-lat-long-points-within-geographical-boundaries

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", help="The latitude and longitude points to download images from", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the images will be stored, (defaults to: data/)", default='data/', type=str)
    parser.add_argument("--icount", help="The amount of images to pull (defaults to 25,000)", default=25000, type=int)
    parser.add_argument("--key", help="Your Google Street View API Key", type=str, required=True)
    return parser.parse_args()

args = get_args()
url = 'https://maps.googleapis.com/maps/api/streetview'

def main():
    # Open and create all the necessary files & folders
    os.makedirs(args.output, exist_ok=True)
    
    points = np.load(args.points)
    
    coord_output_file = open(os.path.join(args.output, 'picture_coords.csv'), 'w', newline='')
    csv_writer = writer(coord_output_file)
    
    for i in tqdm(range(points.shape[0])):
        # Set the parameters for the API call to Google Street View
        for j in range(3):
            params = {
                'key': args.key,
                'size': '640x640',
                'location': str(points[i][0]) + ',' + str(points[i][0]),
                'heading': str(j * 90),
                'pitch': '20',
                'fov': '90'
                }
            
            response = requests.get(url, params)
            filename = f'street_view_{str(i).zfill(6)}_{j}.jpg'
            # Save the image to the output folder
            with open(os.path.join(args.output, filename), "wb") as file:
                file.write(response.content)
            
            # Save the coordinates to the output file
            csv_writer.writerow([filename, addressLoc[1], addressLoc[0]])

    coord_output_file.close()
    
    for i in range(len(cities_count)):
        city_count = cities_count[i]
        city_name = os.listdir(args.cities)[i]
        print(f'{city_count} images pulled from {city_name}')

if __name__ == '__main__':
    main()