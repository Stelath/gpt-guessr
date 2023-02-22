import os
import argparse
import pandas as pd
from geopy.geocoders import Nominatim
from tqdm.auto import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The dataset folder (with CSV file titled picture_coords.csv)", required=True, type=str)
    # parser.add_argument("--workers", help="The number of workers to process the dataset (defaults to 4)", default=4, type=int)
    return parser.parse_args()

def city_state_country(row, geolocator, pbar):
    coord = f"{row['lat']}, {row['lon']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    row['city'] = city
    row['state'] = state
    row['country'] = country
    pbar.update(1)
    return row

def main():
    args = get_args()
    df = pd.read_csv(os.path.join(args.dataset, 'picture_coords.csv'), header=None, names=['lat', 'lon'])
    
    geolocator = Nominatim(user_agent="geoapiExercises")
    pbar = tqdm(total=len(df))
    df = df.apply(city_state_country, args=(geolocator, pbar), axis=1)
    
    df.to_pickle(os.path.join(args.dataset, 'dataset.df'))

if __name__ == '__main__':
    main()