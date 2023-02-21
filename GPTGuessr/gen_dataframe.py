import os
import argparse
import pandas as pd
from geopy.geocoders import Nominatim
from tqdm.auto import tqdm

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The latitude and longitude points of the dataset (csv file)", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the dataframe will be stored, (defaults to: data/)", default='data/', type=str)
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
    df = pd.read_csv(args.dataset, header=None, names=['lat', 'lon'])
    
    geolocator = Nominatim(user_agent="geoapiExercises")
    pbar = tqdm(total=len(df))
    
    df = df.apply(city_state_country, args=(geolocator, pbar), axis=1)
    
    df.to_pickle(os.path.join(args.out, 'dataset.df'), index=False)

if __name__ == '__main__':
    main()