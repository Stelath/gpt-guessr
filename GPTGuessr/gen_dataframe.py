import os
import argparse
import pandas as pd
from geopy.geocoders import Nominatim

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="The latitude and longitude points of the dataset (csv file)", required=True, type=str)
    parser.add_argument("--output", help="The output folder where the dataframe will be stored, (defaults to: data/)", default='data/', type=str)
    return parser.parse_args()

geolocator = Nominatim(user_agent="geoapiExercises")

def city_state_country(row):
    coord = f"{row['lat']}, {row['lon']}"
    location = geolocator.reverse(coord, exactly_one=True)
    address = location.raw['address']
    city = address.get('city', '')
    state = address.get('state', '')
    country = address.get('country', '')
    row['city'] = city
    row['state'] = state
    row['country'] = country
    return row

def main():
    args = get_args()
    df = pd.read_csv(args.dataset, header=None, names=['lat', 'lon'])
    df = df.apply(city_state_country, axis=1)
    df.to_pickle(os.path.join(args.out, 'dataset.df'), index=False)

if __name__ == '__main__':
    main()