#!/bin/bash

python get_kepler_data.py $1 -c 'short' -o Data/$1-s.sh --cmdtype 'curl'
python get_kepler_data.py $1 -c 'long' -o Data/$1-l.sh --cmdtype 'curl'
cd Data
./$1-s.sh
./$1-l.sh
find . -type f -name "*.fits" -size -1k -delete

cd ../
python detrend_and_estimate_ttvs.py --mission 'Kepler' --target $1 --project_dir './' --data_dir 'Data/' --catalog 'cumulative_2025.05.30_14.09.29.csv' --run_id '06_10_25'
python analyze_autocorrelated_noise.py --mission 'Kepler' --target $1 --project_dir './' --data_dir 'Data/' --catalog 'cumulative_2025.05.30_14.09.29.csv' --run_id '06_10_25'
python fit_transit_shape_simultaneous_nested.py --mission 'Kepler' --target $1 --project_dir './' --data_dir 'Data/' --catalog 'cumulative_2025.05.30_14.09.29.csv' --run_id '06_10_25'
python importance.py $1 0
python importance.py $1 1
python importance.py $1 2
python importance.py $1 3
python importance.py $1 4
