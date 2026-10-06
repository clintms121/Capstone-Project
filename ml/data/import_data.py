import h5py
import pandas as pd

# load required data

with h5py.File('H-H1_LOSC_4_V1-1126259446-32.hdf5', 'r') as f:
    strain = f['strain']['Strain'][:]
    ts = f['meta']['GPSstart'][()] + (range(len(strain)) * f['meta']['Dt'][()])


df = pd.DataFrame({'GPS_Time': ts, 'Strain': strain})
df.to_csv('ligo_raw_data.csv', index=False)