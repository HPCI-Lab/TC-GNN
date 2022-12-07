

# Finds storm ids for storms belonging to some basin(in byte format, e.g. b'SI')
def extract_storms(cyclones_data, basin):
    storms = []
    tmp = cyclones_data.basin.values
    for s in range(cyclones_data.storm.size):
        for t in range(cyclones_data.date_time.size):
            if tmp[s][t] == basin:
                if s not in storms:
                    storms.append(s)
    print(f"Found {len(storms)} storms crossing at least once the basin {basin}")
    return storms
