import numpy as np

def gen_points(heatmap, v):
    points = np.empty((heatmap.shape[0] * heatmap.shape[1], 2))
    loc = 0
    print("Predicting")
    rows = heatmap.shape[0]
    for lon in range(rows):
        print(f"{lon}/{rows} ({lon/rows*100:.2f}%)", end="\r")
        for lat in range(heatmap.shape[1]):
            if np.random.uniform() < heatmap[lon][lat]:
                points[loc] = [lat, lon]
                loc += 1
    points = points[:loc]
    np.save(f'points_{v}.npy', points)