import numpy as np
import xgboost as xgb
import sys
from .segmentation_model import pred_proc


img_shape = (512, 512)

def FindAllClusters( intensity, edge ):
    precision = 1e9

    pixels = np.vstack([[[i, j, 0] for i in range(intensity.shape[0])] for j in range(intensity.shape[1])])
    intensity[intensity < 0.45] = 0
    pixels[:, 2] = (intensity[pixels[:, 0], pixels[:, 1]] * precision).astype('int32')
    pixels = pixels[pixels[:, 2] > 0.50 * precision]
    pixels = pixels[pixels[:, 2].argsort()[::-1]]
    
    cluster = 0 * intensity
    features = []
    cluster_size_list = []

    k = 0
    for ip, p in enumerate(pixels):
        if cluster[p[0], p[1]] > 0: continue
        feature = []
        k = k + 1
        
        current = np.array([p])
        cluster_size = 0
        max_next_size = 0

        next_size_duplicates_list = []
        next_size_duplicates_ratio_list = []
        next_size_filled_list = []
        next_size_filled_ratio_list = []
        next_size_list = []
        next_size_diff_list = []
        next_size_ratio_list = []

        intensity_list = []
        edge_list = []
        valid_intensity_list = []
        valid_edge_list = []
        invalid_intensity_list = []
        invalid_edge_list = []
        while current.shape[0] > 0:
            cluster[current[:, 0], current[:, 1]] = k
            cluster_size += current.shape[0]
            next = np.concatenate([current + d for d in [(1,0,0), (0,1,0), (-1,0,0), (0,-1,0)]])
            next = next[np.lexsort([next[:, 0], next[:, 1], next[:, 2]], axis=0)]
            duplicates = np.all(next[:-1, :2] == next[1:, :2], axis=1)
            duplicates = np.concatenate([duplicates, [False]])

            size = next.shape[0]
            next = next[-duplicates]
            next_size_duplicates_list.append(size - next.shape[0])
            next_size_duplicates_ratio_list.append((size - next.shape[0]) / (size + 0.1))
            next = next[(0 <= next[:, 0]) & (next[:, 0] < intensity.shape[0]) & (0 <= next[:, 1]) & (next[:, 1] < intensity.shape[1])]
            size = next.shape[0]
            next = next[cluster[next[:, 0], next[:, 1]] == 0]
            next_size_filled_list.append(size - next.shape[0])
            next_size_filled_ratio_list.append((size - next.shape[0]) / (size + 0.1))

            nextintencity = intensity[next[:, 0], next[:, 1]] * precision
            intensity_list.append(nextintencity)
            nextedge = edge[next[:, 0], next[:, 1]]
            edge_list.append(nextedge)

            idx = (0.0 < nextintencity) & (nextintencity < next[:, 2] * 1.02)

            valid_intensity_list.append(nextintencity[idx])
            invalid_intensity_list.append(nextintencity[-idx])
            valid_edge_list.append(nextedge[idx])
            invalid_edge_list.append(nextedge[-idx])

            idx_keepmax = next[:, 2] < nextintencity
            nextintencity[idx_keepmax] = next[idx_keepmax, 2]
            next[:, 2] = nextintencity
            size = next.shape[0]
            next = next[idx]
            next_size_diff_list.append(size - next.shape[0])
            next_size_ratio_list.append((size - next.shape[0]) / (size + 0.1))
            next_size_list.append(next.shape[0])

            max_next_size = max(max_next_size, next.shape[0])

            current = next

        if cluster_size < 100:
            cluster[cluster == k] = 0
            k = k - 1
        else:
            feature.append(k)
            feature.append(p[2])
            feature.append(len(next_size_list))
            feature.append(max_next_size)
            feature.append(cluster_size)
            for size_list in [next_size_duplicates_list,
                                next_size_duplicates_ratio_list,
                                next_size_filled_list,
                                next_size_filled_ratio_list,
                                next_size_list,
                                next_size_diff_list,
                                next_size_ratio_list]:
                    n = len(size_list)
                    for i in range(4):
                        li = size_list[n * i // 4: n * (i + 1) // 4]
                        feature.append(sum(li) / (len(li) + 0.1))
                    li = list(sorted(size_list))
                    for i in range(1, 5):
                        feature.append(li[n * i // 5])
                    feature.append(sum(li) / n)
            feature.append(np.std(next_size_list))
            for value_list in [intensity_list,
                                edge_list,
                                valid_intensity_list,
                                valid_edge_list,
                                invalid_intensity_list,
                                invalid_edge_list]:
                    np_value_list = np.concatenate(value_list)
                    n = np_value_list.shape[0]
                    for i in range(4):
                        li = np_value_list[n * i // 4: n * (i + 1) // 4]
                        feature.append(li.sum() / (li.shape[0] + 0.1))
                    np_value_list.sort()
                    for i in range(1, 5):
                        feature.append(np_value_list[n * i // 5] if n > 0 else -1)
                    feature.append(np_value_list.sum() / n)
            features.append(feature)
    
    features = np.array(features)
    if features.shape[0] > 0:
        stats = np.concatenate([np.percentile(features, q, axis=0) for q in [25, 50, 75]] + [features.mean(axis=0)])   
        stats = np.tile(stats, (features.shape[0], 1))
        features = np.concatenate([features, stats], axis=1)
    return k, cluster, features


def train_building_proc(arg):
    pred_proc(arg)

    path = arg[0]
    filenames = arg[1]
    ith = arg[2]
    gtf_path = arg[3]


    feature_list = []
    label_list = []
    for fn in filenames:
        area = np.load('area/{}.npz'.format(fn))
        edge = np.load('edge/{}.npz'.format(fn))
        cluster_count, cluster, features = FindAllClusters(area, edge)
        if cluster_count == 0: continue

        truth_polygons = get_polygons(fn)
        truth = []
        for polygon in truth_polygons:
            img = np.zeros(img_shape, dtype=int)
            rr, cc = draw_polygon(polygon)
            img[rr, cc] = 1
            truth.append((img, img.sum()))
        labels = []
        for k in range(1, 1 + cluster_count):
            p = (cluster == k)
            ps = p.sum()
            maxv = 0
            for t, ts in truth:
                v = (p & t).sum()
                v = v / (ps + ts - v)
                maxv = max(maxv, v)
            labels.append(maxv)
            print(fn, k, maxv)
        feature_list.append(features)
        label_list.append(np.array(labels))
    feature_list = np.concatenate(feature_list)
    label_list = np.concatenate(label_list)
    print(feature_list.shape, label_list.shape)
    np.concatenate([feature_list, label_list[:, np.newaxis]], axis=1).dump('train_building_{}.npz'.format(ith))

def train_building(path, gtf_path, create_data=True, nthread=default_nthread):
    if create_data:
        all_files = GetFileList(path)
        n = len(all_files)
        filelist_group = [all_files[n*i//nthread:n*(i+1)//nthread] for i in range(nthread)]

        with multiprocessing.Pool(nthread) as pool:
            pool.map(train_building_proc, [(path, filelist[1], filelist[0], gtf_path) for filelist in enumerate(filelist_group)]) 

        np.concatenate([np.load('train_building_{}.npz'.format(ith)) for ith in range(nthread)]).dump('train_building.npz')
    
    tr = np.load('train_building.npz')
    val_idx = np.array(range(tr.shape[0])) % 100 == 0
    label = tr[:, -1]
    label = 1 / (1 + np.exp(4.0 - 8.0 * label))
    val_label = label[val_idx]
    label = label[-val_idx]
    val = tr[val_idx, :-1]
    tr = tr[-val_idx, :-1]
    dtrain = xgb.DMatrix(tr, label=label)
    dval = xgb.DMatrix(val, label=val_label)
    params = {
        'objective': 'reg:linear',
        'eval_metric': 'rmse',
        'eta': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.6,
        'max_depth': 6,
        'seed': 0,
        'nthread': default_nthread,
        'silent': 1,
    }
    model = xgb.train(params, dtrain, evals=[(dtrain, 'train'), (dval, 'val')], num_boost_round=1000, verbose_eval=True)
    model.save_model('model/xgb.model')

def pred_proc(arg):
    pred_proc(arg)

    path = arg[0]
    filenames = arg[1]

    xgbmodel = xgb.Booster({'nthread': 1})
    xgbmodel.load_model(arg[2])

    minconfidence = 0.22
    for fn in filenames:
        dt2 = datetime.datetime.now()
        area = np.load('area/{}.npz'.format(fn))
        edge = np.load('edge/{}.npz'.format(fn))
        cluster_count, cluster_raw, features = FindAllClusters(area, edge)
        cluster = np.zeros(img_shape, dtype=int)
        if cluster_count > 0:
            confidence = pd.Series(xgbmodel.predict(xgb.DMatrix(features)))
            rank = confidence.rank(method='first')
            maxrank = rank.max()
            minrank = rank[confidence >= minconfidence].min()
            rank = rank - maxrank + 255
            rank[confidence < minconfidence] = 0
            rank[rank < 0] = 0            
            for i in range(cluster_count):
                cluster[cluster_raw == (i + 1)] = rank[i]

        dt3 = datetime.datetime.now()
        print(fn, cluster_count, int((dt3-dt2).total_seconds()))

def predict(path, nthread=default_nthread):
    nsplit = 20
    all_files = GetFileList(path)
    n = len(all_files)
    filelist_group = [all_files[n*i//nsplit:n*(i+1)//nsplit] for i in range(nsplit)]

    with multiprocessing.Pool(nthread) as pool:
        pool.map(pred_proc, [(path, filelist, "model/xgb.model") for filelist in filelist_group])  

if __name__ == "__main__":
    command = sys.argv[1]
    if command == 'train':
        train()
    if command == 'test':
        predict(sys.argv[2])