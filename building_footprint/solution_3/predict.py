def pred_proc(arg):
    path = arg[0]
    filenames = arg[1]

    model_file = '2_96'
    model = create_model()
    model.load_weights('model/{}.h5'.format(model_file))

    for fn in filenames:    
        dt1 = datetime.datetime.now()
        area, edge = model.predict(CreateData(path, fn)[np.newaxis, :])
        area = area[0, :, :, 0]
        edge = edge[0, :, :, 0]
        area.dump('area/{}.npz'.format(fn))
        edge.dump('edge/{}.npz'.format(fn))
        dt2 = datetime.datetime.now()
        print(fn, int((dt2-dt1).total_seconds()))


def predict(path, nthread=4):
    all_files = GetFileList(path)
    n = len(all_files)
    filelist_group = [all_files[n*i//nthread:n*(i+1)//nthread] for i in range(nthread)]

    if True:
        with multiprocessing.Pool(nthread) as pool:
            pool.map(pred_proc, [(path, filelist) for filelist in filelist_group])
    else:
        pred_proc((path, all_files))