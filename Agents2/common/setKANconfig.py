

def setKANconfig(blockId, spline_order, grid_size):
    size = len(blockId)
    configDic = {}
    for i in range(size):
        configDic[blockId[i]] = {
            "grid_size":grid_size[i], 
            'spline_order':spline_order[i]
        }
    return configDic