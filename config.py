import math

class DefaultConfigs(object):
    seed = 666
    # SGD
    weight_decay = 5e-4
    momentum = 0.9
    # learning rate
    init_lr = 0.01
    # training parameters
    train_epoch = 100
    test_epoch = 1
    BATCH_SIZE_TRAIN = 64
    norm_flag = True
    gpus = '0'
    # source data information
    data = 'PaviaU'  # PaviaU-9 / Indian-16  / Houston2018-21  / Houston2013-16
    num_classes = 9
    patch_size = 11
    pca_components = 30
    test_ratio = 0.9
    # model
    model_type = 'VideoMamba'     # LSFAT or LSFAT_WoCToken or LSFAT_Dilate or LSFAT_Shunted or transformer or 3DCNN
    depth = 1
    embed_dim = 32
    dim_inner = 2*embed_dim
    dt_rank = math.ceil(embed_dim/16)
    d_state = 16
    group_type = 'Cube'  # Linear  Patch  Cube
    scan_type = 'Parallel spectral-spatial'  #Spectral-priority  #Spatial-priority  #Cross spectral-spatial  Parallel spectral-spatial #spatial-spectral   spectral-spatial   spectral  spatial
    k_group = 4
    pos = False
    cls = False
    # 3DConv parameters
    conv3D_channel = 32
    conv3D_kernel = (3, 5, 5)
    dim_patch = patch_size - conv3D_kernel[1] + 1  # 8
    dim_linear = pca_components - conv3D_kernel[0] + 1  # 28
    # paths information
    checkpoint_path = './' + "checkpoint/" + data + '/' + model_type + '/' + 'TrainEpoch' + str(train_epoch) + '_TestEpoch' + str(test_epoch) + '_Batch' + str(BATCH_SIZE_TRAIN) \
                      + '/Pos(' + str(pos) + ')' + '_Cls(' + str(cls)+ ')_' + scan_type + '/PatchSize' + str(patch_size) + '_TestRatio' + str(test_ratio) \
                      + '/' + group_type + str(k_group) + '_depth' + str(depth) + '_embed' + str(embed_dim) + '_dtrank' + str(dt_rank) + '_dstate' + str(d_state) + '_3Dconv' + str(conv3D_channel) + '&' + str(conv3D_kernel) + '/'
    logs = checkpoint_path

config = DefaultConfigs()
