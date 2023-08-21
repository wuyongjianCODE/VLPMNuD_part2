import matplotlib.pyplot as plt
import os
import skimage.io as io
def print_ALL_with_metreics_withCOMPARE():
    fid=0
    ourstxt="/home/iftwo/wyj/M/logs/LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4.txt"
    oursdir='.._.._logs_RESULTIMGS_LRPTS20211230T18420779backup_TS_of_loop4_Student_num_4'
    submits=['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',#GT
        "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_continue/val2023-03-09 20:05:52_150623/",#22
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:14:07_328440/',#35
             '/data1/wyj/M/samples/PRM/YOLOX/val_vlplm/',#33
           '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:06:29_049792/',# '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_scn_275/val2023-03-09 19:02:29_255141/',#17
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sc_412/val2023-03-09 20:27:55_874946/',#413
           # '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_sn/val2023-03-09 21:20:14_418623/',#336
           "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",#414
             '/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/fullsup/',#full44

    ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue
        allpic = [17,178,231,273,282]
        allpic = [178, 231]
            #print(dices)
    # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
           # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
        # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):
        if image_id not in allpic:
            continue
        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.05, right=1, top=1, bottom=0.05,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=-0.15)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
def print_self_training():
    fid=0
    submits = ['/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 22:00:26_382902/',  # GT
                '/data1/wyj/M/samples/PRM/YOLOX/val_sc_json/',#sc
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_30_ckpt', #'/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_many4n/val2023-03-10 11:07:59_800838/',  # 293
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_40_ckpt',  # 359
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_70_ckpt',  # 404
                "/data1/wyj/M/samples/PRM/YOLOX/YOLOX_outputs/yolox_s_base/val2023-03-09 21:51:22_686128/",  # 414
                'YOLOX_outputs/yolox_s_from008to246/val_epoch_150_ckpt',  # 414
                ]
    INCH = 20
    H = 16
    fig = plt.gcf()
    fig.set_size_inches(INCH, 2 * INCH)
    HAS_=0
    COL_NUMS=8
    for image_id in range(480):
        try:
            oriim=io.imread('datasets/COCO/val2017/%012d.jpg'%(image_id))
        except:
            continue

        allpic = [178, 231]
        allpic = [18,29,226, 191,227,195,201,204,178,456,299,282]
        allpic = [204,]
        if image_id not in allpic:
            continue
    #         #print(dices)
    # # if dices[0] > 0.1 and dices[1] > 0.1 and dices[2] > 0.05 and dices[3]>0.7 and dices[4]>0.7 :#and filename in set(
    #        # allpic):  # f_dices[1]>0.6 and f_dices[2]>0.6 and f_dices[0]>0.6 :
    #     # and filename in set(['TCGA-E2-A14V-01Z-00-DX1_crop_9.png','TCGA-AR-A1AK-01Z-00-DX1_crop_14.png']):

        fid += 1
        # if fid > 13:
        #     fid = 13
        idx = 1
        for submit in submits:
            idx += 1
            plt.subplot(H, COL_NUMS, idx + (fid%H) * COL_NUMS)
            plt.subplots_adjust(left=0.025, right=0.975, top=0.975, bottom=0.025,wspace=0.05,hspace=0.05)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            im_method=io.imread(submit+'/%012d.jpg'%(image_id))
            plt.imshow(im_method)
        plt.subplot(H, COL_NUMS, 1 + (fid%H) * COL_NUMS)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.imshow(oriim)
        # plt.title('datasets/COCO/val2017/%012d.jpg'%(image_id),y=0)
        plt.margins(0, 0)
        if fid%H==0:
            plt.savefig('TOSHOW/ALLcom{}.png'.format(fid))
    TITLE='abcdefg'
    # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.savefig('TOSHOW/ALLcom.png')
    # ALL=io.imread('TOSHOW/ALLcom.png')
    # io.imsave('TOSHOW/COMPAREALL.png',ALL[450:1800,245:1820,:])
    # plt.show()
# print_ALL_with_metreics_withCOMPARE()
print_self_training()