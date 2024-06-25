import numpy as np
import os
import re


def text_read(file_p, num_col=2):
    f = open(file_p, 'r')
    return_array = []
    for i in range(num_col):
        return_array.append([])
    for line in f.readlines():
        # vas = line.strip().split(r'\s+')
        vas=re.split(r'\s+', line.strip())
        # print(vas)
        for i in range(num_col):
            return_array[i].append(vas[i])
    return return_array


def intervaloverlapvalseconds(i1, i2, normtype=0):
    ov = np.zeros((len(i1), len(i2)))
    for i in range(len(i1)):
        for j in range(len(i2)):
            ov[i, j] = intervalsingleoverlapvalseconds(i1[i], i2[j], normtype)
    return ov


def intervalsingleoverlapvalseconds(i1, i2, normtype):
    i1 = [min(i1), max(i1)]
    i2 = [min(i2), max(i2)]

    ov = 0
    if normtype<0:
        ua = 1
    elif normtype == 1:
        ua = i1[1]-i1[0]
    elif normtype == 2:
        ua = i2[1]-i2[0]
    else:
        bu=[min(i1[0], i2[0]), max(i1[1], i2[1])]
        ua = bu[1] - bu[0]
    bi = [max(i1[0], i2[0]), min(i1[1], i2[1])]
    iw = bi[1] - bi[0]
    if iw > 0:
        if normtype < 0:
            ov = iw
        else:
            ov = iw/ua
    return ov


def TH14EventDetPr(det_events, gt_events, class_n, overlap_thresh):
    gt_video_names = gt_events['video_name']  # type np.array
    det_video_names = det_events['video_name']  # type np.array
    video_names = np.array(list(set(np.concatenate((gt_video_names, det_video_names)))))

    unsortConf = []
    unsortFlag = []
    npos=len(np.where(class_n==gt_events['class_n'])[0])
    assert (npos>0)
    # the Ambiguous exists in gt
    ind_gt_class = np.where(class_n == gt_events['class_n'])[0]  # get the index of gt that contains the name class_n
    ind_det_class = np.where(class_n == det_events['class_n'])[0]
    ind_amb_class = np.where('Ambiguous' == gt_events['class_n'])[0]

    if len(ind_det_class)==0:
        print('Class {cls} no instance, skip '.format(cls=class_n))
        rec=0
        prec=0
        ap=0
        return [rec, prec, ap]

    # correctPortion = np.zeros((len(video_names), 1))
    # groundNum = np.zeros((len(video_names), 1))
    correctPortion = []
    groundNum = []

    # video_names share the unique names in test dataset of videos
    # here, many action instances own the common video name and class id but the different time interval
    for i in range(len(video_names)):
        correctPortion.append(video_names[i])
        gt_ind = list(set(np.where(gt_video_names==video_names[i])[0]).intersection(set(ind_gt_class)))
        amb_ind = list(set(np.where(video_names[i]==gt_video_names)[0]).intersection(set(ind_amb_class)))
        det_ind = list(set(np.where(video_names[i] == det_video_names)[0]).intersection(set(ind_det_class)))

        groundNum.append(len(gt_ind))

        if len(det_ind) > 0:
            det_ind_reserve = np.argsort(-det_events['conf'][det_ind])
            conf = det_events['conf'][det_ind]
            conf = conf[det_ind_reserve]
            det_time_interval = det_events['time_interval'][det_ind]
            det_time_interval = det_time_interval[det_ind_reserve]

            ind_free = np.ones((len(det_ind)))
            ind_amb = np.zeros((len(det_ind)))

            if len(gt_ind) > 0:
                # check the overlap in the same
                ov = intervaloverlapvalseconds(gt_events['time_interval'][gt_ind], det_time_interval )
                for k in range(np.shape(ov)[0]):
                    ind = np.where(ind_free == 1)[0]
                    if len(ind) == 0:
                        break
                    vm = max(ov[k, ind])
                    im = np.where(ov[k, ind]==vm)[0][0]
                    if vm > overlap_thresh:
                        ind_free[ind[im]] = 0

            if len(amb_ind) > 0:
                ovamb = intervaloverlapvalseconds(gt_events['time_interval'][amb_ind], det_time_interval )
                ind_amb = np.sum(ovamb, axis=0)
            # the array storage the index of instance that satisfies the condition
            idx1 = np.where(ind_free == 0)[0]
            # the array storage the index of instance that fails to meet the condition: gt(no ambiguous), ambiguous
            idx2 = np.where((ind_free == 1)*(ind_amb == 0))[0]
            flag = np.concatenate((np.ones((len(idx1),)), 2*np.ones((len(idx2),))), axis=0)

            # the indexes that satisfy all the conditions
            idxall = np.sort(np.concatenate((idx1, idx2)))
            # sorted the indexes for flag value aligned
            ttIdx = np.argsort(np.concatenate((idx1, idx2)))

            flagall = flag[ttIdx]
            unsortConf = np.concatenate((unsortConf, conf[idxall]))
            unsortFlag = np.concatenate((unsortFlag, flagall))

            if len(gt_ind) != 0:
                correctPortion[i] = len(np.where(ind_free==0)[0])/len(gt_ind)

    # conf = np.concatenate((unsortConf, unsortFlag), axis=1)
    conf = np.vstack((unsortConf, unsortFlag))
    ind_s = np.argsort(-conf[0])

    # tp = np.cumsum(conf[1][ind_s]==1)
    tp = np.cumsum(conf[1][ind_s] == 1)
    fp = np.cumsum(conf[1][ind_s]==2)

    tmp = conf[1][ind_s]==1
    rec = tp/npos
    prec = np.true_divide(tp, fp+tp)

    ap = prap(rec, prec, tmp, npos)

    return rec, prec, ap


def TH14EventDetPrVideo(det_events, gt_events, class_n, overlap_thresh):
    gt_video_names = gt_events['video_name']  # type np.array
    det_video_names = det_events['video_name']  # type np.array
    video_names = np.array(list(set(np.concatenate((gt_video_names, det_video_names)))))

    unsortConf = []
    unsortFlag = []
    npos=len(np.where(class_n==gt_events['class_n'])[0])
    assert (npos>0)
    # the Ambiguous exists in gt
    ind_gt_class = np.where(class_n == gt_events['class_n'])[0]  # get the index of gt that contains the name class_n
    ind_det_class = np.where(class_n == det_events['class_n'])[0]
    ind_amb_class = np.where('Ambiguous' == gt_events['class_n'])[0]
    cls_videos_ap = {}
    if len(ind_det_class)==0:
        print('Class {cls} no instance, skip '.format(cls=class_n))
        rec=0
        prec=0
        ap=0
        return rec, prec, ap, cls_videos_ap

    # correctPortion = np.zeros((len(video_names), 1))
    # groundNum = np.zeros((len(video_names), 1))
    correctPortion = []
    groundNum = []

    # cls_videos_ap = {}
    # aps = 0
    # cul_num = 0

    # video_names share the unique names in test dataset of videos
    # here, many action instances own the common video name and class id but the different time interval
    for i in range(len(video_names)):
        correctPortion.append(video_names[i])
        gt_ind = list(set(np.where(gt_video_names==video_names[i])[0]).intersection(set(ind_gt_class)))
        amb_ind = list(set(np.where(video_names[i]==gt_video_names)[0]).intersection(set(ind_amb_class)))
        det_ind = list(set(np.where(video_names[i] == det_video_names)[0]).intersection(set(ind_det_class)))

        groundNum.append(len(gt_ind))

        if len(det_ind) > 0:
            det_ind_reserve = np.argsort(-det_events['conf'][det_ind])
            conf = det_events['conf'][det_ind]
            conf = conf[det_ind_reserve]
            det_time_interval = det_events['time_interval'][det_ind]
            det_time_interval = det_time_interval[det_ind_reserve]

            ind_free = np.ones((len(det_ind)))
            ind_amb = np.zeros((len(det_ind)))

            if len(gt_ind) > 0:
                # check the overlap in the same
                ov = intervaloverlapvalseconds(gt_events['time_interval'][gt_ind], det_time_interval )
                for k in range(np.shape(ov)[0]):
                    ind = np.where(ind_free == 1)[0]
                    if len(ind) == 0:
                        break
                    vm = max(ov[k, ind])
                    im = np.where(ov[k, ind]==vm)[0][0]
                    if vm > overlap_thresh:
                        ind_free[ind[im]] = 0

            if len(amb_ind) > 0:
                ovamb = intervaloverlapvalseconds(gt_events['time_interval'][amb_ind], det_time_interval )
                ind_amb = np.sum(ovamb, axis=0)
            # the array storage the index of instance that satisfies the condition
            idx1 = np.where(ind_free == 0)[0]
            # the array storage the index of instance that fails to meet the condition: gt(no ambiguous), ambiguous
            idx2 = np.where((ind_free == 1)*(ind_amb == 0))[0]
            flag = np.concatenate((np.ones((len(idx1),)), 2*np.ones((len(idx2),))), axis=0)

            # the indexes that satisfy all the conditions
            idxall = np.sort(np.concatenate((idx1, idx2)))
            # sorted the indexes for flag value aligned
            ttIdx = np.argsort(np.concatenate((idx1, idx2)))

            flagall = flag[ttIdx]
            unsortConf = np.concatenate((unsortConf, conf[idxall]))
            unsortFlag = np.concatenate((unsortFlag, flagall))

            if len(gt_ind) != 0:
                correctPortion[i] = len(np.where(ind_free==0)[0])/len(gt_ind)
                v_pos = len(gt_ind)
                rec, prec, ap = calculate_ap(conf[idxall], flagall, v_pos)
                # if video_names[i] not in cls_videos_ap.keys():
                # cls_videos_ap[video_names[i]] = [rec, prec, ap]
                cls_videos_ap[video_names[i]] = [ap, v_pos]
                # aps += ap
                # cul_num += 1
    # mean_aps = aps // cul_num
    # print(mean_aps)

    rec, prec, ap = calculate_ap(unsortConf, unsortFlag, npos)

    return rec, prec, ap, cls_videos_ap


def calculate_ap(unsortConf, unsortFlag, npos):
    conf = np.vstack((unsortConf, unsortFlag))
    ind_s = np.argsort(-conf[0])

    tp = np.cumsum(conf[1][ind_s] == 1)
    fp = np.cumsum(conf[1][ind_s] == 2)

    tmp = conf[1][ind_s] == 1
    rec = tp / npos
    prec = np.true_divide(tp, fp + tp)

    ap = prap(rec, prec, tmp, npos)
    return rec, prec, ap


def prap(rec, prec, tmp, npos):
    ap = 0
    for i in range(len(prec)):
        if tmp[i]:  # here need to be changed
            ap += prec[i]
    ap /= npos
    return ap


def TH14evalDet1(det_filename, gt_path, subset, threshold=0.5):
    [th14classids, th14classnames] = text_read(gt_path+'/detclasslist.txt', num_col=2)
    th14classids = np.array(list(range(len(th14classids))))
    gt_events = {}
    gt_events['video_name'] = []
    gt_events['time_interval'] = []
    gt_events['class_n'] = []
    gt_events['conf'] = []

    gt_events_count = 0
    # th14_class_names_amb = th14classnames.insert(0, 'Ambiguous')  # 21 class names with Ambiguous
    th14_class_names_amb = ['Ambiguous'] + th14classnames  # 21 class names with Ambiguous

    for i in range(len(th14_class_names_amb)):
        class_n = th14_class_names_amb[i]
        gt_file_name = gt_path+'/'+class_n+'_'+subset+'.txt'
        if not os.path.exists(gt_file_name):
            print('TH14evaldet: Could not find GT file ', gt_file_name)
            exit(0)
        [video_names, t1, t2] = text_read(gt_file_name, num_col=3)
        # print(t1)
        # print(t1[0])
        t1 = list(map(float, t1))
        t2 = list(map(float, t2))

        # the gt_events storage all of the action instance which contains four properties,
        # e.g. video name, time interval, class name, and confidence scores of timestamp
        for j in range(len(video_names)):
            gt_events_count += 1
            gt_events['video_name'].append(video_names[j])
            gt_events['time_interval'].append([t1[j], t2[j]])
            gt_events['class_n'].append(class_n)
            gt_events['conf'].append(1)

    if not os.path.exists(det_filename):
        print('TH14evaldet: Could not find file ', det_filename)
        exit(0)

    # get the location val captured by the proposed method
    # e.g. video name, time interval: start and end, class id, and confidence scores of timestamp
    # v.s. video_test_0001223 77.83 82.27 93 0.671 or video_test_0001223 29.63 33.20 93 0.6520
    [video_names, t1, t2, cls_id, conf] = text_read(det_filename, num_col=5)
    t1 = list(map(float, t1))
    t2 = list(map(float, t2))
    conf = list(map(float, conf))
    cls_id = np.array(list(map(int, cls_id)))


    # print(cls_id)

    for i in range(len(video_names)):
        video_names[i] = video_names[i].replace('.mp4', '')

    # own the same type to gt_events
    det_events = {}
    det_events['video_name'] = []
    det_events['time_interval'] = []
    det_events['class_n'] = []
    det_events['conf'] = []
    for i in range(len(video_names)):
        # ind = np.where(th14classids==cls_id[i])[0]
        ind = np.where(th14classids == cls_id[i])[0]
        if len(ind)>0:
            det_events['video_name'].append(video_names[i])
            det_events['time_interval'].append([t1[i], t2[i]])
            det_events['class_n'].append(th14classnames[int(ind[0])])
            det_events['conf'].append(conf[i])
        else:
            print("Warning: Reported class Id {id} is not among THUMOS14 detection classes.".format(id=int(cls_id[i])))
            continue

    for key in gt_events.keys():
        gt_events[key] = np.array(gt_events[key])

    for key in det_events.keys():
        det_events[key] = np.array(det_events[key])

    ap_all = []
    cls_video_aps_dict = {}
    pr_all = {}
    pr_all['class_n'] = []
    pr_all['class_id'] = []
    pr_all['overlap_thresh'] = []
    pr_all['prec'] = []
    pr_all['rec'] = []
    pr_all['ap'] = []
    # pr_all['cls_video_aps'] = []

    for i in range(len(th14classnames)):
        class_n = th14classnames[i]
        class_id = th14classnames.index(class_n)
        # assert (len(class_id) == 1)
        rec, prec, ap, cls_video_aps = TH14EventDetPrVideo(det_events, gt_events, class_n, threshold)
        pr_all['class_n'].append(class_n)
        pr_all['class_id'].append(class_id)
        pr_all['overlap_thresh'].append(threshold)
        pr_all['prec'].append(prec)
        pr_all['rec'].append(rec)
        pr_all['ap'].append(ap)
        cls_video_aps_dict[str(class_id)] = [class_n, ap, cls_video_aps]
        # pr_all['cls_video_aps'].append(cls_video_aps)
        ap_all.append(ap)

        print('AP:{ap0:.3f} at overlap {ov0:.1f} for {cls_n}'.format(ap0=ap, ov0=threshold, cls_n=class_n))

    r_map = np.mean(ap_all)
    print('MAP: {m:.5f}'.format(m=r_map))
    return ap_all, r_map, cls_video_aps_dict, pr_all['class_n']



if __name__=="__main__":
    path = '../Labels/thumos14-test-annotations/detclasslist.txt'
    det_file_path = '../outputs/predictions-raw/thumos-I3D-run-0-18000-flow-test'

    [video_names, t1, t2, cls_id, conf] = text_read(det_file_path, num_col=5)
    print(video_names[0])
    # th14classids, th14classnames = text_read(path)
    # print(th14classnames)
    # for i in range(len(th14classnames)):
    #     class_n = th14classnames[i]
    #     class_id = th14classnames.index(class_n)
    #     print(class_id)

    # print(th14classids)
    # th14classids = list(map(int, th14classids))
    # print(th14classids)
    # ind = th14classids.index(99)
    # print(ind)
    # # th14classnames.repalce('D', 'd')
    # for i in range(len(th14classnames)):
    #     th14classnames[i] = th14classnames[i].replace('B', '')
    # print(th14classnames)
    # a = text_read(path)
    # a.insert(0, 1)
    # print(a)
    # f = '0.38'
    # f1
    # f2
    # ['%f', "%s", "%d"]

    # f = f.format('%f')
    # print(type(f))
    print("Warning: Reported class Id {id} is not among THUMOS14 detection classes.".format(id=10))
    pass
