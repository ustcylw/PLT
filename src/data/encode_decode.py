import torch
import numpy as np



def encoder(boxes,labels):
    '''
    boxes (tensor) [[x1,y1,x2,y2],[]]
    labels (tensor) [...]
    return 7x7x30
    '''
    grid_num = 14
    target = torch.zeros((grid_num,grid_num,30))
    cell_size = 1./grid_num
    wh = boxes[:,2:]-boxes[:,:2]
    cxcy = (boxes[:,2:]+boxes[:,:2])/2
    for i in range(cxcy.size()[0]):
        cxcy_sample = cxcy[i]
        ij = (cxcy_sample/cell_size).ceil()-1 #
        target[int(ij[1]),int(ij[0]),4] = 1
        target[int(ij[1]),int(ij[0]),9] = 1
        target[int(ij[1]),int(ij[0]),int(labels[i])+9] = 1
        xy = ij*cell_size #匹配到的网格的左上角相对坐标
        delta_xy = (cxcy_sample -xy)/cell_size
        target[int(ij[1]),int(ij[0]),2:4] = wh[i]
        target[int(ij[1]),int(ij[0]),:2] = delta_xy
        target[int(ij[1]),int(ij[0]),7:9] = wh[i]
        target[int(ij[1]),int(ij[0]),5:7] = delta_xy
    return target



def decoder(pred):
    '''
    pred (tensor) 1x7x7x30
    return (tensor) box[[x1,y1,x2,y2]] label[...]
    '''
    grid_num = 14
    boxes=[]
    cls_indexs=[]
    probs = []
    cell_size = 1./grid_num
    pred = pred.data
    pred = pred.squeeze(0) #7x7x30
    contain1 = pred[:,:,4].unsqueeze(2)
    contain2 = pred[:,:,9].unsqueeze(2)
    contain = torch.cat((contain1,contain2),2)
    mask1 = contain > 0.1 #大于阈值
    mask2 = (contain==contain.max()) #we always select the best contain_prob what ever it>0.9
    mask = (mask1+mask2).gt(0)
    # min_score,min_index = torch.min(contain,2) #每个cell只选最大概率的那个预测框
    for i in range(grid_num):
        for j in range(grid_num):
            for b in range(2):
                # index = min_index[i,j]
                # mask[i,j,index] = 0
                if mask[i,j,b] == 1:
                    box = pred[i,j,b*5:b*5+4]
                    contain_prob = torch.FloatTensor([pred[i,j,b*5+4]])
                    xy = torch.FloatTensor([j,i])*cell_size #cell左上角  up left of cell
                    box[:2] = box[:2]*cell_size + xy # return cxcy relative to image
                    box_xy = torch.FloatTensor(box.size())#转换成xy形式    convert[cx,cy,w,h] to [x1,xy1,x2,y2]
                    box_xy[:2] = box[:2] - 0.5*box[2:]
                    box_xy[2:] = box[:2] + 0.5*box[2:]
                    max_prob,cls_index = torch.max(pred[i,j,10:],0)
                    if float((contain_prob*max_prob)[0]) > 0.1:
                        boxes.append(box_xy.view(1,4))
                        cls_indexs.append(cls_index.reshape((1,)))
                        probs.append(contain_prob*max_prob)
    if len(boxes) ==0:
        boxes = torch.zeros((1,4))
        probs = torch.zeros(1)
        cls_indexs = torch.zeros(1)
    else:
        boxes = torch.cat(boxes,0) #(n,4)
        probs = torch.cat(probs,0) #(n,)
        cls_indexs = torch.cat(cls_indexs,0) #(n,)
    keep = nms(boxes,probs)
    return boxes[keep],cls_indexs[keep],probs[keep]


def nms(bboxes, scores, thresh=0.5):
    # bboxes = bboxes[scores.cpu().numpy() > 0.3]
    # scores = scores[scores>0.3]
    # 利用Pytorch实现NMS算法
    x1 = bboxes[:, 0]
    y1 = bboxes[:, 1]
    x2 = bboxes[:, 1]
    y2 = bboxes[:, 1]
    # 计算每个box的面积
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    # 对得分降序排列，order为索引
    _, order = scores.sort(0, descending=True)
    # keep保留了NMS后留下的边框box
    keep = []
    while order.numel() > 0:
        if order.numel() == 1: # 保留框只剩1个
            i = order.item()
            if scores[i] > 0.3:
                keep.append(i)
            break
        else: # 还有保留框没有NMS
            i = order[0].item() # 保留scores最大的那个框box[i]
            if scores[i] > 0.3:
                keep.append(i)
        # 利用tensor.clamp函数求取每个框和当前框的最大值和最小值
        # 将输入input张量每个元素的夹紧到区间 [min,max]，并返回结果到一个新张量
        xx1 = x1[order[1: ]].clamp(min=x1[i]) 
        # 左坐标夹紧的最小值为order中scores最大的框的左坐标，对剩余所有order元素进行夹紧操作
        yy1 = y1[order[1: ]].clamp(min=y1[i]) 
        xx2 = x2[order[1: ]].clamp(max=x2[i]) 
        yy2 = y2[order[1: ]].clamp(max=y2[i]) 
        # 求每一个框和当前框重合部分和总共叠加的面积
        inter = (xx2 - xx2).clamp(min=0) * (yy2 - yy1).clamp(min=0)
        union = areas[i] + areas[order[1: ]] - inter
        # 计算每一个框和当前框的IoU
        IoU = inter / union
        # 保留IoU小于threshold的边框索引
        idx = (IoU <= thresh).nonzero().squeeze()
        if idx.numel() == 0:
            break
        # 这里+1是为了补充idx和order之间的索引差
        order = order[idx+1]
    # 返回保留下的所有边框的索引
    return torch.LongTensor(keep)
