import numpy as np
import pandas as pd

# 按合并列聚合数据，并提供聚合后每个组的正样本数量，负样本数量，给后面合并使用
def getBaseData(df, variable, flag, sample=None, varOrder=True):
    """
    param df:            DataFrame|数据集
    param varialbe:      str|数据集中需要合并的列名
    param flag:          str|正负样本标识的列名
    param sample:        int|抽样数目，默认不进行抽样
    param varOrder:      Boolean|合并列的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase
    """

    # 判断是否需要抽样操作
    if sample is not None:
        df = df.sample(n=sample)
        
    # 对数据进行预处理
    total_num = df.groupby([variable])[flag].count() # 合并变量每个值的数目
    total_num = pd.DataFrame({'total_num':total_num})
    positive_class = df.groupby([variable])[flag].sum() # 合并变量每个值的正样本数
    positive_class = pd.DataFrame({'positive_class':positive_class})
    regroup = pd.merge(total_num, positive_class, left_index=True, right_index=True, how='inner')
    regroup.reset_index(inplace=True) # groupby处理之后正好是按变量大小排序的
    regroup['negative_class'] = regroup['total_num'] - regroup['positive_class'] # 合并变量每个值的负样本数
    regroup['positive_pct'] = regroup['positive_class']/regroup['total_num']
    if varOrder is False: # 如果合并变量的值没有大小顺序，则需要按正样本占比排序
        regroup.sort_values(by='positive_pct', inplace=True) # 后面计算只用正样本数和负样本数，去除多余变量
    regroup = regroup.drop(['total_num','positive_pct'], axis=1) 
    np_regroup = np.array(regroup) # 把DataFrame转换为numpy，提高运行效率
    return np_regroup

# 输出合并后结果
def getMergedResult(np_regroup, variable, varOrder, varInterval):
    """
    param np_regroup:    numpy|已经合并好的数据
    param varialbe:      str|本次合并的列名
    param varOrder:      Boolean|合并列的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase
    param varInterval:   Boolean|合并列的值是否是区间，默认为False，如果是分箱后合并要设置为True
    """

    result_data = pd.DataFrame()
    result_data['variable'] = [variable] * np_regroup.shape[0]
   
    if varOrder:
        list_temp = []
        for i in np.arange(np_regroup.shape[0]):
            if i==0:
                if varInterval:
                    x = '<=' + str(np_regroup[i, 0].right)
                else:
                    x = '<=' + str(np_regroup[i, 0])
            elif i==np_regroup.shape[0] - 1:
                if varInterval:
                    x = '>' + str(np_regroup[i-1, 0].right)
                else:
                    x = '>' + str(np_regroup[i-1, 0])
            else:
                if varInterval:
                    x = '(' + str(np_regroup[i-1, 0].right) + ',' + str(np_regroup[i, 0].right)  + ']'
                else:
                    x = '(' + str(np_regroup[i-1, 0]) + ',' + str(np_regroup[i, 0])  + ']'
            list_temp.append(x)
        result_data['interval'] = list_temp
    else:
        result_data['interval'] = np_regroup[:, 0]
       
    result_data['flag_0'] = np_regroup[:, 2]
    result_data['flag_1'] = np_regroup[:, 1]
    
    return result_data
        

# 通过正样本率进行合并，核心是每次将正样本率差距最小的区间合并，直到达到设定的数量
def varMergeByPct(df, variable, flag, bins=10, sample=None, varInterval=False, varOrder=True):
    """
    param df:            DataFrame|数据集
    param varialbe:      str|数据集中需要合并的列名
    param flag:          str|正负样本标识的列名
    param bins:          int|合并后数量（要求合并后数量等于此值）
    param sample:        int|抽样数目，默认不进行抽样
    param varInterval:   Boolean|合并列的值是否是区间，默认为False，如果是分箱后合并要设置为True
    param varOrder:      Boolean|合并列的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase
    """
    
    # 获取计算所需基础数据
    np_regroup = getBaseData(df, variable, flag, sample, varOrder)
    
           
    # 对相邻两个区间的正样本率（组内正样本数/组内总样本数) 之差的绝对值进行计算
    pctdiff_table = np.array([]) # 保存相邻两组的正样本率之差的绝对值
    for i in np.arange(np_regroup.shape[0] - 1):
        pctdiff = abs(np_regroup[i+1, 1]/(np_regroup[i+1, 1] + np_regroup[i+1, 2]) 
                   - np_regroup[i, 1]/(np_regroup[i, 1] + np_regroup[i, 2]))
        pctdiff_table = np.append(pctdiff_table, pctdiff)
              
    # 把正样本率之差的绝对值最小的两个区间合并, 直到数量小于设定的合并后数量bins
    while(1):
        if len(pctdiff_table) <= (bins-1):
            break
        pctdiff_min_index = np.argwhere(pctdiff_table == min(pctdiff_table))[0] # 找出正样本率之差的绝对值最小的索引位置
        np_regroup[pctdiff_min_index, 1] = np_regroup[pctdiff_min_index, 1] + np_regroup[pctdiff_min_index+1, 1] # 正样本合并
        np_regroup[pctdiff_min_index, 2] = np_regroup[pctdiff_min_index, 2] + np_regroup[pctdiff_min_index+1, 2] # 负样本合并
        if varOrder:
            np_regroup[pctdiff_min_index, 0] = np_regroup[pctdiff_min_index+1, 0] # 更新分箱变量的范围
        else:
            np_regroup[pctdiff_min_index, 0] = np_regroup[pctdiff_min_index, 0] + '|' + np_regroup[pctdiff_min_index+1, 0] # 更新分箱变量的范围(无序的话要保留每个区间)
        
        np_regroup = np.delete(np_regroup, pctdiff_min_index + 1, axis=0)
        
        # 更新正样本率之差 pctdiff_table
        if(pctdiff_min_index == np_regroup.shape[0] - 1): # 如果正样本率之差绝对值的最小值是最后两个区间的时候
            pctdiff_table[pctdiff_min_index - 1] = abs(np_regroup[pctdiff_min_index, 1]/(np_regroup[pctdiff_min_index, 1] + np_regroup[pctdiff_min_index, 2]) 
                       - np_regroup[pctdiff_min_index-1, 1]/(np_regroup[pctdiff_min_index-1, 1] + np_regroup[pctdiff_min_index-1, 2]))
            pctdiff_table = np.delete(pctdiff_table, pctdiff_min_index, axis=0)

        elif(pctdiff_min_index == 0): # 如果正样本率之差绝对值的最小值是最前面两个区间的时候
            pctdiff_table[pctdiff_min_index] = abs(np_regroup[pctdiff_min_index+1, 1]/(np_regroup[pctdiff_min_index+1, 1] + np_regroup[pctdiff_min_index+1, 2]) 
                       - np_regroup[pctdiff_min_index, 1]/(np_regroup[pctdiff_min_index, 1] + np_regroup[pctdiff_min_index, 2]))
            pctdiff_table = np.delete(pctdiff_table, pctdiff_min_index+1, axis=0)
        else:
            pctdiff_table[pctdiff_min_index - 1] = abs(np_regroup[pctdiff_min_index, 1]/(np_regroup[pctdiff_min_index, 1] + np_regroup[pctdiff_min_index, 2]) 
                       - np_regroup[pctdiff_min_index-1, 1]/(np_regroup[pctdiff_min_index-1, 1] + np_regroup[pctdiff_min_index-1, 2]))
            pctdiff_table[pctdiff_min_index] = abs(np_regroup[pctdiff_min_index+1, 1]/(np_regroup[pctdiff_min_index+1, 1] + np_regroup[pctdiff_min_index+1, 2]) 
                       - np_regroup[pctdiff_min_index, 1]/(np_regroup[pctdiff_min_index, 1] + np_regroup[pctdiff_min_index, 2]))
            pctdiff_table = np.delete(pctdiff_table, pctdiff_min_index+1, axis=0)
          
    # 返回结果
    return getMergedResult(np_regroup, variable, varOrder, varInterval)


# 按卡方合并（有的地方也叫卡方分箱，这里为了区别分箱和合并，把卡方合并，和其他合并放在一起讨论)
def varMergeByChiSquare(df, variable, flag, confidenceVal=3.841, bins=10, sample=None, varInterval=False, varOrder=True):
    """
    param df:            DataFrame|数据集
    param varialbe:      str|数据集中需要分箱的列名（本函数适用对数值类型的列进行分箱）
    param flag:          str|正负样本标识的列名
    param confidenceVal: float|卡方临界值（3.841是自由度1置信度95%对应的临界值，小于此值说明组还可以合并)
    param bins:          int|分箱数量（要求分箱后数量小于等于此值）
    param sample:        int|抽样数目，默认不进行抽样
    param varInterval:   Boolean|合并列的值是否是区间，默认为False，如果是分箱后合并要设置为True
    param varOrder:      Boolean|合并列的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase

    """
    
    # 获取计算所需基础数据
    np_regroup = getBaseData(df, variable, flag, sample, varOrder)
    
    # 处理连续没有正样本或负样本的区间，进行区间合并(如果连续区间的正样本或负样本数为0，则卡方计算的分母为0，所以要合并)
    i = 0 
    while(i <= np_regroup.shape[0] - 2):
        if((np_regroup[i,1] == 0 and np_regroup[i+1, 1]==0) 
           or (np_regroup[i, 2]==0 and np_regroup[i+1, 2] == 0)):
            np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i+1, 1] # 正样本合并
            np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i+1, 2] # 负样本合并
            if varOrder:
                np_regroup[i, 0] = np_regroup[i+1, 0] # 更新分箱变量范围
            else:
                np_regroup[i, 0] = np_regroup[i, 0] + '|' + np_regroup[i+1, 0] # # 更新分箱变量的范围(无序的话要保留每个区间)
            
            np_regroup = np.delete(np_regroup, i+1, axis=0) # 删除整行
            i = i - 1
        i = i + 1
        
    # 对相邻两个区间进行卡方值计算
    chi_table = np.array([]) # 保存项相邻两个区间的卡方值
    for i in np.arange(np_regroup.shape[0] - 1):
        chi = (np_regroup[i, 1] * np_regroup[i+1, 2] - np_regroup[i, 2] * np_regroup[i+1, 1])**2 * \
              (np_regroup[i, 1] + np_regroup[i, 2] + np_regroup[i+1, 1] + np_regroup[i+1, 2]) / \
              ((np_regroup[i, 1] + np_regroup[i, 2]) * (np_regroup[i+1, 1] + np_regroup[i+1, 2]) * \
               (np_regroup[i, 1] + np_regroup[i+1, 1]) * (np_regroup[i, 2] + np_regroup[i+1, 2]))
        chi_table = np.append(chi_table, chi)
    
    # 把卡方值最小的两个区间合并, 直到分箱数量小于等于设定的分箱数量bins且相邻区间的卡方值没有大于设定的临界值confidenceVal的组了
    while(1):
        if(len(chi_table) <= (bins-1) and min(chi_table) >= confidenceVal):
            break
        chi_min_index = np.argwhere(chi_table == min(chi_table))[0] # 找出卡方值最小的索引位置
        np_regroup[chi_min_index, 1] = np_regroup[chi_min_index, 1] + np_regroup[chi_min_index+1, 1] # 正样本合并
        np_regroup[chi_min_index, 2] = np_regroup[chi_min_index, 2] + np_regroup[chi_min_index+1, 2] # 负样本合并
        if varOrder:
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index+1, 0] # 更新分箱变量的范围
        else:
            np_regroup[chi_min_index, 0] = np_regroup[chi_min_index, 0] + '|' + np_regroup[chi_min_index+1, 0] # 更新分箱变量的范围(无序的话要保留每个区间)
        
        np_regroup = np.delete(np_regroup, chi_min_index + 1, axis=0)
        
        # 更新卡方值表 chi_table
        if(chi_min_index == np_regroup.shape[0] - 1): # 如果卡方最小值是最后两个区间的时候
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index-1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index-1, 2] * np_regroup[chi_min_index, 1])**2 * \
                  (np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index-1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                  ((np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index-1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * \
                   (np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index-1, 2] + np_regroup[chi_min_index, 2]))      
            chi_table = np.delete(chi_table, chi_min_index, axis=0)
        elif(chi_min_index == 0): # 如果卡方最小值是最前面两个区间的时候
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index+1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index+1, 1])**2 * \
                  (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index+1, 1] + np_regroup[chi_min_index+1, 2]) / \
                  ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index+1, 1] + np_regroup[chi_min_index+1, 2]) * \
                   (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index+1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index+1, 2]))      
            chi_table = np.delete(chi_table, chi_min_index+1, axis=0)
        else:
            chi_table[chi_min_index - 1] = (np_regroup[chi_min_index-1, 1] * np_regroup[chi_min_index, 2] - np_regroup[chi_min_index-1, 2] * np_regroup[chi_min_index, 1])**2 * \
                  (np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index-1, 2] + np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) / \
                  ((np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index-1, 2]) * (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * \
                   (np_regroup[chi_min_index-1, 1] + np_regroup[chi_min_index, 1]) * (np_regroup[chi_min_index-1, 2] + np_regroup[chi_min_index, 2]))  
            chi_table[chi_min_index] = (np_regroup[chi_min_index, 1] * np_regroup[chi_min_index+1, 2] - np_regroup[chi_min_index, 2] * np_regroup[chi_min_index+1, 1])**2 * \
                  (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2] + np_regroup[chi_min_index+1, 1] + np_regroup[chi_min_index+1, 2]) / \
                  ((np_regroup[chi_min_index, 1] + np_regroup[chi_min_index, 2]) * (np_regroup[chi_min_index+1, 1] + np_regroup[chi_min_index+1, 2]) * \
                   (np_regroup[chi_min_index, 1] + np_regroup[chi_min_index+1, 1]) * (np_regroup[chi_min_index, 2] + np_regroup[chi_min_index+1, 2]))      
            chi_table = np.delete(chi_table, chi_min_index+1, axis=0)
    
    # 返回结果
    return getMergedResult(np_regroup, variable, varOrder, varInterval)


# 获取按分割点分割后的IV值
def getIV(dat, splits, pos_col, neg_col):
    """
    param dat:         np.array|数据集
    param splits:      np.array(一维数组)|分割点
    param pos_col:     str|正样本列
    param neg_col:     str|负样本列
    """
    splits = np.sort(splits)
    split_list = []
    for i in range(len(splits)):
        split_list.append(int(splits[i])+1) # 因为np.vsplit中是按不到每个元素的分割, 而实际分割点是在分割点之上（包含）都算在一个区间，所以+1
    dat_splits = np.vsplit(dat, split_list) 
    iv_list = np.array([])
    total_pos = np.sum(dat[:, pos_col])
    total_neg = np.sum(dat[:, neg_col])
    for s in dat_splits:
        pos = np.sum(s[:, pos_col])
        neg = np.sum(s[:, neg_col])
        if pos == 0 or neg == 0:
            return -np.inf
        iv = (pos/total_pos - neg/total_neg) * np.log((pos/total_pos)/(neg/total_neg))
        iv_list = np.append(iv_list, iv) 
    return np.sum(iv_list)    


# 获取数据集中IV值最大的分割点
def getMaxIVSplit(dat, pos_col, neg_col): 
    """
    param dat:         np.array|数据集
    param pos_col:     str|正样本列
    param neg_col:     str|负样本列
    """
    if dat.shape[0] <= 1:
        return None
    else:
        iv_list = np.array([]) 
        for i in range(dat.shape[0]-1):
            p1 = np.sum(dat[0:(i+1), pos_col])
            n1 = np.sum(dat[0:(i+1), neg_col])
            p2 = np.sum(dat[i+1:, pos_col])
            n2 = np.sum(dat[i+1:, neg_col])
            if (p1==0 or p2==0 or n1==0 or n2==0):
                iv_list = np.append(iv_list, -np.inf)
            else:
                iv1 = (p1/(p1+p2) - n1/(n1+n2)) * np.log((p1/(p1+p2))/(n1/(n1+n2)))
                iv2 = (p2/(p1+p2) - n2/(n1+n2)) * np.log((p2/(p1+p2))/(n2/(n1+n2)))
                iv_list = np.append(iv_list, iv1+iv2)
        
        iv_max = max(iv_list)      
        iv_split_index = np.argwhere(iv_list == iv_max)[0]
        return iv_split_index
    

# 每次选取分割的点，使得IV值最大，不断分割，直至分割后的数量达到设定的数量
def varMergeByIVSplit(df, variable, flag, bins=10, sample=None, varInterval=False, varOrder=True):
    """
    param df:            DataFrame|数据集
    param varialbe:      str|数据集中需要分箱的列名（本函数适用对数值类型的列进行分箱）
    param flag:          str|正负样本标识的列名
    param bins:          int|分箱数量（要求分箱后数量小于等于此值）
    param sample:        int|抽样数目，默认不进行抽样
    param varInterval:   Boolean|合并列的值是否是区间，默认为False，如果是分箱后合并要设置为True
    param varOrder:      Boolean|合并列的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase
    """
       
    # 获取计算所需基础数据
    np_regroup = getBaseData(df, variable, flag, sample, varOrder)
    
    split_table = np.array([]) # 用来保存分割的位置
    for t in range(bins-1):
        iv_best = None
        iv_best_index = None
        if t == 0:        
            iv_best_index = getMaxIVSplit(np_regroup, pos_col=1, neg_col=2)
            split_table = np.append(split_table, iv_best_index)
        else:
            iv_best_index_waitlist = np.array([]) # 用来保存每组中的最优分割位置
            for s in range(len(split_table)):
                if s == 0:
                    start = 0
                    end = int(split_table[s])+1
                    iv_index_select = getMaxIVSplit(np_regroup[start:end,:], pos_col=1, neg_col=2)
                    if iv_index_select is not None:
                        iv_best_index_waitlist = np.append(iv_best_index_waitlist, iv_index_select)
                    
                elif s>=1 and s <= len(split_table) - 1:
                    start = int(split_table[s-1])+1
                    end = int(split_table[s])+1
                    iv_index_select = getMaxIVSplit(np_regroup[start:end,:], pos_col=1, neg_col=2)
                    if iv_index_select is not None:
                        iv_best_index_waitlist = np.append(iv_best_index_waitlist, iv_index_select + split_table[s-1] + 1)
                    
                # 如果是最后一个split        
                if s == len(split_table)-1:  
                    # 最后一段
                    start = int(split_table[s])+1
                    iv_index_select = getMaxIVSplit(np_regroup[start:,:], pos_col=1, neg_col=2)
                    if iv_index_select is not None:
                        iv_best_index_waitlist = np.append(iv_best_index_waitlist, iv_index_select + split_table[s] + 1)
            for k in range(len(iv_best_index_waitlist)):
                splits = np.append(split_table, iv_best_index_waitlist[k])
                if k==0:
                    iv_best = getIV(np_regroup, splits, pos_col=1, neg_col=2)
                    iv_best_index = iv_best_index_waitlist[k]
                else:
                    iv_calc = getIV(np_regroup, splits, pos_col=1, neg_col=2)
                    if iv_calc > iv_best:
                        iv_best = iv_calc
                        iv_best_index = iv_best_index_waitlist[k]
            if iv_best_index is None: # 无法找到最优index，则结束分割
                break
            split_table = np.append(split_table, iv_best_index)
            split_table = np.sort(split_table) # 每次加入后要重新排序
            
   # 保存结果
    result_data = pd.DataFrame()
    result_data['variable'] = [variable] * bins
    
    list_temp = []
    list_pos = []
    list_neg = []
    for s in range(len(split_table)):
        if s == 0:
            start = 0 
            end = int(split_table[s])+1
            if varOrder:
                if varInterval:
                    x = '<=' + str(np_regroup[end-1, 0].right)
                else:
                    x = '<=' + str(np_regroup[end-1, 0])
            else:
                x = '|'.join(np_regroup[start:end, 0])
        elif s >= 1 and s <= len(split_table) - 1:
            start = int(split_table[s-1])+1
            end = int(split_table[s])+1
            if varOrder:
                if varInterval:
                    x = '(' + str(np_regroup[start-1, 0].right) + ',' + str(np_regroup[end-1, 0].right)  + ']'
                else:
                    x = '(' + str(np_regroup[start-1, 0]) + ',' + str(np_regroup[end-1, 0])  + ']'
            else:
                x = '|'.join(np_regroup[start:end, 0])
        y = np.sum(np_regroup[start:end, 1])
        z = np.sum(np_regroup[start:end, 2])
        list_temp.append(x)
        list_pos.append(y)
        list_neg.append(z)
        if s == len(split_table) - 1:
            start = int(split_table[s])+1
            if varOrder:
                if varInterval:
                    x = '>' + str(np_regroup[start-1, 0].right)
                else:
                    x = '>' + str(np_regroup[start-1, 0])
            else:
                x = '|'.join(np_regroup[start:, 0])
            y = np.sum(np_regroup[start:, 1])
            z = np.sum(np_regroup[start:, 2])
            list_temp.append(x)
            list_pos.append(y)
            list_neg.append(z)
                    
    result_data['interval'] = list_temp
    result_data['flag_0'] = list_pos
    result_data['flag_1'] = list_neg
    
    return result_data


# 把类型变量转换为WOE编码（类型变量->数值型变量）
def categoryToWOE(df, variable, flag, sample=None, varOrder=True):
    """
    param df:            DataFrame|数据集
    param varialbe:      str|数据集中需要转换的列名
    param flag:          str|正负样本标识的列名
    param sample:        int|抽样数目，默认不进行抽样
    param varOrder:      Boolean|需要转换的列中的值是否存在顺序关系，默认为True，如果是种类变量且无大小顺序，要设置为Flase
    """
   # 获取计算所需基础数据
    np_regroup = getBaseData(df, variable, flag, sample, varOrder)
    
    pos_class_totalcnt = np.sum(np_regroup[:,1])
    neg_class_totalcnt = np.sum(np_regroup[:,2])
    
    # 处理没有正样本或负样本的区间，进行区间合并(如果某个组的正样本或负样本数为0，则计算WOE会出现ln函数中分子或者分母为0的情况，无法计算)
    i = 0 
    while(i <= np_regroup.shape[0] - 1):
        if((np_regroup[i,1] == 0 or np_regroup[i, 2]==0)):
            if i == 0:
                np_regroup[i, 1] = np_regroup[i, 1] + np_regroup[i+1, 1] # 正样本合并
                np_regroup[i, 2] = np_regroup[i, 2] + np_regroup[i+1, 2] # 负样本合并
                np_regroup[i, 0] = str(np_regroup[i, 0]) + '|' + str(np_regroup[i+1, 0]) # 更新变量范围(要保留每个区间)
                np_regroup = np.delete(np_regroup, i+1, axis=0)
            else:
                np_regroup[i-1, 1] = np_regroup[i-1, 1] + np_regroup[i, 1] # 正样本合并
                np_regroup[i-1, 2] = np_regroup[i-1, 2] + np_regroup[i, 2] # 负样本合并
                np_regroup[i-1, 0] = str(np_regroup[i-1, 0]) + '|' + str(np_regroup[i, 0]) # 更新变量范围(要保留每个区间)
                np_regroup = np.delete(np_regroup, i, axis=0) # 删除整行
            i = i - 1
        i = i + 1
        
    np_regroup_df = pd.DataFrame(np_regroup, columns = [variable, 'poscnt', 'negcnt'])
    np_regroup_df['poscnt_pct'] = np_regroup_df['poscnt']/pos_class_totalcnt
    np_regroup_df['negcnt_pct'] = np_regroup_df['negcnt']/neg_class_totalcnt
    tmp = np_regroup_df['poscnt_pct']/np_regroup_df['negcnt_pct']
    np_regroup_df['WOE'] = tmp.apply(lambda x: np.log(x))
    # np_regroup_df['IV'] = (np_regroup_df['poscnt_pct'] - np_regroup_df['negcnt_pct']) * np_regroup_df['WOE']

    variable_woe = dict()
    for item in np_regroup_df[[variable, 'WOE']].values:
        key = str(item[0]).split('|') # 恢复变量的值
        value = item[1]
        for k in key:
            variable_woe[k] = value
            
    variable_woe_df = pd.DataFrame([variable_woe]).T
    variable_woe_df = variable_woe_df.reset_index()
    variable_woe_df.columns = [variable, variable+'_WOE']
    
    return variable_woe_df 