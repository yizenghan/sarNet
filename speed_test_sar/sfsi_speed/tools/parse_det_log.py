import os
file_list = ['/data/home/zhez/git/msra_mmclassification_dev/log/r34_baseline_90ep',
             '/data/home/zhez/git/msra_mmclassification_dev/log/LG_3_3',
             '/data/home/zhez/git/msra_mmclassification_dev/log/LG_5_3',
             '/data/home/zhez/git/msra_mmclassification_dev/log/LG_7_3']
output_suffix = '_smooth'

key_str = 'INFO - Epoch '
# key_str = 'INFO - Epoch('
decay = 0.8
for path in file_list:
    item_dict = {}
    fin = open(path + '.txt', 'r')
    fout = open(path + output_suffix + ".txt", 'w')
    contents = fin.readlines()
    line_cnt = 0
    for line in contents:
        start_idx = line.find(key_str)
        if start_idx != -1:
            line_cnt += 1
            line = line[start_idx + len(key_str):]

            # split iter_epoch info
            tmp = line.split('\t')
            iter_epoch_str = tmp[0]
            line = tmp[1]

            items = line.split(', ')
            output_str = ''
            for item in items:
                key_value = item.split(' ')
                if ('eta' in item) or len(key_value) == 1:
                    continue
                if(key_value[0] in item_dict):
                    item_dict[key_value[0]] += float(key_value[1])
                else:
                    item_dict[key_value[0]] = float(key_value[1])

                output_str += "{0} {1} ".format(key_value[0],  round(item_dict[key_value[0]] / line_cnt, 3))
                item_dict[key_value[0]] *= decay
            line_cnt *= decay
            fout.write(iter_epoch_str +  output_str + "\r\n")

    fout.close()