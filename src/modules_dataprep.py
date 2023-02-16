# libraries
import re
import pandas as pd
import numpy as np
import json

def get_exploinfo(path_to_json,dictnames):
    """ Create a dataset with info by video 
    such as gopro model, seconds of metadata, frequency used for accelerometer and gyroscope
    
    Arguments:
    path_to_json: path where the files can be found
    dictnames: list of files' names to be read from

    Output:
    DataFrame with the following columns:
    video_id, takeoff (yes/no), gopro model, seconds of metadata, 
    frequency of accelerometer, frequency of gyroscope

    """
    video_name   = []
    gopro_models = []
    metadata_len = []
    freq_accl    = []
    freq_gyro    = []

    for file in dictnames:
        fullfile  = path_to_json+file
        read_dict = np.load(fullfile+'.npy',allow_pickle='TRUE').item()
        
        gopro_models.append(read_dict['DEVC0']['MINF'])
        
        if 'longboard' in file:
            metadata = read_dict['NOTE0'].split(',')[1]
        else:
            metadata = read_dict['NOTE'].split(',')[1]
            
        match    = [float(x) for x in re.findall("([0-9]+[,.]+[0-9]+)", metadata)]
        metadata_len.append(match)
        
        if 'longboard' in file:
            sub1 = "at"
            sub2 = "("
            idx1 = read_dict['INFO0'].index(sub1)
            idx2 = read_dict['INFO0'].index(sub2)
            res = read_dict['INFO0'][idx1 + len(sub1) + 1: idx2]
            accl = [float(x) for x in re.findall("([0-9]+[,.]+[0-9]+)",res)]
            idx1 = read_dict['INFO1'].index(sub1)
            idx2 = read_dict['INFO1'].index(sub2)
            res = read_dict['INFO1'][idx1 + len(sub1) + 1: idx2]
            gyro = [float(x) for x in re.findall("([0-9]+[,.]+[0-9]+)",res)]
        else:    
            accl = [float(x) for x in re.findall("([0-9]+[,.]+[0-9]+)",read_dict['INFO0'])]
            gyro = [float(x) for x in re.findall("([0-9]+[,.]+[0-9]+)",read_dict['INFO1'])]
        freq_accl.append(accl)
        freq_gyro.append(gyro)
        
        video_name.append(file.split('_')[0])
    
    metadata_len = [item for sublist in metadata_len for item in sublist] 
    freq_accl    = [item for sublist in freq_accl for item in sublist]
    freq_gyro    = [item for sublist in freq_gyro for item in sublist]
    takeoff      = [x.split('_')[1].split('.')[0] for x in dictnames]

    combined_df = pd.DataFrame(
        {'video_id':video_name,
        'takeoff': takeoff,
        'gopro_model':gopro_models,
        'metadata_len': metadata_len,
        'freq_accl':freq_accl,
        'freq_gyro':freq_gyro        
        })
    
    return combined_df

def replacenth(string, sub, wanted, n):
    """ Function to replace a substring by another substring in nth position
    Arguments:
    string: complete string containing substring to replace
    sub: substring to replace
    wanted: substring that should replace sub
    n: occurrence of sub to be replaced (starting with 1)
    
    Output:
    string where nth occurrence of sub has been replaced by wanted
    """
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    newString = before + after
    return(newString)

def get_accgyro(path_to_json,files_list):
    """ Function to concatenate accelerometer and gyroscope data from gpfm files
    Arguments:
    path_to_json: path where the files can be found
    files_list: list of files to be read from

    Output:
    DataFrame containing 3 columns for ACCL, 3 columns for GYRO, DEVC id column, video id column and target column

    """
    output = pd.DataFrame()
    for k in range(len(files_list)):
        surfjsonfile=files_list[k]
        jsonfile = path_to_json+surfjsonfile

        with open(jsonfile, 'r') as file:
            data = file.read().replace('\n', '')

        nb_occ_devc = len([m.start() for m in re.finditer('DEVC',data)])
        nb_occ_info = len([m.start() for m in re.finditer('INFO',data)])
        #nb_occ_strm = len([m.start() for m in re.finditer('STRM',data)])


        for m in range(0,nb_occ_devc):
            data = replacenth(data, 'DEVC', 'DEVC'+str(m),m+1)

        for m in range(0,nb_occ_info):
            data = replacenth(data, 'INFO', 'INFO'+str(m),m+1)

        data1 = '' + data

        for j in range(nb_occ_devc-1):
            index_devc = [m.start() for m in re.finditer('DEVC',data1)]
            index_strm = [m.start() for m in re.finditer('STRM',data1[index_devc[j]:index_devc[j+1]])]
            for i in range(0,len(index_strm)):
                data1 = data1[:index_devc[j]] + replacenth(data1[index_devc[j]:index_devc[j+1]], 'STRM', 'STRM'+str(i), i+1) + data1[index_devc[j+1]:]

        index_devc = [m.start() for m in re.finditer('DEVC',data1)]
        index_strm_last = [m.start() for m in re.finditer('STRM',data1[index_devc[nb_occ_devc-1]:])]  

        for i in range(0,len(index_strm_last)):    
            data1 = data1[:index_devc[nb_occ_devc-1]]+replacenth(data1[index_devc[nb_occ_devc-1]:], 'STRM', 'STRM'+str(i), i+1)
        json_object = json.loads(data1)

        tot = pd.DataFrame()
        for j in range(0,nb_occ_devc-2):
            acc = json_object['DEVC'+str(j+2)]['STRM0']['ACCL']
            accone = pd.DataFrame(np.reshape(acc,(len(acc)//3,3)),columns=['Y_ACCL','-X_ACCL','Z_ACCL'])
            gyro = json_object['DEVC'+str(j+2)]['STRM1']['GYRO']
            gyroone = pd.DataFrame(np.reshape(gyro,(len(gyro)//3,3)),columns=['Y_GYRO','-X_GYRO','Z_GYRO'])
            accgyro = pd.concat((accone,gyroone),axis=1)
            accgyro['DEVC'] = 'DEVC'+str(j+2)
            tot = pd.concat((tot,accgyro),axis=0)

        tot['video_id'] = surfjsonfile.split('_')[0]
        tot['video_idx'] = k
        tot['target'] = surfjsonfile.split('_')[1].split('.')[0]

        output = pd.concat((output,tot),axis=0)
        output = output.reset_index(drop=True)
        output['target'] = output['target'].str.replace('fall','no')
        output['label'] = output.target.eq('yes').mul(1)
    return(output)
    
def saving_json_as_dict(path_to_json,files_list,newnames_list):
    """ Save json files as ditionnary to use them easily
    Arguments:
    path_to_json: path where the files can be found
    files_list: list of files to be read from
    newnames_list: list of new names for the files to be saved (example: add '_dict')
    
    Output: files are saved in path_to_json
    """
    for k in range(len(files_list)):
        surfjsonfile=files_list[k]
        jsonfile = path_to_json+surfjsonfile

        with open(jsonfile, 'r') as file:
            data = file.read().replace('\n', '')

        nb_occ_devc = len([m.start() for m in re.finditer('DEVC',data)])
        nb_occ_info = len([m.start() for m in re.finditer('INFO',data)])
        #nb_occ_strm = len([m.start() for m in re.finditer('STRM',data)])


        for m in range(0,nb_occ_devc):
            data = replacenth(data, 'DEVC', 'DEVC'+str(m),m+1)

        for m in range(0,nb_occ_info):
            data = replacenth(data, 'INFO', 'INFO'+str(m),m+1)

        if 'longboard' in surfjsonfile:
            nb_occ_note = len([m.start() for m in re.finditer('NOTE',data)])
            for m in range(0,nb_occ_note):
                data = replacenth(data, 'NOTE', 'NOTE'+str(m),m+1)

        data1 = '' + data

        for j in range(nb_occ_devc-1):
            index_devc = [m.start() for m in re.finditer('DEVC',data1)]
            index_strm = [m.start() for m in re.finditer('STRM',data1[index_devc[j]:index_devc[j+1]])]
            for i in range(0,len(index_strm)):
                data1 = data1[:index_devc[j]] + replacenth(data1[index_devc[j]:index_devc[j+1]], 'STRM', 'STRM'+str(i), i+1) + data1[index_devc[j+1]:]

        index_devc = [m.start() for m in re.finditer('DEVC',data1)]
        index_strm_last = [m.start() for m in re.finditer('STRM',data1[index_devc[nb_occ_devc-1]:])]  

        for i in range(0,len(index_strm_last)):    
            data1 = data1[:index_devc[nb_occ_devc-1]]+replacenth(data1[index_devc[nb_occ_devc-1]:], 'STRM', 'STRM'+str(i), i+1)
        json_object = json.loads(data1)

        np.save(path_to_json+newnames_list[k]+'.npy',json_object)

def create_newidx(input_dset):
    """ Create video_newidx to be used for model
    Argument:
    input_dset: input dataset with a variable called video_idx to be 'reindexed' starting at 0

    Output: the input dataset is modified with a new variable video_newidx
    """
    list_vid = list(input_dset['video_id'].unique())
    newidx = []
    for i, id in enumerate(input_dset['video_id']): 
        newidx.append(list_vid.index(id)) 
    input_dset['video_newidx'] = newidx