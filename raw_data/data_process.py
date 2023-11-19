# coding:utf-8
'''
**************************************************
@File   ：LSTM -> data_process
@IDE    ：PyCharm
@Author ：Tianze Zhang
@Desc   , extract raw xml data to construct the sentence template
@Date   ：2023/11/18 0:43
**************************************************
'''
import xml.dom.minidom
import csv
import os
import pandas as pd


def get_file_name(file_path):
    file_names = os.listdir(file_path)
    return file_names
def read_xml(xml_path, data):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    backstories = root.getElementsByTagName('BackstoryDef')
    for backstory in backstories:
        title = backstory.getElementsByTagName('title')[0].childNodes[0].data
        name = backstory.getElementsByTagName('defName')[0].childNodes[0].data
        desc = backstory.getElementsByTagName('baseDesc')[0].childNodes[0].data
        attributes = backstory.getElementsByTagName('skillGains')[0].getElementsByTagName('li')
        print(title)
        print(name)
        print(desc)
        attribute_str = ''
        for attribute in attributes:
            attribute_name = attribute.getElementsByTagName('key')[0].childNodes[0].data
            attribute_gain = attribute.getElementsByTagName('value')[0].childNodes[0].data
            attribute_gain = int(attribute_gain)
            if attribute_gain > 0:
                attribute_desc = attribute_name+'+'+str(attribute_gain)
            else:
                attribute_desc = attribute_name+str(attribute_gain)
            attribute_str = attribute_str + attribute_desc + '\t'
            # print(attribute_desc)
        print(attribute_str)
        data_ls = [title, name, desc, attribute_str]
        data.append(data_ls)
    return data

if __name__ == '__main__':
    csv_head = ['Title', 'Name', 'Desc', 'Attribute']
    data = []
    file_path = './basegame'
    file_names = get_file_name(file_path)
    for file_name in file_names:
        xml_path = 'basegame/'+file_name
        read_xml(xml_path, data)
    print(len(data))
    df = pd.DataFrame(data)
    df.columns = csv_head
    ########### Save as cpkl ##################
    df.to_pickle('backstory.pkl')

    ########### Save as csv ##################
    # with open('backstory.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(csv_head)
    #     writer.writerows(data)
