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


def get_filefolder_name(filefolder_path):
    filefolder_names = os.listdir(filefolder_path)
    return filefolder_names


def read_xml(xml_path, data, story_head):
    dom = xml.dom.minidom.parse(xml_path)
    root = dom.documentElement
    for head in story_head:
        if root.getElementsByTagName(head):
            backstories = root.getElementsByTagName(head)
            for backstory in backstories:
                if backstory.getElementsByTagName('title'):
                    title = backstory.getElementsByTagName('title')[0].childNodes[0].data
                else:
                    continue
                if backstory.getElementsByTagName('defName'):
                    name = backstory.getElementsByTagName('defName')[0].childNodes[0].data
                else:
                    continue
                if backstory.getElementsByTagName('baseDesc'):
                    desc = backstory.getElementsByTagName('baseDesc')[0].childNodes[0].data
                elif backstory.getElementsByTagName('baseDescription'):
                    desc = backstory.getElementsByTagName('baseDescription')[0].childNodes[0].data
                else:
                    continue
                if backstory.getElementsByTagName('skillGains'):
                    attributes = backstory.getElementsByTagName('skillGains')[0].getElementsByTagName('li')
                else:
                    continue
                # print(title)
                # print(name)
                # print(desc)
                attribute_str = ''
                for attribute in attributes:
                    if attribute.getElementsByTagName('key'):
                        attribute_name = attribute.getElementsByTagName('key')[0].childNodes[0].data
                    else:
                        continue
                    if attribute.getElementsByTagName('value'):
                        attribute_gain = attribute.getElementsByTagName('value')[0].childNodes[0].data
                        attribute_gain = int(attribute_gain)
                        if attribute_gain > 0:
                            attribute_desc = attribute_name + '+' + str(attribute_gain)
                        else:
                            attribute_desc = attribute_name + str(attribute_gain)
                        attribute_str = attribute_str + attribute_desc + '\t'
                    else:
                        continue
                    # print(attribute_desc)
                # print(attribute_str)
                data_ls = [title, name, desc, attribute_str]
                data.append(data_ls)
            return data


if __name__ == '__main__':
    csv_head = ['Title', 'Name', 'Desc', 'Attribute']
    story_head = ['BackstoryDef', 'StoriesRetold.SRBackstoryDef', 'AlienRace.BackstoryDef', 'AlienRace.AlienBackstoryDef', 'ZCB.ZCBackstoryDef', 'CommunityCoreLibrary.BackstoryDef']
    data = []
    file_path = './story'
    fds_name = get_filefolder_name(file_path)
    for fd_name in fds_name:
        print(fd_name)
        file_names = get_file_name(file_path + '/' + fd_name)
        for file_name in file_names:
            # print(file_name)
            xml_path = 'story/' + fd_name + '/' + file_name
            read_xml(xml_path, data, story_head)
            print(len(data))
    df = pd.DataFrame(data)
    df.columns = csv_head
    ########### Save as cpkl ##################
    df.to_pickle('backstory_large.pkl')

    ########### Save as csv ##################
    # with open('backstory.csv', 'w') as file:
    #     writer = csv.writer(file)
    #     writer.writerow(csv_head)
    #     writer.writerows(data)
