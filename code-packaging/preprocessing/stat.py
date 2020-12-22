import os
import numpy as np
import pandas as pd

def main(image_folder_path):
    idx=0
    df = pd.DataFrame(columns = ['id','gender','age','tex_num','face'])
    columns = list(df)
    data = []
    for filename in os.listdir(image_folder_path):
    
        if(filename.endswith(".jpg")):
                filename_without_ext = filename.split("_tex")[0]
                info = filename_without_ext.split("_")
                t_id = int(info[0])
                t_gender = info[1]
                t_age = int(info[2])
                t_tex_num = int(info[3])
                t_face = ""
                for i in info[4:]:
                    t_face += i + "_"
                values = [t_id, t_gender, t_age, t_tex_num, t_face]
                zipped = zip(columns,values)
                a_dictionary = dict(zipped)
                data.append(a_dictionary)
    df = df.append(data,True)
    #print(df.head)
    
    print(df.count())

    df_without_face = df.loc[:,'id':'age']
    basic_info = df_without_face.drop_duplicates(['id','gender','age'])
    #print(basic_info)
    #print(df.groupby(['gender']).count())
    #print(df.groupby(['id']).count())
    print("Number of face types for each id\n" + str((df.groupby('id').count())['face']))
    print("Average number of face types for each id : " + str(format((df.groupby('id').count())['face'].mean(),'.2f')))
    print("\nGroup by gender\n")
    for name, group in df.groupby('gender'):
        print(name)
        print(group.head(10))
        print("...\n\n")
        print(name + " average age: " + str(format(group['age'].mean(),'.2f'))+"\n")
    print("\n<Distinct people info>\n")
    print("Number of distinct people: " + str(basic_info.count()['id']))

    print("\nGroup by gender\n")
    print(basic_info.groupby(['gender']).count()['id'])
    print("Average age: " + str(format(basic_info['age'].mean(),'.2f'))+"\n")
    for name, group in basic_info.groupby('gender'):
        print(name)
        print(group)
        print(name + " average age: " + str(format(group['age'].mean(),'.2f'))+"\n" +
                "max age: " + str(group['age'].max()) +  " min age: " + str(group['age'].min()) + "\n")
    #print(basic_info.describe())

    


if __name__ == "__main__":
    main("../FaceScape_noempty")
