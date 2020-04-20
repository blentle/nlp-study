import csv
with open('E:\\后场nlp学习\\数据集\\AutoMaster_TrainSet.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)