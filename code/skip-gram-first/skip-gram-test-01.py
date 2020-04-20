import csv
with open('E:\\nlp-dataset\\AutoMaster_TrainSet.csv', 'r', encoding='utf-8') as f:
    reader = csv.reader(f)
    for row in reader:
        print(row)