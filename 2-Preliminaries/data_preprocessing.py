import os
import pandas as pd
import torch

'''https://d2l.ai/chapter_preliminaries/pandas.html'''

if  __name__ == '__main__':

    '''Reading the Dataset'''

    os.makedirs(os.path.join('..', 'data'), exist_ok=True)
    data_file = os.path.join('..', 'data', 'house_tiny.csv')
    with open(data_file, 'w') as f:
        f.write('''NumRooms,RoofType,Price\nNA,NA,127500\n2,NA,106000\n4,Slate,178100\nNA,NA,140000''')

    data = pd.read_csv(data_file)
    print(data)

    '''Data Preparation'''

    inputs, targets = data.iloc[:, :2], data.iloc[:, 2]
    inputs = pd.get_dummies(inputs, dummy_na=True)
    print(inputs)

    inputs = inputs.fillna(inputs.mean())
    print(inputs)


    '''Conversion to the Tensor Format'''
    X, y = torch.tensor(inputs.values), torch.tensor(targets.values)
    print(X, y)





