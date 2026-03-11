import pandas as pd

class HeartDataLoader:
    def __init__(self, train_path: str, test_path: str):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self):
        train = pd.read_csv(self.train_path)
        test = pd.read_csv(self.test_path)
        
        # Сразу дропаем бесполезную колонку
        if 'Unnamed: 0' in train.columns:
            train = train.drop(columns=['Unnamed: 0'])
        if 'Unnamed: 0' in test.columns:
            test = test.drop(columns=['Unnamed: 0'])
            
        return train, test