import re
import numpy as np


class RecursivePredictor():

    def __init__(self, header, model, label_codes_dict, y):
        """

        Parameters
        ----------
        header : list
            Список заголовков колонок
        model : Optional
            Модель для предсказания с методом predict
        label_codes_dict : dict
            Словарь кодирования категориальных признаков: {'column_name': {'label': code}}
        """     
        self.header = header
        self.model = model
        self.label_codes_dict = label_codes_dict
        self.y = y
		
    def get_recursive_prediction(self, X_1, num_col_for_rec=7, day_range=28, start=1914):
        """
        
        Итеративное получение предсказаний по дням

        Parameters
        ----------
        X : numpy ndarray
            Матрица признаков (кол-во элементов, кол-во признаков)
        day_range : int, optional
            Количество дней, на которое надо делать прогноз, by default 28

        Returns
        -------
        numpy ndarray
            Матрица предсказаний (кол-во элементов, )
        """ 
        ind_day = self.header.index('day_as_int')
        X = X_1.copy()
        predictions = np.zeros(X.shape[0])
        for day in range(day_range):
            mask = X[:, ind_day] == start + day
            for i in range(1, num_col_for_rec+1):
                str_col = self.header[-i].split('_')
                
                if len(str_col) == 2:
                    if np.sum(self.y[:,1] == start + day - int(str_col[1])) == 0:
                        X[:, -i][mask] = 0
                    else:
                        X[:, -i][mask] = self.y[:,0][self.y[:,1] == start + day - int(str_col[1])]
                    
                    
                elif len(str_col) == 3:
                    if day == 0:
                        continue
                    if float(str_col[2]) >= 1:
                        if np.sum(self.y[:,1] == start + day - int(str_col[1])- int(str_col[2])) == 0:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - 1] + self.y[:,0][self.y[:,1] == start + day - int(str_col[1])] * 1/int(str_col[1]) 
                        else:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - 1] + (
                                self.y[:,0][self.y[:,1] == start + day - int(str_col[1])] - self.y[:,0][self.y[:,1] == start + day - int(str_col[1])- int(str_col[2])]) * 1/int(str_col[1]) 
                        
                    else:
                        if np.sum(self.y[:,1] == start + day - int(str_col[1])) == 0:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - 1] * beta
                        else:
                            beta = float(str_col[2])
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - 1] * beta + self.y[:,0][self.y[:,1] == start + day - int(str_col[1])] *  (1 - beta) 
                        
                        
                        
                elif len(str_col) == 5:
                    period = int(str_col[4])
                    if day - period < 0:
                        continue
                    if float(str_col[2]) >= 1:
                        if np.sum(self.y[:,1] == start + day - int(str_col[1])- int(str_col[2]) * period) == 0:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - period] + self.y[:,0][self.y[:,1] == start + day - int(str_col[1])] * 1/int(str_col[1])
                        else:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - period] + (
                              self.y[:,0][self.y[:,1] == start + day - int(str_col[1])] - self.y[:,0][self.y[:,1] == start + day - int(str_col[1])- int(str_col[2]) * period]) * 1/int(str_col[1])
                        
                    else:
                        beta = float(str_col[2])
                        if np.sum(self.y[:,1] == start + day - int(str_col[1]) * period) == 0:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - period]*beta
                        else:
                            X[:, -i][mask] = X[:, -i][X[:, ind_day] == start + day - period]*beta + self.y[:,0][self.y[:,1] == start + day - int(str_col[1]) * period] * (1 - beta) 
                        
                        
            pred = self.model.predict(X[mask])
            self.y[:,0][self.y[:, 1] == start + day] = pred
            predictions[mask] = pred
        return predictions

    def fill_final_submission(self, sample_submission, X, predictions=None, day_range=28, start=1914):
        """
        
        Заполнение шаблона посылки

        Parameters
        ----------
        sample_submission : DataFrame
            шаблон посылки для соревнования
        X : numpy ndarray
            Матрица признаков (кол-во элементов, кол-во признаков)
        predictions : numpy ndarray, optional
            Массив предсказаний, если он получен заранее, by default None
        day_range : int, optional
            Количество дней, на которое надо делать прогноз, by default 28
        """        
        if predictions is not None:
            predictions = self.get_recursive_prediction(X, start=start)
        
        index_to_item_id = {value: key for key, value in self.label_codes_dict['item_id'].items()}
        X_test_item_ids = [index_to_item_id[elem] for elem in X[:, self.header.index('item_id')]]
        index_to_store_id = {value: key for key, value in self.label_codes_dict['store_id'].items()}
        X_test_store_ids = [index_to_store_id[elem] for elem in X[:, self.header.index('store_id')]]
        X_test_submission_index = [
            f'{item_id}_{store_id}_validation'
            for item_id, store_id in zip(X_test_item_ids, X_test_store_ids)
        ]
        predictions = predictions.reshape(30490, day_range)
        sample_submission.iloc[:30490,1:] = predictions
        sample_submission.to_csv('submission.csv',index=False)