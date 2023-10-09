from sklearn.base import BaseEstimator,TransformerMixin


class commaremover(BaseEstimator,TransformerMixin):
    def __init__(self):
        pass
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        if X['Previous CTC'].dtype==object:
            X['Previous CTC']=X['Previous CTC'].str.replace(",","",regex=True).astype(float)
        else:
             pass
        return X
    

if __name__ =='__main__':
	main()   
