#Allo Allo

from sklearn_pandas 
import CategoricalImputer
data = np.array(['a', 'b', 'b', np.nan], dtype=object)
imputer = CategoricalImputer()
imputer.fit_transform(data)
array(['a', 'b', 'b', 'b'], dtype=object)
