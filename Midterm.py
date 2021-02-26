import pandas as pd
import numpy as np
import pandas as pd 

#pandas kullanarak excelde daha önceden transpose ettiğim verilerimi okuttum:
datas = pd.read_excel('otu.xlsx','Sheet1')

#Bağımsız niteliği X, Bağımlıyı ise y değişkenine atadım,
#Eğitim ve test setlerini ayırdım.%20 test için ayrılırken, %80lik kısmını ise veri için ayırdım:
from sklearn.model_selection import train_test_split
datas_train, datas_test= train_test_split(datas,test_size=0.20)

X_train=datas_train.iloc[:,1:] 
X_test=datas_test.iloc[:,1:]
y_train=datas_train.iloc[:,0] 
y_test=datas_test.iloc[:,0]

#Verileri normalize/standardize edelim:
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)

X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

#Model için k en yakın komşu algoritmasını kullandım ve bununla alakalı scikit-learn sınıfını kullandım,
#Ayrıca parametre de verdim :
from sklearn.neighbors import KNeighborsClassifier
enyakin_komsu = KNeighborsClassifier(n_neighbors=5)

#Modeli eğittim:
enyakin_komsu.fit(X_train, y_train)

#Test seti ile hedef sınıfları tahmin ettim:
y_pred=enyakin_komsu.predict(X_test)

#Hata matrisi(confusion matrix) ile modelimin başarısını ölçtüm,
#Tahmin sonuçlarımı test sonuçları ile karşılaştırdım:
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
    
import numpy as np
#sınıftaki her bir farkli eleman icin (left/right) hesaplanir:
for j in np.unique(datas.iloc[:,0]):
  #o sınıf değeri için gözlem sayısı:
  summ=len([i for i in range(0,len(y_test)) if y_test.iloc[i]==j])
  # o sınıf değeri için yapılan başarılı tahmin sayısı:
  pred_success=len([i for i in range(0,len(y_test)) if y_test.iloc[i]==j and y_test.iloc[i]==y_pred[i]])
  if j=="left":
    # eğer ki sınıfı "left" ise ([birinci sınıf başarılı tahmin sayısı]/[birinci sınıf toplam gözlem sayısı]) sensitivity olarak sonuçlar yazilir: 
    print("sensitivity: "+str(round(pred_success*100/summ))+"%.")
  else:
    # eğer ki sınıf "right" ise ([ikinci sınıf başarılı tahmin sayısı]/[ikinci sınıf toplam gözlem sayısı])  specifity olarak sonuçlar yazılır:
    print("specifity: "+str(round(pred_success*100/summ))+"%.")   
    
    
    
    
    
    
    
    
    
    