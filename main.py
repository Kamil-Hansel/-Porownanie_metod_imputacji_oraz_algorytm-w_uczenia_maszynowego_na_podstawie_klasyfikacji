import numpy as np
#!pip install ucimlrepo
from ucimlrepo import fetch_ucirepo
from sklearn import metrics
import time

def sprawdzenie_int(x,wektor_pominiec): #Funkcja sprawdzająca czy zimputowane dane są liczbami całkowitymi w kolumnach zawierających dane kategoryczne
    for i in range(X.shape[1]):
      if i not in wektor_pominiec:
        are_integers = np.all(x[:,i] == np.floor(x[:,i]))
        print(i,are_integers)



secondary_mushroom = fetch_ucirepo(id=848) #Pobranie danych

X = secondary_mushroom.data.features #Podział danych na klasy
y = secondary_mushroom.data.targets

print(secondary_mushroom.variables)#Drukowanie informacji dotyczących zmiennych

original_columns = X.columns #Zapisanie oryginalnych kolumn

from sklearn import preprocessing
le = preprocessing.OrdinalEncoder()

#Kodowanie danych
for column in X.columns:
    if X[column].dtype == type(object):
        X.loc[:, column] = le.fit_transform(X[[column]])
        print("kolumna:",column,"Mapowanie:", dict(zip(le.categories_[0], range(len(le.categories_[0]))))) #Drukowanie przemapowanych danych

le.fit(y)
y = le.transform(y) #Kodowanie danych
print("kolumna y:","Mapowanie:", dict(zip(le.categories_[0], range(len(le.categories_[0]))))) #Drukowanie przemapowanych danych

#Zamiania formatu danych
X=X.astype(float)
y=y.astype(float)

#Obliczanie współczynników pustych wartości dla każdej kolumny oraz odrzucanie tych, w których dany współczynnik jest wyższy od 0.5
nan_counts = X.isnull().sum()
nan_ratios = nan_counts / len(X)
print(nan_ratios)
prog=0.5
X = X.loc[:, nan_ratios <= prog]
print("Kolumny o wartości procentowej pustych kolumn wyzszej od",prog,"to: \n",nan_ratios[nan_ratios>prog])
remaining_columns = X.columns
omitted_column_indices = []
for column_name in list(set(original_columns) - set(remaining_columns)):
    if column_name in original_columns:
        omitted_column_indices.append(original_columns.get_loc(column_name))

print(omitted_column_indices)

#Statyczne wprowadzanie kolumn które będą pominięte w funkcji sprawdzenie_int()
wektor_pominiec=[14, 11, 10, 17, 13, 0, 8, 9]

#Pętla nieskończona pozwalająca na wybranie dowolnego algorytmu imputacji
while 1<3:
    print("1. Imputacja moda")
    print("2. Imputacja mediana")
    print("3. Imputacja IterativeImputer")
    print("4. Imputacja KNN")
    print("5. Brak imputacji")
    tryb=int(input("podaj tryb:"))
    # Imputacja moda
    if tryb==1:
        start_time = time.time()

        #Zastosowanie imputacji
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
        imp.fit(X)
        X = imp.transform(X)

        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Imputacja mediana
    elif tryb==2:
        start_time = time.time()

        # Zastosowanie imputacji
        from sklearn.impute import SimpleImputer
        imp = SimpleImputer(missing_values=np.nan, strategy='median')
        imp.fit(X)
        X = imp.transform(X)

        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Imputacja IterativeImputer
    elif tryb==3:
        start_time = time.time()

        # Zastosowanie imputacji
        from sklearn.experimental import enable_iterative_imputer
        from sklearn.impute import IterativeImputer
        imp = IterativeImputer(max_iter=10, random_state=0)
        imp.fit(X)
        X = imp.transform(X)

        #Sprawdzanie czy wartości wprowadzone przez IterativeImputer są liczbami całkowitymi. Jeżeli nie, liczby te zostają zaokrąglone
        sprawdzenie_int(X,wektor_pominiec)
        print("zaokrąglanie wartości")
        for i in range(X.shape[1]):
                X[:,i]=np.ndarray.round(X[:,i])
        sprawdzenie_int(X,wektor_pominiec)
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego

        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Imputacja KNN
    elif tryb==4:
        start_time = time.time()

        # Zastosowanie imputacji
        from sklearn.impute import KNNImputer
        imputer = KNNImputer(n_neighbors=1)
        imputer.fit(X)
        X = imputer.transform(X)

        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Brak imputacji
    elif tryb==5:
        break
    else:
        print("wybrano niewłaściwy tryb")


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #Podział danych na zbiór testowy oraz treningowy

#Drukowanie rozmiarów ramek danych
print("Training set shape: ", X_train.shape, y_train.shape)
print("Testing set shape: ", X_test.shape, y_test.shape)

#Pętla nieskończona pozwalająca na wybranie dowolnego modelu ML
while 1<3:
    print("1.Drzewo decyzyjne")
    print("2.Regresja logistyczna")
    print("3.Naiwny klasyfikator bayesowski")
    print("4.SVM")
    print("5.HistGradientBoostingClassifier")
    tryb=int(input("podaj tryb:"))
    #Drzewo decyzyjne
    if tryb==1:
        start_time = time.time()

        #Dopasowanie modelu oraz przeprowadzanie predykcji
        from sklearn import tree
        clf = tree.DecisionTreeClassifier(max_depth=10)
        clf = clf.fit(X_train, y_train.ravel())
        y_pred = clf.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Drukowanie dokładności modelu
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Regresja logistyczna
    elif tryb==2:
        start_time = time.time()

        # Dopasowanie modelu oraz przeprowadzanie predykcji
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(solver='liblinear', C=0.5)
        model = model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Drukowanie dokładności modelu
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #Naiwny klasyfikator bayesowski
    elif tryb==3:
        start_time = time.time()

        # Dopasowanie modelu oraz przeprowadzanie predykcji
        from sklearn.naive_bayes import GaussianNB
        gnb = GaussianNB()
        y_pred = gnb.fit(X_train, y_train.ravel()).predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Drukowanie dokładności modelu
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #SVM
    elif tryb==4:
        start_time = time.time()

        # Dopasowanie modelu oraz przeprowadzanie predykcji
        from sklearn import svm
        h1=svm.LinearSVC(C=1)
        h1.fit(X_train,y_train.ravel())
        y_pred=h1.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Drukowanie dokładności modelu
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    #HistGradientBoostingClassifier
    elif tryb==5:
        start_time = time.time()

        # Dopasowanie modelu oraz przeprowadzanie predykcji
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(max_bins=255, max_iter=100)
        model.fit(X_train, y_train.ravel())
        y_pred = model.predict(X_test)

        print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) #Drukowanie dokładności modelu
        execution_time = round((time.time() - start_time), 3) #Mierzenie czasu obliczeniowego
        print("czas obliczeniowy wynosi %s sekund" % round((time.time() - start_time), 3))
        break
    else:print("wybrano niewłaściwy tryb")
