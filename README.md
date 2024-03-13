# udemi_All_of_AI
## Artificial Neural Network <br/>

#### Importing the libraries
```python
import numpy as np  # numpy 라이브러리 불러오기
import pandas as pd  # pandas 라이브러리 불러오기
import tensorflow as tf  # tensorflow 라이브러리 불러오기
```
```python
tf.__version__    # tensorflow 버전확인
```
### Part 1 - Data Preprocessing

<details>
<summary>Importing the dataset</summary>


```python
dataset = pd.read_csv('Churn_Modelling.csv') # 데이터 셋 불러오기
X = dataset.iloc[:, 3:-1].values   # x 데이터셋 분리 (Exit 제외 feature)
y = dataset.iloc[:, -1].values   # Y 데이터셋 분리(Exit 라벨)
```
```python
dataset
```
<div>

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>RowNumber</th>
      <th>CustomerId</th>
      <th>Surname</th>
      <th>CreditScore</th>
      <th>Geography</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Tenure</th>
      <th>Balance</th>
      <th>NumOfProducts</th>
      <th>HasCrCard</th>
      <th>IsActiveMember</th>
      <th>EstimatedSalary</th>
      <th>Exited</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>15634602</td>
      <td>Hargrave</td>
      <td>619</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>2</td>
      <td>0.00</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101348.88</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>15647311</td>
      <td>Hill</td>
      <td>608</td>
      <td>Spain</td>
      <td>Female</td>
      <td>41</td>
      <td>1</td>
      <td>83807.86</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>112542.58</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>15619304</td>
      <td>Onio</td>
      <td>502</td>
      <td>France</td>
      <td>Female</td>
      <td>42</td>
      <td>8</td>
      <td>159660.80</td>
      <td>3</td>
      <td>1</td>
      <td>0</td>
      <td>113931.57</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>15701354</td>
      <td>Boni</td>
      <td>699</td>
      <td>France</td>
      <td>Female</td>
      <td>39</td>
      <td>1</td>
      <td>0.00</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>93826.63</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>15737888</td>
      <td>Mitchell</td>
      <td>850</td>
      <td>Spain</td>
      <td>Female</td>
      <td>43</td>
      <td>2</td>
      <td>125510.82</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>79084.10</td>
      <td>0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>9995</th>
      <td>9996</td>
      <td>15606229</td>
      <td>Obijiaku</td>
      <td>771</td>
      <td>France</td>
      <td>Male</td>
      <td>39</td>
      <td>5</td>
      <td>0.00</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>96270.64</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9996</th>
      <td>9997</td>
      <td>15569892</td>
      <td>Johnstone</td>
      <td>516</td>
      <td>France</td>
      <td>Male</td>
      <td>35</td>
      <td>10</td>
      <td>57369.61</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>101699.77</td>
      <td>0</td>
    </tr>
    <tr>
      <th>9997</th>
      <td>9998</td>
      <td>15584532</td>
      <td>Liu</td>
      <td>709</td>
      <td>France</td>
      <td>Female</td>
      <td>36</td>
      <td>7</td>
      <td>0.00</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>42085.58</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9998</th>
      <td>9999</td>
      <td>15682355</td>
      <td>Sabbatini</td>
      <td>772</td>
      <td>Germany</td>
      <td>Male</td>
      <td>42</td>
      <td>3</td>
      <td>75075.31</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>92888.52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9999</th>
      <td>10000</td>
      <td>15628319</td>
      <td>Walker</td>
      <td>792</td>
      <td>France</td>
      <td>Female</td>
      <td>28</td>
      <td>4</td>
      <td>130142.79</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>38190.78</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>10000 rows × 14 columns</p>
</div>


```python
print(X)   # x변수에 들어간 Dataframe 확인
```

[[619 'France' 'Female' ... 1 1 101348.88]
 [608 'Spain' 'Female' ... 0 1 112542.58]
 [502 'France' 'Female' ... 1 0 113931.57]
 ...
 [709 'France' 'Female' ... 0 1 42085.58]
 [772 'Germany' 'Male' ... 1 0 92888.52]
 [792 'France' 'Female' ... 1 0 38190.78]]



```python
print(y)   # x변수에 들어간 list 확인
```
[1 0 1 ... 1 1 0]


</details>

<details>
<summary>Encoding categorical data</summary>
</br>
  
Label Encoding the "Gender" column
```python
from sklearn.preprocessing import LabelEncoder   # Label Encoder 불러오기
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:,2])
# 3번째 열(gender)를 Label Encoding하여 0과 1로 바꿈
```
One Hot Encording the"Geography" column
```python
from sklearn.compose import ColumnTransformer    
from sklearn.preprocessing import OneHotEncoder
#columnTransformer와 OneHotEncoder를 통해 나라별로 카테고리화 시키기 위한 라이브러리 호출
```
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
# X에서 2번째 컬럼인 Geography(나라)컬럼을 나누어서 One hot Encoding
# reminder = 'passthrough'는 변환하는 컬럼외 나머지는 그대로 남김

X = np.array(ct.fit_transform(X))
# ColumnTransformer를 fit_transform 메서드를 통해 X변수에 적용하여 numpy_array화 시킴
```
```python
print(X)
```
[[1.0 0.0 0.0 ... 1 1 101348.88]
 [0.0 0.0 1.0 ... 0 1 112542.58]
 [1.0 0.0 0.0 ... 1 0 113931.57]
 ...
 [1.0 0.0 0.0 ... 0 1 42085.58]
 [0.0 1.0 0.0 ... 1 0 92888.52]
 [1.0 0.0 0.0 ... 1 0 38190.78]]



</details>

<details>
<summary>Splitting the dataset into the Training set and Test set</summary>
</br>

```python
from sklearn.model_selection import train_test_split
# Train Test 데이터 셋을 나누기 위한 라이브러리 호출
```
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
# Train Test 데이터 셋 나누기 test_size = 0.2(Train 80%, Test 20%)
```
</details>


### Part 2 - Building the ANN
<details>
<summary>Initializing the ANN</summary>
</br>
  
```python
ann = tf.keras.models.Sequential() #ANN 초기화
```  
</details>

<details>
<summary>Adding the input layer and the first hidden layer</summary>
</br>
뉴런의 갯수를 정하는 방법 - 경험 관련있는 숫자, 과하지 않은 숫자

```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```

</details>

<details>
<summary>Adding the second hidden layer</summary>

```python
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
```
</details>

<details>
<summary>Adding the output layer</summary>
</br>

unit 의 갯수를 정하는 것은 출력 값에 따라 결정 된다 0과 1로 표현되는 값을 예측하는 것이므로 units = 1</br>
2개 이상의 카테고리를 예측하는 경우에는 activation = 'softmax'로 설정해야 함

```python
ann.add(tf.keras.layers.Dense(units= 1  , activation= 'sigmoid'))
```

</details>

### Part 3 - Training the ANN

<details>
<summary>Compiling the ANN</summary>
</br>
  
```python
ann.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])
```
</details>

<details>
<summary>Training the ANN on the Training set</summary>
</br>
batch_size = 32가 default

```python
ann.fit(X_train, y_train, batch_size= 32, epochs = 100)
```

Epoch 1/100</br>
250/250 [==============================] - 2s 2ms/step - loss: 0.5045 - accuracy: 0.7960</br>
Epoch 2/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4548 - accuracy: 0.7960</br>
Epoch 3/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4344 - accuracy: 0.7960</br>
Epoch 4/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4258 - accuracy: 0.7960</br>
Epoch 5/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4214 - accuracy: 0.7960</br>
Epoch 6/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4169 - accuracy: 0.7960</br>
Epoch 7/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4125 - accuracy: 0.8095</br>
Epoch 8/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4078 - accuracy: 0.8259</br>
Epoch 9/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.4034 - accuracy: 0.8278</br>
Epoch 10/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.3995 - accuracy: 0.8305</br>
Epoch 11/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.3953 - accuracy: 0.8314</br>
Epoch 12/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.3909 - accuracy: 0.8361</br>
Epoch 13/100</br>
...</br>
Epoch 99/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.3346 - accuracy: 0.8619</br>
Epoch 100/100</br>
250/250 [==============================] - 1s 3ms/step - loss: 0.3347 - accuracy: 0.8615</br>

</details>

### Part 4 - Making the predictions and evaluating the model
</br>

<details>
<summary>Predicting the result of a single observation</summary>



Use our ANN model to predict if the customer with the following informations will leave the bank: 
</br>
Geography: France (1, 0, 0)</br>
</br>
Credit Score: 600</br>
</br>
Gender: Male(1)</br>
</br>
Age: 40 years old</br>
</br>
Tenure: 3 years</br>
</br>
Balance: \$ 60000</br>
</br>
Number of Products: 2</br>
</br>
Does this customer have a credit card ? Yes (1)</br>
</br>
Is this customer an Active Member: Yes (1)</br>
</br>
Estimated Salary: \$ 50000</br>
</br>
So, should we say goodbye to that customer ?</br>

Solution

```python
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```
Therefore, our ANN model predicts that this customer stays in the bank!

**Important note 1:** Notice that the values of the features were all input in a double pair of square brackets. That's because the "predict" method always expects a 2D array as the format of its inputs. And putting our values into a double pair of square brackets makes the input exactly a 2D array.</br>
특성 값은 모두 이중 대괄호 쌍에 입력되었습니다. 이는 "예측" 메서드가 항상 입력 형식으로 2D 배열을 기대하기 때문입니다. 그리고 값을 이중 대괄호 안에 넣으면 입력이 정확히 2D 배열이 됩니다.</br>

**Important note 2:** Notice also that the "France" country was not input as a string in the last column but as "1, 0, 0" in the first three columns. That's because of course the predict method expects the one-hot-encoded values of the state, and as we see in the first row of the matrix of features X, "France" was encoded as "1, 0, 0". And be careful to include these values in the first three columns, because the dummy variables are always created in the first columns.</br>
또한 "프랑스" 국가는 마지막 열에 문자열로 입력되지 않고 처음 세 열에 "1, 0, 0"으로 입력되었습니다. 이는 물론 예측 메서드가 원-핫 인코딩된 상태 값을 예상하고 특성 X 행렬의 첫 번째 행에서 볼 수 있듯이 "프랑스"가 "1, 0, 0"으로 인코딩되었기 때문입니다. 더미 변수는 항상 첫 번째 열에 생성되므로 처음 세 열에 이러한 값을 포함하도록 주의하세요.

</details>

<details>
<summary>Predicting the Test set results</summary>

```python
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```
</details>

