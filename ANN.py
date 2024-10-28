# import files
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the Iris_dataset
iris = load_iris()
x = iris.data
y= (iris.target == 0).astype(int) #Binary Classification :Setosa (class 0) vs. others

# Split the data into traning and test
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

# Standrardize the Feature
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)

#Create a Sequentential model
model = Sequential()

# Add the input layer and the first hidden layer
model.add(Dense(units=6,activation='relu',input_dim=4)) # 4 features in the Iris Dataset

# Add another hidden layer
model.add(Dense(units=6,activation='relu')) 

#Add the output layer with sigmoid activation for binary classification
model.add(Dense(units=1,activation='sigmoid')) 

#Compile the Model
model.compile(optimizer=Adam(learning_rate=0.001),loss='binary_crossentropy',metrics=['accuracy'])

#Display Model Summary
model.summary()

#Train the model
model.fit(x_train,y_train,epochs=12,batch_size=8,validation_split=0.2)

ypred = model.predict(x_test)
ypred = (ypred > 0.5)

#Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test,y_test)
print(f'Test Loss: {loss:.4f} Test Accuracy:{accuracy:.4f}')