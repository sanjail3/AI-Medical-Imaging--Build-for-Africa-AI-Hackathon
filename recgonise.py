import csv
from tensorflow.keras.models import load_model
import pandas as pd
import pathlib
import numpy as np
from tensorflow.keras.preprocessing import image



img_height = 224
img_width = 224
batch_size = 32

df = pd.read_csv('patient_record.csv')
print(type(df.loc[df['PatientID'] == "000c1434d8d7"].loc[0]))

def append_to_patient_record_csv( id, sex, selected_date, eye_part):
    with open("patient_record.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([id, sex, selected_date, eye_part])


def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(img_height, img_width))
    img = image.img_to_array(img)
    return img



def get_result():
    model = load_model('best_model.h5')
    df = pd.read_csv('patient_record.csv',index_col=False)
    id_code = df['PatientID'].iloc[:]

    img_height = 224
    img_width = 224
    batch_size = 32
    predicted_labels=[]
    for i in id_code:
        image_path = pathlib.Path(f"image Database/{i}.png")
        preprocessed_image = preprocess_image(str(image_path))
        # resized_image = cv2.resize(preprocessed_image, (img_width, img_height))

        prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
        predicted_class = np.argmax(prediction, axis=1)
        class_labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR',
                        'Severe']  # Replace with your actual class labels
        predicted_label = class_labels[predicted_class[0]]
        print("Predicted Class:", predicted_class)
        print("Predicted Label:", predicted_label)
        predicted_labels.append(predicted_label)
    df['Labels']=predicted_labels
    df.to_csv('out_csv.csv',index=False)
    return predicted_label




def get_result_for_single_image(patientid):
    model = load_model('best_model.h5')
    img_height = 224
    img_width = 224
    batch_size = 32
    predicted_labels = []
    image_path = pathlib.Path(f"image Database/{patientid}.png")
    preprocessed_image = preprocess_image(str(image_path))
    # Perform operations on the preprocessed image
    # Example: Pass the preprocessed image to the model for prediction
    prediction = model.predict(np.expand_dims(preprocessed_image, axis=0))
    predicted_class = np.argmax(prediction, axis=1)
    class_labels = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR',
                    'Severe']  # Replace with your actual class labels
    predicted_label = class_labels[predicted_class[0]]
    print("Predicted Class:", predicted_class)
    print("Predicted Label:", predicted_label)
    df = pd.read_csv('patient_record.csv')
    row_list= list(df.loc[df['PatientID'] == patientid].iloc[0])
    print(row_list)
    print(row_list)
    row_list.append(predicted_label)
    df_new=pd.read_csv('out_csv.csv')
    print(row_list)
    print(df_new.columns)

    with open("out_csv.csv", "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(row_list)












