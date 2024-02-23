import pandas as pd

import pathlib



def generateReport(patID):
    print(patID)
    df = pd.read_csv('out_csv.csv', names=['PatientID', 'Sex', 'Date', 'Eye Part', 'Labels'])

    report_fields = ["PatientID", "Sex", "Date", "Eye Part", "Labels"]
    if len(df):
        exam = df.loc[df['PatientID'] == patID]
        print(exam.columns)

        report_data = exam.iloc[0]
        print(report_data)
        report_dict = {}
        for field in range(0, len(report_fields)):
            report_dict[report_fields[field]] = str(report_data[field])

        image_path = pathlib.Path(f"image Database/{patID}.png")
        report_dict['image'] = str(image_path)
        print("Dictionary")
        print(report_dict)
        return report_dict




