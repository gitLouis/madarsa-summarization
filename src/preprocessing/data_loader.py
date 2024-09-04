from pathlib import Path
import pandas as pd

def basic_data_loader(raw_data_path: Path):
    # prepare input/output folder paths


    filepaths = {'course_start':raw_data_path/'ערבית מדוברת: מתחילים - שאלון פתיחת קורס.csv',
                'mid_course': raw_data_path/'ערבית מדוברת: מתחילים - שאלון במהלך הקורס.csv',
                'course_end': raw_data_path/' "ערבית מדוברת: מתחילים" - שאלון סיום קורס.csv'
                }
    
    text_column_index = {'course_start':[-1,-2],
                        'mid_course': [4,5,-1],
                        'course_end': [5,6,-2]
                        }

    text_columns_names = dict()
    dfs = dict()


    for qstr_key, file_path in filepaths.items():
        df = pd.read_csv(file_path)
        text_cols = text_column_index[qstr_key]
        text_columns_names[qstr_key] = list(df.columns[text_cols])
        dfs[qstr_key] = df

    return dfs, text_columns_names