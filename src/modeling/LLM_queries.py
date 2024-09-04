import ollama 
from textwrap import dedent
import math
import re


# models = ['llama3.1:latest', 'aminadaven/dictalm2.0-instruct:f16']
def summarize(col_name: str, sentences: str, model='aminadaven/dictalm2.0-instruct:f16', verbose=0): # model in 'llama3'
    
    # system_instructions = ' '.join([
    #     f"המטרה שלך היא לתרגם לעברית ואז לסכם במדויק בעברית.",
    #     f"מצורפות תשובות התלמידים לגבי השאלה הבאה [<'{col_name.strip()}'>].",
    # f"סכם את התשובות בפסקה אחת בדיוק עם לכל היותר {max(math.ceil(math.log(len(sentences))),2)} משפטים.",
    # "הקפד לוודא שהתשובה מבוססת רק על הדעות שניתנו.",
    # "מבחינה דקדוקית, נסח את הסיכום בגוף ראשון יחיד, כאילו אתה אחד הסטודנטים.",
    # "כתוב את הסיכום בעברית בלבד, ללא תוספות לפני או אחרי הסיכום",
    # "בתשובתך אל תוסיף על הקלט ואל תגיב ואל תביע את דעתך. רק סכם בהתבסס על הקלט מהסטודנטים בלבד."
    # ])
    # system_instructions = dedent(f"""\
    #                 You are a student in an arabic language course. 
    #                 Your task is return concise summary of the opinion of the other students.
    #                 The students are answering the question: [<'{col_name.strip()}'>]
    #                 You are given list of students' opinions, separated by |, Summarize their opinions. 
    #                 Format your response with up to {min(max(math.ceil(math.log(len(sentences))),2), 5)} short sentences, phrased like a student's opinion.
    #                 Your response must be in English.
    #                 Do not answer the students, do not add your opinion, do not add comments before or after the summary. Just summarize!
    #                 """)
    num_output_sentences_ = min(max(math.ceil(math.log(len(sentences))),2), 5)
    system_instructions = dedent(f"""\
                    You are a student in an arabic language course. 
                    Your task is to summarize the opinion of the other students.
                    You are given list of students' opinions separated by |, Summarize their opinions. 
                    The students are answering the question: [<'{col_name.strip()}'>]
                    Format your response with up to {num_output_sentences_} short sentences, phrased like a student's opinion.
                    Your response must be in English. 
                    Focus on bringing the main common response that all students share and one rarer response not necessarily agreeing with the main common response.
                    Do not answer the students, do not add your opinion, do not add comments before or after the summary. Just summarize!
                    """)

        # "Ensure that the quote-examples are very different from each other and direct quotes from the input, in Hebrew."


    messages = [
            {
            'role': 'system',
            'content': system_instructions
            },
            {'role': 'user', 
                'content': sentences}
    ]
    if verbose>10:
        print(model)
        print(messages)
    response = ollama.chat(model, messages=messages)
    return response    


def topic_info_to_keypoints(topic_row ,verbose=0):
    topic_row = str('.'.join([topic_row['Name'], str(set(topic_row['Representation'])), str(set(topic_row['Representative_Docs']))]))
    heb_eng_arab_numeric = re.compile(r'[^\u0590-\u05FF\u0600-\u06FFa-zA-Z]+')
    cleaned_topic_row = heb_eng_arab_numeric.sub(' ', str(topic_row))
    cleaned_topic_row = re.sub(r'\s+', '', cleaned_topic_row).strip()
    if verbose>9:
        print(cleaned_topic_row)

    messages = [
            {
            'role': 'system',
            'content':  'תפקידך למצוא עד חמש מילות מפתח של הטקסט ולהחזירן מופרדות בסימון נקודה. הקפד שכל מילה נבחרת תהיה מהטקסט הנתון ושהמילים תהיינה שונות אחת מן השניה. החזר לכל היותר חמש מילים שונות, בעברית בשורה אחת קצרה, ללא אף מילה נוספת לפני או אחרי, ללא מספור וללא מעבר שורה וללא הסבר נוסף.'
            },
            {'role': 'user', 
                'content': cleaned_topic_row}
    ]

    resp = ollama.chat(model='llama3.1:latest', messages=messages)
    if verbose>9:
        print(resp['message']['content'])
        print()
    return resp['message']['content']

def summarize_topic(col_name, sentences, context_length=1000, verbose=True):
    s = '|'.join(sentences)
    topic_sub_responses = dict()
    if verbose:
        print('='*80)
        print(f'SUMMARIZATION. sentences example:')
        print(sentences[:3])
        print('='*80)
    for batch_start in range(0, len(s), context_length):
        batch_end = batch_start + 20 + context_length
        if verbose>3: print(f'{batch_start}-{batch_end} out of {len(s)}')
        sub_response = summarize(col_name, s[batch_start:batch_end], verbose=verbose)
        if verbose>3: print(sub_response['message']['content'], end='\n\n')
        topic_sub_responses[(batch_start,batch_end)] = sub_response
        # if batch_start > context_length:
        #     break

    clean_topic_subsummaries = [d_['message']['content'] for d_ in topic_sub_responses.values()]
    clean_topic_subsummaries = '|'.join(clean_topic_subsummaries).replace('\n','.').replace('•','.').replace('*','.')
    topic_summary = summarize(col_name, clean_topic_subsummaries, verbose=verbose)
    if verbose:
        print('='*80)
        print(f'SUMMARIZATION DONE:')
        topic_summary_ = topic_summary["message"]["content"]
        print(f'{topic_summary_=}')
        print('='*80)

    return topic_summary
