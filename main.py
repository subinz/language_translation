import streamlit as st
from transformers import FSMTForConditionalGeneration, FSMTTokenizer

# @st.cache(allow_output_mutation=True)

def translate_de_to_en(tokenizer, model, sentence):
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    outputs = model.generate(input_ids)
    # print(model.forward(input_ids)['logits'])
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

def clear_form():
    st.session_state["form-text"] = ""


def main():
    mname = "facebook/wmt19-de-en"
    tokenizer = FSMTTokenizer.from_pretrained(mname)
    model = FSMTForConditionalGeneration.from_pretrained(mname)
    st.set_page_config(layout="wide")
    st.markdown("<h1 style='text-align: center; color: black;'>Language Translator</h1>", unsafe_allow_html=True)
    with st.expander("Project Details"):
        st.markdown("""
        ## Add the deatils

        ### Business Purpose

        Add the details.

        
        ### Benefits

        Add the details
        """, unsafe_allow_html=True)

    with st.form("myform"):
        text = st.text_area(label="Enter the German text", placeholder="Type or paste German text here", key='form-text')
        col1, col2 = st.columns([1,7])
        with col1:
            translate = st.form_submit_button("Translate")
        with col2:
            clear = st.form_submit_button("Clear", on_click=clear_form)

    if translate:
        translated_text = translate_de_to_en(tokenizer, model, text)
        st.write('Translated Sentence:', translated_text)
        

if __name__ == '__main__':
    main()
