from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.agents import create_csv_agent
from langchain.llms import OpenAI

openai_api_key=st.secrets["OPENAI_API_KEY"]
def main():
    load_dotenv()
    st.set_page_config(page_title="Project - FDRaj")
    st.header("Project - FDRaj")
    
    # upload file
    csv = st.file_uploader("Upload your CSV only", type="csv")
    
    # extract the text
    if csv is not None:
      agent = create_csv_agent(OpenAI(temperature=0), csv, verbose=True, header=True)
      print(agent.agent.llm_chain.prompt.template)
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        answer = agent.run(user_question)
           
        st.write(answer)
    

if __name__ == '__main__':
    main()
