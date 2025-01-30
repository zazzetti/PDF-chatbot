from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
import uvicorn
import os
from dotenv import load_dotenv
#from langchain_deepseek import ChatDeepSeek


custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

app = FastAPI()
load_dotenv()
UPLOAD_FOLDER = "data" 
api_key = os.getenv("MISTRAL_API_KEY")

conversation_chain = None

def get_pdf_text(pdf_files):
    text = ""
    for pdf_file in pdf_files:
        pdf_reader = PdfReader(pdf_file.file)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)



def get_vectorstore(chunks):
    embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                     model_kwargs={'device':'cpu'})
    vectorstore=FAISS.from_texts(texts=chunks,embedding=embeddings)
    return vectorstore

def initialize_conversation_chain(vectorstore):
    global conversation_chain
    #llm = ChatDeepSeek(
    #model="deepseek-chat", temperature=0.2)
    llm=ChatMistralAI(temperature=0.2, mistral_api_key=api_key)

    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), condense_question_prompt=CUSTOM_QUESTION_PROMPT, memory=memory
    )




def get_pdf_text_from_folder(folder_path):
    text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            with open(pdf_path, "rb") as pdf_file:
                pdf_reader = PdfReader(pdf_file)
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
    return text


@app.post("/process_folder/")
async def process_pdf_folder():
    try:
        raw_text = get_pdf_text_from_folder(UPLOAD_FOLDER)
        if not raw_text:
            raise HTTPException(status_code=400, detail="No valid PDF text found.")
        
        text_chunks = get_text_chunks(raw_text)
        vectorstore = get_vectorstore(text_chunks)
        initialize_conversation_chain(vectorstore)
        
        return JSONResponse(content={"message": "PDFs from folder processed successfully"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/")
async def chat_with_pdfs(question: str = Form(...)):
    if not conversation_chain:
        raise HTTPException(status_code=400, detail="No documents uploaded yet")
    response = conversation_chain({'question': question})
    chat_history = response['chat_history']
    return JSONResponse(content={"response": chat_history[-1].content})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
