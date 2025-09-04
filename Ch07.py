import os
import time
import random
import zipfile
import io
import requests
from bs4 import BeautifulSoup
import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.summarize import load_summarize_chain
from more_itertools import batched
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings # 引入 Gemini 相關套件

class PdfLoader:
    def __init__(self, google_api_key):
        # 將環境變數設定為 Google API Key
        os.environ['GOOGLE_API_KEY'] = google_api_key
        
        # 使用 ChatGoogleGenerativeAI，並指定模型名稱，例如 'gemini-pro' 或 'gemini-1.5-pro-latest'
        self.llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-1.5-pro-latest")
        
        # 提示詞的部分不需要變動
        self.data_prompt = ChatPromptTemplate.from_messages(messages=[("system", "你的任務是對年報資訊進行摘要總結。"
                                                                       "以下為提供的年報資訊：{text},"
                                                                       "請給我重點數據, 如銷售增長情形、營收變化、開發項目等,"
                                                                       "最後請使用繁體中文輸出報告")])
        self.data_chain = load_summarize_chain(llm=self.llm, chain_type='stuff', prompt=self.data_prompt)

    def annual_report(self, id, y):
        wait_time = random.uniform(2, 6)
        url = 'https://doc.twse.com.tw/server-java/t57sb01'
        folder_path = '/content/drive/MyDrive/StockGPT/PDF/'
        
        data = {
            "id": "",
            "key": "",
            "step": "1",
            "co_id": id,
            "year": y,
            "seamon": "",
            "mtype": 'F',
            "dtype": 'F04'
        }
        
        with requests.post(url, data=data) as response:
            time.sleep(wait_time)
            link = BeautifulSoup(response.text, 'html.parser')
            link1 = link.find('a').text
            print(link1)
        
        data2 = {
            'step': '9',
            'kind': 'F',
            'co_id': id,
            'filename': link1
        }
        
        file_extension = link1.split('.')[-1]
        
        if file_extension == 'zip':
            with requests.post(url, data=data2) as response2:
                if response2.status_code == 200:
                    zip_data = io.BytesIO(response2.content)
                    with zipfile.ZipFile(zip_data) as myzip:
                        for file_info in myzip.namelist():
                            if file_info.endswith('.pdf'):
                                with myzip.open(file_info) as myfile:
                                    with open(folder_path + y + '_' + id + '.pdf', 'wb') as f:
                                        f.write(myfile.read())
                                    print('ok')
        else:
            with requests.post(url, data=data2) as response2:
                time.sleep(wait_time)
                link = BeautifulSoup(response2.text, 'html.parser')
                link1 = link.find('a')['href']
                print(link1)
            
            response3 = requests.get('https://doc.twse.com.tw' + link1)
            time.sleep(wait_time)
            
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            with open(folder_path + y + '_' + id + '.pdf', 'wb') as file:
                file.write(response3.content)
            print('OK')
            
    def pdf_loader(self, file, size, overlap):
        logging.getLogger("pdfminer").setLevel(logging.ERROR)
        loader = PDFPlumberLoader(file)
        doc = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=size,
                                                         chunk_overlap=overlap)
        docs = text_splitter.split_documents(doc)
        
        # 使用 GoogleGenerativeAIEmbeddings 來替代 OpenAIEmbeddings
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        
        faiss_db = None
        for doc_batch in batched(docs, 100):
            doc_batch = list(doc_batch)
            if faiss_db is None:
                faiss_db = FAISS.from_documents(doc_batch, embeddings)
            else:
                faiss_db.add_documents(doc_batch)
        
        file_name = file.split("/")[-1].split(".")[0]
        db_file = '/content/drive/MyDrive/StockGPT/DB/'
        if not os.path.exists(db_file):
            os.makedirs(db_file)
        faiss_db.save_local(db_file + file_name)
        
        return faiss_db
        
    def analyze_chain(self, db, input):
        data = db.similarity_search(input, k=5)
        
        result = self.data_chain.invoke({"input_documents": data})
        return result['output_text']
