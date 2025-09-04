import getpass
import google.generativeai as genai
import yfinance as yf
import numpy as np
import requests
import os
import datetime as dt
from bs4 import BeautifulSoup
import pandas as pd

# 初始化 Gemini API
GOOGLE_API_KEY = getpass.getpass("請輸入Gemini API金鑰：")
genai.configure(api_key=GOOGLE_API_KEY)


class StockInfo():
    # 取得全部股票的股號、股名
    def stock_name(self):
        response = requests.get('https://isin.twse.com.tw/isin/C_public.jsp?strMode=2')
        url_data = BeautifulSoup(response.text, 'html.parser')
        stock_company = url_data.find_all('tr')

        data = [
            (row.find_all('td')[0].text.split('\u3000')[0].strip(),
             row.find_all('td')[0].text.split('\u3000')[1],
             row.find_all('td')[4].text.strip())
            for row in stock_company[2:] if len(row.find_all('td')[0].text.split('\u3000')[0].strip()) == 4
        ]

        df = pd.DataFrame(data, columns=['股號', '股名', '產業別'])
        return df

    # 取得股票名稱
    def get_stock_name(self, stock_id, name_df):
        return name_df.set_index('股號').loc[stock_id, '股名']


class StockAnalysis():
    def __init__(self, google_api_key):
        # 初始化 Gemini Model
        self.model = genai.GenerativeModel("gemini-1.5-pro")
        self.stock_info = StockInfo()
        self.name_df = self.stock_info.stock_name()

    # 從 yfinance 取得一周股價資料
    def stock_price(self, stock_id="大盤", days=15):
        if stock_id == "大盤":
            stock_id = "^TWII"
        else:
            stock_id += ".TW"

        end = dt.date.today()
        start = end - dt.timedelta(days=days)

        df = yf.download(stock_id, start=start, auto_adjust=False, multi_level_index=False)
        df.columns = ['調整後收盤價', '收盤價', '最高價', '最低價', '開盤價', '成交量']

        data = {
            '日期': df.index.strftime('%Y-%m-%d').tolist(),
            '收盤價': df['收盤價'].tolist(),
            '每日報酬': df['收盤價'].pct_change().tolist(),
        }
        return data

    # 基本面資料
    def stock_fundamental(self, stock_id="大盤"):
        if stock_id == "大盤":
            return None

        stock_id += ".TW"
        stock = yf.Ticker(stock_id)

        quarterly_revenue_growth = np.round(
            stock.quarterly_financials.loc["Total Revenue"].pct_change(-1, fill_method=None).dropna().tolist(), 2
        )
        quarterly_eps = np.round(stock.quarterly_financials.loc["Basic EPS"].dropna().tolist(), 2)
        quarterly_eps_growth = np.round(
            stock.quarterly_financials.loc["Basic EPS"].pct_change(-1, fill_method=None).dropna().tolist(), 2
        )

        dates = [date.strftime('%Y-%m-%d') for date in stock.quarterly_financials.columns]

        data = {
            '季日期': dates[:len(quarterly_revenue_growth)],
            '營收成長率': quarterly_revenue_growth.tolist(),
            'EPS': quarterly_eps.tolist(),
            'EPS 季增率': quarterly_eps_growth.tolist()
        }
        return data

    # 新聞資料
    def stock_news(self, stock_name="大盤"):
        if stock_name == "大盤":
            stock_name = "台股 -盤中速報"

        data = []
        json_data = requests.get(
            f'https://ess.api.cnyes.com/ess/api/v1/news/keyword?q={stock_name}&limit=6&page=1'
        ).json()

        items = json_data['data']['items']
        for item in items:
            news_id = item["newsId"]
            title = item["title"]
            publish_at = item["publishAt"]
            utc_time = dt.datetime.utcfromtimestamp(publish_at)
            formatted_date = utc_time.strftime('%Y-%m-%d')

            url = requests.get(f'https://news.cnyes.com/news/id/{news_id}').content
            soup = BeautifulSoup(url, 'html.parser')
            p_elements = soup.find_all('p')
            p = ''.join([paragraph.get_text() for paragraph in p_elements[4:]])

            data.append([stock_name, formatted_date, title, p])
        return data

    # 呼叫 Gemini 取回回覆
    def get_reply(self, messages):
        try:
            # 把 messages 轉成一個 prompt
            prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])
            response = self.model.generate_content(prompt)
            reply = response.text
        except Exception as err:
            reply = f"發生錯誤: {str(err)}"
        return reply

    def ai_helper(self, user_msg):
        code_example = """def calculate(table_company, table_daily, table_quarterly): ..."""

        user_requirement = [{
            "role": "user",
            "content": f"The user requirement: {user_msg}"
        }]

        msg = [{
            "role": "system",
            "content": "You are a stock selection strategy robot..."
        }, {
            "role": "assistant",
            "content": code_example
        }]
        msg += user_requirement

        reply_data = self.get_reply(msg)
        cleaned_code = reply_data.replace("```", "").replace("python", "")
        return user_requirement, cleaned_code

    def ai_debug(self, history, code_str, error_msg):
        msg = [{
            "role": "system",
            "content": "You will act as a professional Python code generation robot..."
        }]
        msg += history
        msg += [{
            "role": "assistant",
            "content": code_str
        }, {
            "role": "user",
            "content": f"The error code:{code_str}\nThe error message:{error_msg}"
        }]

        reply_data = self.get_reply(msg)
        cleaned_code = reply_data.replace("```", "").replace("python", "")
        return cleaned_code

    def generate_content_msg(self, stock_id, name_df):
        stock_name = self.stock_info.get_stock_name(stock_id, name_df) if stock_id != "大盤" else stock_id
        price_data = self.stock_price(stock_id)
        news_data = self.stock_news(stock_name)

        content_msg = f'你現在是一位專業的證券分析師...\n近期價格資訊:\n {price_data}\n'
        if stock_id != "大盤":
            stock_value_data = self.stock_fundamental(stock_id)
            content_msg += f'每季營收資訊：\n {stock_value_data}\n'

        content_msg += f'近期新聞資訊: \n {news_data}\n請給我{stock_name}近期的趨勢報告...'
        return content_msg

    def stock_gpt(self, stock_id):
        content_msg = self.generate_content_msg(stock_id, self.name_df)
        msg = [{
            "role": "system",
            "content": "你現在是一位專業的證券分析師..."
        }, {
            "role": "user",
            "content": content_msg
        }]

        reply_data = self.get_reply(msg)
        return reply_data
