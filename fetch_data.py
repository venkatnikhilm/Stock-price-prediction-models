import yfinance as yf

class FetchData:
    def __init__(self, symbol, start_date, end_date):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def fetch_data(self):
        try:
            stock_data = yf.download(self.symbol, start=self.start_date, end=self.end_date)
            return stock_data
        except Exception as e:
            print(f"Error fetching data: {str(e)}")
            return None