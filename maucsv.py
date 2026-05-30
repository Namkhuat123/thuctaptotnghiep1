aximport pandas as pd
import datetime
import os
import time
import random
import numpy as np

from vnstock import Vnstock

# ================== KHỞI TẠO ==================
vn = Vnstock()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "vn_stock_data")
os.makedirs(DATA_DIR, exist_ok=True)

# ================== DANH SÁCH 100 MÃ ==================
symbols = [
    "VCB","CTG","BID","MBB","TCB","VPB","ACB","STB","HDB","EIB",
    "VIC","VHM","NVL","DXG","DIG","KBC","CEO","PDR","NLG","BCM",
    "SSI","VND","HCM","VCI","SHS","FTS","BSI","ORS","AGR","TVS",
    "FPT","CMG","FOX","ELC","ITD","CTR","VGI","VTP","VNZ","VDS",
    "HPG","HSG","NKG","GEX","DCM","DPM","BFC","LAS","CSV","PHR",
    "VNM","MWG","PNJ","MSN","SAB","PLX","FRT","DGW","PET","ANV",
    "GAS","PVD","PVS","POW","GEG","NT2","REE","BWE","PC1","CII",
    "GMD","HAH","VSC","VTO","VJC","HVN","SGN","ACV","SCS","NCT"
]

# ================== THỜI GIAN ==================
start_date = (datetime.date.today() - datetime.timedelta(days=5*365)).strftime("%Y-%m-%d")
end_date = datetime.date.today().strftime("%Y-%m-%d")

# ================== CỘT CHUẨN ==================
TARGET_COLUMNS = ['Date','Open','High','Low','Close','Volume','Adjusted Close']

COLUMN_MAPPING = {
    'time': 'Date',
    'date': 'Date',
    'tradingdate': 'Date',
    'open': 'Open',
    'high': 'High',
    'low': 'Low',
    'close': 'Close',
    'volume': 'Volume',
    'adjclose': 'Adjusted Close',
    'adjustedclose': 'Adjusted Close'
}

# ================== HÀM CÀO 1 MÃ ==================
def crawl_one_symbol(symbol):
    try:
        print(f"⏳ Đang tải {symbol} ...")

        stock = vn.stock(symbol=symbol)
        df = stock.quote.history(start=start_date, end=end_date)

        if df.empty:
            print(f"⚠ {symbol}: Không có dữ liệu")
            return

        # Chuẩn hóa cột
        df.columns = df.columns.str.lower()
        df.rename(columns=COLUMN_MAPPING, inplace=True)

        if 'Date' not in df.columns or 'Close' not in df.columns:
            print(f"❌ {symbol}: Thiếu cột Date / Close")
            return

        final_df = pd.DataFrame()

        for col in TARGET_COLUMNS:
            if col in df.columns:
                final_df[col] = df[col]
            elif col == 'Adjusted Close':
                final_df[col] = df['Close']
            else:
                final_df[col] = np.nan

        final_df['Date'] = pd.to_datetime(final_df['Date']).dt.strftime('%d-%m-%Y')

        for col in TARGET_COLUMNS[1:]:
            final_df[col] = pd.to_numeric(final_df[col], errors='coerce')

        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")
        final_df.to_csv(file_path, index=False)

        print(f"✔ {symbol}: Lưu thành công")

        # Delay để tránh bị chặn
        time.sleep(random.uniform(0.6, 1.2))

    except Exception as e:
        print(f"❌ {symbol}: Lỗi {e}")

# ================== MAIN ==================
if __name__ == "__main__":
    print(f"🚀 Bắt đầu cào {len(symbols)} mã cổ phiếu Việt Nam\n")

    for symbol in symbols:
        crawl_one_symbol(symbol)

    print(f"\n🎉 Hoàn thành! Dữ liệu đã lưu tại: {DATA_DIR}")
