import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

import math
import warnings
warnings.filterwarnings('ignore')

# --- CẤU HÌNH GIAO DIỆN STREAMLIT ---
st.set_page_config(
    page_title="Dự báo Giá Cổ phiếu bằng LSTM",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📈 Dự báo Giá Cổ phiếu bằng Mô hình LSTM")
st.markdown("Đây là ứng dụng Streamlit dựa trên mô hình Mạng Hồi quy Dài-Ngắn Hạn (LSTM) để phân tích và dự báo giá đóng cửa của cổ phiếu.")


# --- HÀM TẠO TẬP DỮ LIỆU CHUỖI THỜI GIAN (TIME STEPS) ---
def create_dataset(dataset, time_step=1):
    """
    Tạo chuỗi (X, Y) cho mô hình LSTM.
    X: time_step ngày trước đó (đầu vào)
    Y: giá ngày tiếp theo (đầu ra/nhãn)
    """
    X, Y = [], []
    for i in range(len(dataset) - time_step):
        # Lấy 'time_step' ngày trước đó làm X (đầu vào)
        a = dataset[i:(i + time_step), 0]
        X.append(a)
        # Giá ngày tiếp theo làm Y (đầu ra/dự đoán)
        Y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(Y)

# --- HÀM CHÍNH CHẠY MÔ HÌNH (Dùng caching để tránh huấn luyện lại) ---
@st.cache_data(show_spinner=False)
def run_lstm_model(df, column_name, time_step, epochs, batch_size):
    """Thực hiện toàn bộ quá trình: tiền xử lý, xây dựng, huấn luyện và dự đoán."""
    
    # 1. Chuẩn bị Dữ liệu
    data = df[column_name].values.reshape(-1, 1)
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    # Chia tập Train và Test (80/20)
    train_ratio = 0.8
    training_size = int(len(scaled_data) * train_ratio)
    train_data = scaled_data[:training_size]
    test_data = scaled_data[training_size:]

    # Kiểm tra kích thước dữ liệu để đảm bảo đủ mẫu
    if len(train_data) <= time_step or len(test_data) <= time_step:
        st.error(f"Kích thước dữ liệu quá nhỏ so với Time Step ({time_step}).")
        raise ValueError("Dữ liệu không đủ để tạo tập train/test.")

    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    
    # Thay đổi hình dạng dữ liệu cho LSTM: [mẫu, time_step, features]
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # 2. Xây dựng Mô hình LSTM
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    
    st.subheader("Tóm Tắt Mô Hình")
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    st.text('\n'.join(model_summary))

    # 3. Huấn luyện Mô hình
    with st.spinner(f"Đang Huấn luyện mô hình với **{epochs}** epochs và **Batch Size {batch_size}**..."):
        history = model.fit(
            X_train,
            Y_train,
            validation_data=(X_test, Y_test),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0 
        )
    st.success("Huấn luyện mô hình hoàn tất!")

    # 4. Dự đoán và Đánh giá
    train_predict = model.predict(X_train, verbose=0)
    test_predict = model.predict(X_test, verbose=0)

    # Đảo ngược chuẩn hóa
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    Y_train_actual = scaler.inverse_transform(Y_train.reshape(-1, 1))
    Y_test_actual = scaler.inverse_transform(Y_test.reshape(-1, 1))

    # Tính RMSE
    train_rmse = math.sqrt(mean_squared_error(Y_train_actual, train_predict))
    test_rmse = math.sqrt(mean_squared_error(Y_test_actual, test_predict))
    
    # Chuẩn bị dữ liệu cho biểu đồ
    look_back = time_step
    
    # Dự đoán trên tập Train
    train_predict_plot = np.empty_like(data)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[look_back:len(train_predict) + look_back, :] = train_predict

    # Dự đoán trên tập Test
    test_predict_plot = np.empty_like(data)
    test_predict_plot[:, :] = np.nan
    start_index = len(train_data) + look_back
    end_index = start_index + len(test_predict) 
    test_predict_plot[start_index:end_index, :] = test_predict
    
    # Tạo DataFrame kết quả
    result_df = pd.DataFrame(data, columns=['Giá Thực tế'], index=df.index)
    result_df['Dự đoán (Train)'] = train_predict_plot
    result_df['Dự đoán (Test)'] = test_predict_plot

    return result_df, train_rmse, test_rmse, start_index


# --- SIDEBAR: CẤU HÌNH THAM SỐ VÀ TẢI DỮ LIỆU ---
st.sidebar.header("Tải Dữ liệu")
uploaded_file = st.sidebar.file_uploader(
    "Vui lòng tải lên tệp CSV chứa dữ liệu giá cổ phiếu", 
    type="csv"
)

if uploaded_file is not None:
    # Đọc tệp và hiển thị
    df = pd.read_csv(uploaded_file)
    # Thử tìm cột 'Date' hoặc 'Ngày' và đặt làm index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'Ngày' in df.columns:
        df['Ngày'] = pd.to_datetime(df['Ngày'])
        df.set_index('Ngày', inplace=True)
        
    st.sidebar.success("Tải tệp thành công!")
    st.sidebar.dataframe(df.head())
    
    # Lấy tên cột giá đóng cửa
    column_options = [col for col in df.columns if df[col].dtype in ['float64', 'int64']]
    
    if not column_options:
        st.sidebar.error("Không tìm thấy cột dữ liệu số trong tệp.")
        st.stop()
        
    selected_column = st.sidebar.selectbox(
        "Chọn cột giá để dự đoán (Ví dụ: Close)",
        column_options,
        index=column_options.index('Close') if 'Close' in column_options else 0
    )
    
    st.sidebar.header("Cấu hình Mô hình LSTM")
    
    # Tham số đầu vào cho người dùng
    time_step = st.sidebar.slider(
        "Time Step (Số ngày nhìn lại):", 
        min_value=10, 
        max_value=120, 
        value=60, 
        step=10,
        help="Số ngày giá cổ phiếu trước đó được dùng để dự đoán giá ngày tiếp theo."
    )
    epochs = st.sidebar.slider(
        "Số Epochs (Huấn luyện):", 
        min_value=10, 
        max_value=200, 
        value=100, 
        step=10,
        help="Số lần mô hình nhìn qua toàn bộ tập dữ liệu."
    )
    batch_size = st.sidebar.slider(
        "Batch Size:", 
        min_value=16, 
        max_value=128, 
        value=64, 
        step=16,
        help="Số mẫu được xử lý trước khi cập nhật trọng số."
    )
    
    # Nút chạy mô hình
    if st.sidebar.button("Bắt đầu Huấn luyện và Dự đoán"):
        
        try:
            # Chạy mô hình
            result_df, train_rmse, test_rmse, start_index = run_lstm_model(
                df, selected_column, time_step, epochs, batch_size
            )
            
            # --- HIỂN THỊ KẾT QUẢ ĐÁNH GIÁ ---
            st.subheader("Kết quả Đánh giá Độ chính xác (RMSE)")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("RMSE (Tập Huấn luyện)", f"{train_rmse:.2f} VND")
            
            with col2:
                st.metric("RMSE (Tập Kiểm tra)", f"{test_rmse:.2f} VND")
                
            st.markdown(f"**Lưu ý:** Mô hình nhìn lại **{time_step}** ngày để dự đoán giá.")
            
            # --- HIỂN THỊ BIỂU ĐỒ TỔNG THỂ ---
            st.subheader("Biểu đồ Dự báo Giá Cổ phiếu (Tổng thể)")
            
            fig, ax = plt.subplots(figsize=(15, 6))
            ax.plot(result_df.index, result_df['Giá Thực tế'], label='Giá Thực tế', color='#1f77b4', linewidth=2)
            ax.plot(result_df.index, result_df['Dự đoán (Train)'], label='Dự đoán (Train)', color='#2ca02c', linestyle='--')
            ax.plot(result_df.index, result_df['Dự đoán (Test)'], label='Dự đoán (Test)', color='#d62728', linestyle='--')
            
            ax.set_title('Dự báo Giá Cổ phiếu bằng Mô hình LSTM (Tổng thể)', fontsize=16)
            # Điều chỉnh nhãn trục X nếu index là datetime
            if isinstance(result_df.index, pd.DatetimeIndex):
                ax.set_xlabel('Ngày', fontsize=12)
                fig.autofmt_xdate() # Tự động xoay ngày
            else:
                ax.set_xlabel('Chỉ số Ngày (Index)', fontsize=12)
                
            ax.set_ylabel('Giá Cổ phiếu', fontsize=12)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig)
            
            
            # --- HIỂN THỊ BIỂU ĐỒ CHI TIẾT TẬP TEST ---
            st.subheader("Biểu đồ So sánh Giá Thực tế và Dự đoán trên Tập Kiểm tra")

            # Lọc dữ liệu chỉ cho tập test
            test_data_actual = result_df['Giá Thực tế'].iloc[start_index:]
            test_data_predicted = result_df['Dự đoán (Test)'].iloc[start_index:]
            
            fig_test, ax_test = plt.subplots(figsize=(15, 6))
            ax_test.plot(test_data_actual.index, test_data_actual, label='Giá Thực tế (Test Set)', color='#1f77b4', linewidth=2)
            ax_test.plot(test_data_predicted.index, test_data_predicted, label='Dự đoán (Test Set)', color='#d62728', linewidth=2)
            
            ax_test.set_title('So sánh Chi tiết trên Tập Kiểm tra', fontsize=16)
            
            if isinstance(result_df.index, pd.DatetimeIndex):
                ax_test.set_xlabel('Ngày', fontsize=12)
                fig_test.autofmt_xdate()
            else:
                ax_test.set_xlabel('Chỉ số Ngày (Index)', fontsize=12)
            
            ax_test.set_ylabel('Giá Cổ phiếu', fontsize=12)
            ax_test.legend(loc='upper left', fontsize=10)
            ax_test.grid(True, linestyle=':', alpha=0.6)
            st.pyplot(fig_test)
            
        except ValueError as e:
            # Bắt lỗi khi dữ liệu không đủ
            if "Dữ liệu không đủ" in str(e):
                st.error("Lỗi: Dữ liệu quá ngắn hoặc Time Step quá lớn. Vui lòng giảm Time Step.")
            else:
                st.error(f"Đã xảy ra lỗi trong quá trình chạy mô hình: {e}")

else:
    st.info("Vui lòng tải lên tệp CSV dữ liệu cổ phiếu của bạn ở cột bên trái để bắt đầu.")