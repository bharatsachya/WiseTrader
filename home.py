import streamlit as st
import yfinance as yf
from pages import wise_cnn as wp
from pages import wise_lstm as ls


# import wise_lstm # Import the functions from separate files for each page
# import wise_rnn # Import the functions from separate files for each p
# Config
st.set_page_config(layout="wide", page_icon="🤑", page_title="Lovanshu | WiseTrader📈")

# Contact
with st.sidebar.expander("📬 Contact"):
    st.write("**GitHub:**", "[bharatsachya/WiseTrader](https://github.com/bharatsachya/WiseTrader)")

# Main content
st.markdown("""<h1 style='text_align:center'>Wise Trader</h1>
            <h2 style='text_align:center'>All Neural Network Algorithms in One App</h2>
            """, True)
st.markdown("---")
st.title('Stock Watcher')

stock_ticker = st.text_input('Enter Stock Ticker', value='NV20.NS', key='unique_key_for_text_input')

df = yf.download(stock_ticker, period='id', interval="1m")

data = df[['Open', 'High', 'Low', 'Close', 'Volume']].values

st.write(data)

if st.button('stocks'):
    # Handle button click event if needed
    pass

st.subheader("🚀 WiseTrader Pages")

# Sidebar navigation
page_options = ["Wise CNN", "Wise RNN", "Wise LSTM"]
selected_page = st.sidebar.selectbox("Select Page", page_options)

# Display the selected page
if selected_page == "Wise CNN":
    wp.display_page()
elif selected_page == "Wise LSTM":
    ls.display_page()

st.markdown("---")

st.markdown("### 🎯 Contributing")
st.markdown("""
****
""", unsafe_allow_html=True)
