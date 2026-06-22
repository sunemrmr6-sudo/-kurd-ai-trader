import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="AI Professional Trading System", layout="centered")
st.title("🦅 سیستەمی پێشکەوتووی سیگناڵی AI & Order Block")

col1, col2 = st.columns(2)
with col1:
    ticker = st.text_input("نیشانەی دراو (بۆ نموونە: EURUSD=X یان BTC-USD):", "EURUSD=X")
with col2:
    balance = st.number_input("قەبارەی ئەکاونتەکەت بە دۆلار ($):", min_value=10, value=1000)

risk_percent = st.slider("ڕێژەی سەرکێشی (Risk %) لە هەر تریدێکدا:", 0.5, 5.0, 1.0)

if st.button("🚀 دەستپێکردنی شیکاری قووڵ"):
    with st.spinner("AI خەریکی پشکنینی پۆتانسێلی بازاڕ و ڕەوتی بانکەکانە..."):
        try:
            data = yf.download(ticker, period="1y", interval="1d")
            
            if data.empty:
                st.error("داتا نەدۆزرایەوە!")
            else:
                is_forex = "=" in ticker or len(ticker) == 6
                
                data['Bullish_OB'] = np.where((data['Close'] > data['Open']) & (data['Close'].shift(1) < data['Open'].shift(1)), 1, 0)
                data['Bearish_OB'] = np.where((data['Close'] < data['Open']) & (data['Close'].shift(1) > data['Open'].shift(1)), 1, 0)
                data['Price_Change'] = data['Close'].pct_change()
                
                data['Target'] = np.where(data['Price_Change'].shift(-1) > 0, 1, 0)
                data.dropna(inplace=True)
                
                features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Bullish_OB', 'Bearish_OB']
                X = data[features]
                y = data['Target']
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                model.fit(X_train, y_train)
                
                predictions = model.predict(X_test)
                acc = accuracy_score(y_test, predictions) * 100
                
                last_candle = X.tail(1)
                ai_pred = model.predict(last_candle)[0]
                current_price = float(data['Close'].iloc[-1])
                
                prev_high = float(data['High'].iloc[-2])
                prev_low = float(data['Low'].iloc[-2])
                
                st.markdown("---")
                st.subheader(f"📊 ڕێژەی هێز و دروستی ستراتیژی لەم دراوەدا: {acc:.2f}%")
                
                if ai_pred == 1 and data['Bullish_OB'].iloc[-1] == 1:
                    st.success("🟢 **سیگناڵی بەهێز: BUY (کڕین)**")
                    entry = current_price
                    sl = prev_low
                    risk_amount = entry - sl
                    tp = entry + (risk_amount * 2)
                    
                elif ai_pred == 0 and data['Bearish_OB'].iloc[-1] == 1:
                    st.error("🔴 **سیگناڵی بەهێز: SELL (فرۆشتن)**")
                    entry = current_price
                    sl = prev_high
                    risk_amount = sl - entry
                    tp = entry - (risk_amount * 2)
                else:
                    st.warning("🟡 **سیگناڵ: چاوەڕوانی (WAIT)**")
                    st.info("AI پێشنیار دەکات ئێستا نەچیتە کڕین یان فرۆشتنەوە چونکە مەرجەکانی Order Block تەواو نین.")
                    entry = None
                
                if entry is not None:
                    st.markdown("### 📋 زانیارییەکانی پۆزیشن:")
                    st.write(f"📥 **نرخی چوونە ژوورەوە:** {entry:.5f if is_forex else .2f}")
                    st.write(f"🛑 **ستۆپ Lۆس (Stop Loss):** {sl:.5f if is_forex else .2f}")
                    st.write(f"🎯 **تێک پڕۆفیت (Take Profit):** {tp:.5f if is_forex else .2f}")
                    
                    allowed_loss = balance * (risk_percent / 100)
                    st.markdown("---")
                    st.subheader("💰 بەڕێوەبردنی سەرمایە (Risk Management):")
                    st.write(f"💵 بڕی پارەی ڕێگەپێدراو بۆ زیان لەم تریدەدا: **${allowed_loss:.2f}**")
                    
                    if is_forex:
                        pips_at_risk = abs(entry - sl) * 10000
                        if pips_at_risk > 0:
                            lot_size = allowed_loss / (pips_at_risk * 10)
                            st.info(f"📏 پێشنیاری قەبارەی گرێبەست (Lot Size): **{lot_size:.2f} Lot**")
                    else:
                        crypto_amount = allowed_loss / risk_amount
                        st.info(f"🪙 پێشنیاری بڕی کڕین: **{crypto_amount:.4f}** لەم دراوە.")
                        
                st.markdown("---")
                
        except Exception as e:
            st.error(f"هەڵەیەک لە کاتی شیکاردا ڕوویدا: {e}")
