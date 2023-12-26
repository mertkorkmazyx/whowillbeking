import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime
from scipy.stats import norm


#Güvenilirlik
def calculate_confidence_interval(predictions):
    mean = np.mean(predictions)
    std_dev = np.std(predictions)
    z_value = norm.ppf(0.975)  # %95 güven aralığı için

    lower_bound = mean - z_value * std_dev
    upper_bound = mean + z_value * std_dev

    return lower_bound, upper_bound, std_dev

# Tahminleme Yapma
def predict_without_plot(model_chandler, model_joey, model_kararsiz, user_date):
    random_dates = pd.to_datetime(np.random.choice(pd.date_range(end=user_date, periods=90), 3), format='%Y-%m-%d')

    # Tahminleme için boş liste oluşturma
    chandler_preds = []
    joey_preds = []
    kararsiz_preds = []

    # Her Tarih için Aşağıdaki işlemler uygulanır(Normalize edilip, listeye ekleniyor)
    for date in random_dates:
        user_data = pd.DataFrame({'Tarih': [date], 'Toplam': [np.nan], 'Kararsız': [np.nan]})
        user_data.set_index('Tarih', inplace=True)

        chandler_pred = model_chandler.predict(user_data.index.values.reshape(-1, 1))
        joey_pred = model_joey.predict(user_data.index.values.reshape(-1, 1))
        kararsiz_pred = model_kararsiz.predict(user_data.index.values.reshape(-1, 1))

        total_pred = chandler_pred + joey_pred + kararsiz_pred
        scale_factor = 100 / total_pred.sum()
        chandler_pred *= scale_factor
        joey_pred *= scale_factor
        kararsiz_pred *= scale_factor

        chandler_preds.append(chandler_pred[0])
        joey_preds.append(joey_pred[0])
        kararsiz_preds.append(kararsiz_pred[0])

    # Ortalamalar alınıyor
    chandler_pred_avg = np.mean(chandler_preds)
    joey_pred_avg = np.mean(joey_preds)
    kararsiz_pred_avg = np.mean(kararsiz_preds)

    #Güven Aralıkları ve Standart Sapma Hesaplanıyor
    chandler_lower, chandler_upper, chandler_std_dev = calculate_confidence_interval(chandler_preds)
    joey_lower, joey_upper, joey_std_dev = calculate_confidence_interval(joey_preds)
    kararsiz_lower, kararsiz_upper, kararsiz_std_dev = calculate_confidence_interval(kararsiz_preds)

    return (
        chandler_pred_avg, joey_pred_avg, kararsiz_pred_avg,
        chandler_lower, chandler_upper, chandler_std_dev,
        joey_lower, joey_upper, joey_std_dev,
        kararsiz_lower, kararsiz_upper, kararsiz_std_dev
    )

# Veriyi oku ve işle
df = pd.read_csv('veri_seti.csv')
df.replace('–', np.nan, inplace=True)
df = df.dropna()
df['Tarih'] = pd.to_datetime(df[['Gün', 'Ay', 'Yıl']].astype(str).agg('-'.join, axis=1), format='%d-%m-%Y')
df['Toplam'] = df['Chandler'] + df['Joey']
df = df[['Tarih', 'Toplam', 'Kararsız']]
df.set_index('Tarih', inplace=True)

# Modelleme - RandomForestRegressor
random_state_chandler = np.random.randint(1, 100)
random_state_joey = np.random.randint(1, 100)
random_state_kararsiz = np.random.randint(1, 100)

model_chandler = RandomForestRegressor(n_estimators=100, random_state=random_state_chandler)
model_joey = RandomForestRegressor(n_estimators=100, random_state=random_state_joey)
model_kararsiz = RandomForestRegressor(n_estimators=100, random_state=random_state_kararsiz)

# Modeli eğit
model_chandler.fit(df.index.values.reshape(-1, 1), df['Toplam'])
model_joey.fit(df.index.values.reshape(-1, 1), df['Toplam'])
model_kararsiz.fit(df.index.values.reshape(-1, 1), df['Kararsız'])

# Streamlit uygulaması
# Sayfa başlığı

st.markdown("""
    <div style='text-align: center;'>
        <h1>Who Will Be King?</h1>
    </div>
""", unsafe_allow_html=True)

# Resim
image_path = "images/Friendship2.jpg"
st.image(image_path, caption='İyi Olan Kazansın!', use_column_width=True)

# Kullanıcıdan tarih girişi al
user_input = st.text_input("Tarih (GG-AA-YYYY formatında):")

if user_input:
    try:
        user_date = datetime.strptime(user_input, "%d-%m-%Y")

        user_data = pd.DataFrame({'Tarih': [user_date], 'Toplam': [np.nan], 'Kararsız': [np.nan]})
        user_data.set_index('Tarih', inplace=True)

        # Tahmin yap ve sonuçları al
        (
            chandler_pred, joey_pred, kararsiz_pred,
            chandler_lower, chandler_upper, chandler_std_dev,
            joey_lower, joey_upper, joey_std_dev,
            kararsiz_lower, kararsiz_upper, kararsiz_std_dev
        ) = predict_without_plot(model_chandler, model_joey, model_kararsiz, user_date)

        # Sonuçları görüntüle
        st.text("\nSonuçlar:")
        st.text(f"Tarih: {user_date}")
        st.text(f"Tahmini Chandler: {chandler_pred:.2f} Güven Aralığı: ({chandler_lower:.2f}, {chandler_upper:.2f}) Standart Sapma: {chandler_std_dev:.2f}")
        st.text(f"Tahmini Joey: {joey_pred:.2f} Güven Aralığı: ({joey_lower:.2f}, {joey_upper:.2f}) Standart Sapma: {joey_std_dev:.2f}")
        st.text(f"Tahmini Kararsız: {kararsiz_pred:.2f} Güven Aralığı: ({kararsiz_lower:.2f}, {kararsiz_upper:.2f}) Standart Sapma: {kararsiz_std_dev:.2f}")

        # Çizgi grafiği oluştur
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(df.index, df['Toplam'], label='Gerçek Toplam', color='blue')
        ax.plot(user_date, chandler_pred, marker='o', markersize=8, label='Tahmini Chandler', color='orange')
        ax.plot(user_date, joey_pred, marker='o', markersize=8, label='Tahmini Joey', color='green')
        ax.plot(user_date, kararsiz_pred, marker='o', markersize=8, label='Tahmini Kararsız', color='red')

        # Güven aralıklarını göster
        ax.fill_betweenx(y=[0, 100], x1=user_date, x2=user_date, color='orange', alpha=0.3)
        ax.fill_betweenx(y=[0, 100], x1=user_date, x2=user_date, color='green', alpha=0.3)
        ax.fill_betweenx(y=[0, 100], x1=user_date, x2=user_date, color='red', alpha=0.3)

        ax.set_xlabel('Tarih')
        ax.set_ylabel('Oy Yüzdesi')
        ax.set_title('Tahminler ve Güven Aralıkları')
        ax.legend()
        st.pyplot(fig)

    except ValueError:
        st.text("Geçersiz tarih formatı. Tekrar deneyin.")
