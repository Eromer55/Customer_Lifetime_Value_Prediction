##############################################################################################
# Customer Life Value (Müşteri Yaşam Boyu Değeri)
###############################################################################################

##############################################################################################
# 1.Veri Hazırlama
###############################################################################################

##############################################################################################
# İş Problemi
###############################################################################################

#Bir e-ticaret şirketi müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.

#Değişkenler

# İnvoiceNo : Fatura numarası : Her işleme yani faturaya eşsiz numara. C ile başlıyorsa iptal edilen işlem.
# StockCode : Ürün kodu. Her bir ürün için eşsiz numara.
# Description : Ürün ismi
# Quantity : Ürün adedi. Faturalardaki ürünlerden kaçar tane satıldığını ifade etmektedir.
# UnitPrice :  Ürün fiyatı (Sterlin üzerinden)
# InvoiceDate :  Fatura tarihi ve zamanı
# CustomerId : Eşsiz Müşteri numarası
# Country : Ülke ismi : Müşterinin yaşadığı ülke

import datetime as dt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows, None)
#Aşağıdaki kod çıktılarda ki sayısal değerlerin kaç decimal gözükmesini ayarlar
pd.set_option('display.float_format', lambda x : '%.5f' % x)

df_ = pd.read_excel('crm_analytics/online_retail_II.xlsx', sheet_name="Year 2009-2010")
df = df_.copy()
df.head()
df.shape
df.isnull().sum()

df = df[~df['Invoice'].str.contains("C", na=False)]

df = df[df['Quantity'] > 0]

df.dropna(inplace=True)

df.describe().T

df['Total_Price'] = df['Quantity'] * df['Price']

cltv_c = df.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                        'Quantity': lambda x: x.sum(),
                                        'Total_Price': lambda x: x.sum()})

cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

##############################################################################################
# 2. Average Order Value (average_order_value = total_price / total_transaction)
###############################################################################################

cltv_c.head()

cltv_c['average_order_value'] = cltv_c["total_price"] / cltv_c['total_transaction']

##############################################################################################
# 3. Satın Alma Sıklığı - Purchase Frequency (total_transaction / total_number_of_customers)
###############################################################################################

cltv_c.head()
cltv_c.shape
cltv_c.shape[0]
cltv_c['purchase_frequency'] = cltv_c["total_transaction"] / cltv_c.shape[0]

##############################################################################################
# 4. Repeat Rate & Churn Rate (birden fazla alışveriş yapan müşteri sayısı / tüm müşteriler) - Tekrarlama ve Kaybetme Oranı
###############################################################################################

cltv_c.head()

repeat_rate = cltv_c[cltv_c['total_transaction'] > 1].shape[0] / cltv_c.shape[0]

churn_rate = 1 - repeat_rate

##############################################################################################
# 5. Profit Margin (profit_margin = total_price * 0.10)- Kar Marjı
###############################################################################################

cltv_c.head()

cltv_c['profit_margin'] = cltv_c['total_price'] * 0.10

##############################################################################################
# 6. Customer Value (customer_value = average_order_value * purchase_frequency)- Müşteri Değeri
###############################################################################################

cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c['purchase_frequency']

##############################################################################################
# 7. Customer LifeTime Value (CLTV = (customer_value / churn_rate) * profit_margin)- Müşteri Yaşam Boyu Değeri
###############################################################################################

cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

cltv_c.sort_values(by = "cltv", ascending=False).head()

##############################################################################################
# 8. Segmentlerin Oluşturulması
###############################################################################################
cltv_c['segment'] = pd.qcut(cltv_c['cltv'], 4, labels=["D", "C", "B", "A"])

cltv_c.groupby('segment').agg(['count', 'mean', 'sum'])

##############################################################################################
# 9. Tüm Sürecin Fonksiyonlaştırılması
###############################################################################################

def creat_cltv_c(dataframe, profit = 0.10):

    # Veriyi Hazırlama
    dataframe = dataframe[~dataframe['Invoice'].str.contains("C", na=False)]
    dataframe = dataframe[dataframe['Quantity'] > 0]
    dataframe.dropna(inplace=True)
    dataframe['Total_Price'] = dataframe['Quantity'] * dataframe['Price']
    cltv_c = dataframe.groupby('Customer ID').agg({'Invoice': lambda x: x.nunique(),
                                            'Quantity': lambda x: x.sum(),
                                            'Total_Price': lambda x: x.sum()})
    cltv_c.columns = ['total_transaction', 'total_unit', 'total_price']

    # avg_order_value
    cltv_c['average_order_value'] = cltv_c["total_price"] / cltv_c['total_transaction']

    # purchase_frequency
    cltv_c['purchase_frequency'] = cltv_c["total_transaction"] / cltv_c.shape[0]

    # repeat rate & churn_rate
    repeat_rate = cltv_c[cltv_c['total_transaction'] > 1].shape[0] / cltv_c.shape[0]
    churn_rate = 1 - repeat_rate

    # profit_margin
    cltv_c['profit_margin'] = cltv_c['total_price'] * profit

    # Customer_Value
    cltv_c['customer_value'] = cltv_c['average_order_value'] * cltv_c['purchase_frequency']

    # Customer_Lifetime_Value
    cltv_c['cltv'] = (cltv_c['customer_value'] / churn_rate) * cltv_c['profit_margin']

    # Segment
    cltv_c['segment'] = pd.qcut(cltv_c['cltv'], 4, labels=["D", "C", "B", "A"])

    return cltv_c

df = df_.copy()

clv = creat_cltv_c(df)
clv