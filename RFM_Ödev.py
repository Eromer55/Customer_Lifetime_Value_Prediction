##############################################################################################
# RFM İle Müşteri Segmentasyonu
###############################################################################################

##############################################################################################
# İş Problemi
###############################################################################################

# Online ayakkabı mağazası olan FLO müşterilerini segmentlere ayırıp bu segmentlere göre pazarlama stratejileri belirlemek istiyor.
# Buna yönelik olarak müşterilerin davranışları tanımlanacak ve bu davranışlardaki öbeklenmelere göre gruplar oluşturulacak.

##############################################################################################
# Veri Seti Hikayesi
###############################################################################################

# Veri seti Flo’dan son alışverişlerini 2020 - 2021 yıllarında OmniChannel (hem online hem offline alışveriş yapan)
# olarak yapan müşterilerin geçmiş alışveriş davranışlarından elde edilen bilgilerden oluşmaktadır.

# Değişkenler
# master_id Eşsiz müşteri numarası
# order_channel Alışveriş yapılan platforma ait hangi kanalın kullanıldığı (Android, ios, Desktop, Mobile)
# last_order_channel En son alışverişin yapıldığı kanal
# first_order_date Müşterinin yaptığı ilk alışveriş tarihi
# last_order_date Müşterinin yaptığı son alışveriş tarihi
# last_order_date_online Müşterinin online platformda yaptığı son alışveriş tarihi
# last_order_date_offline Müşterinin offline platformda yaptığı son alışveriş tarihi
# order_num_total_ever_online Müşterinin online platformda yaptığı toplam alışveriş sayısı
# order_num_total_ever_offline Müşterinin offline'da yaptığı toplam alışveriş sayısı
# customer_value_total_ever_offline Müşterinin offline alışverişlerinde ödediği toplam ücret
# customer_value_total_ever_online Müşterinin online alışverişlerinde ödediği toplam ücret
# interested_in_categories_12 Müşterinin son 12 ayda alışveriş yaptığı kategorilerin listesi

##############################################################################################
# Görev 1 Veriyi Anlama ve Hazırlama
###############################################################################################
import datetime as dt
import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 500)
#Aşağıdaki kod çıktılarda ki sayısal değerlerin kaç decimal gözükmesini ayarlar
pd.set_option('display.float_format', lambda x : '%.2f' % x)
pd.set_option('display.width', 1000)
from sklearn.preprocessing import MinMaxScaler

# Adım1: flo_data_20K.csv verisini okuyunuz.Dataframe’in kopyasını oluşturunuz.
df_ = pd.read_csv('crm_analytics/FLO_RFM_Analizi/flo_data_20k.csv')
df = df_.copy()
df.head()

# Adım2: Veri setinde a. İlk 10 gözlem, b. Değişken isimleri, c. Betimsel istatistik, d. Boş değer, e. Değişken tipleri, incelemesi yapınız.
df.head(10)
df.columns
df.shape
df.describe().T
df.isnull().sum()
df.info()

# Adım3: Omnichannel müşterilerin hem online'dan hemde offline platformlardan alışveriş yaptığını ifade etmektedir. Her bir müşterinin toplam
# alışveriş sayısı ve harcaması için yeni değişkenler oluşturunuz.
df.head()
df['order_num_total'] = df['order_num_total_ever_online'] + df['order_num_total_ever_offline']
df['customer_value_total'] = df['customer_value_total_ever_offline'] + df['customer_value_total_ever_online']
df.head()

# Adım4: Değişken tiplerini inceleyiniz. Tarih ifade eden değişkenlerin tipini date'e çeviriniz.

df.info()
#df['first_order_date'] = pd.to_datetime(df['first_order_date'])
date_columns = df.columns[df.columns.str.contains("date")]
df[date_columns] = df[date_columns].apply(pd.to_datetime)
df.info()

# Adım5: Alışveriş kanallarındaki müşteri sayısının, toplam alınan ürün sayısının ve toplam harcamaların dağılımına bakınız.
df.groupby('order_channel').agg({'master_id': lambda x : x.nunique(),
                                'order_num_total': lambda x: x.sum(),
                                 'customer_value_total': lambda x:x.sum()})

# Adım6: En fazla kazancı getiren ilk 10 müşteriyi sıralayınız.
#df.groupby('master_id').agg({'customer_value_total' : 'sum'}).sort_values("customer_value_total", ascending=False).head(10)
df.sort_values("customer_value_total", ascending=False)[:10]

# Adım7: En fazla siparişi veren ilk 10 müşteriyi sıralayınız.
#df.groupby('master_id').agg({'order_num_total' : 'sum'}).sort_values("order_num_total", ascending=False).head(10)
df.sort_values("order_num_total", ascending=False).head(10)

# Adım8 : Veri ön hazırlığı kısmını fonksiyonlaştıralım.
def check_df(dataframe, head=5):
    print("##################### Shape ###################")
    print(dataframe.shape)
    print("##################### Types ###################")
    print(dataframe.dtypes)
    print("##################### Head ###################")
    print(dataframe.head(head))
    print("##################### Tail ###################")
    print(dataframe.tail(head))
    print("##################### NA ###################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles ###################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

    dataframe['order_num_total'] = dataframe['order_num_total_ever_online'] + dataframe['order_num_total_ever_offline']
    dataframe['customer_value_total'] = dataframe['customer_value_total_ever_offline'] + dataframe['customer_value_total_ever_online']

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    return dataframe

check_df(df)

##############################################################################################
# Görev 2: RFM Metriklerinin Hesaplanması
###############################################################################################

# Adım 1: Recency, Frequency ve Monetary tanımlarını yapınız.
# recency: Analiz tarihi ile son yaptığı işlem arasında geçen gün sayısı
# frequency: Toplam eşsiz işlem sayısı(total transaction)
# monetary: Toplam kazandırdığı miktar

#Adım 2: Müşteri özelinde Recency, Frequency ve Monetary metriklerini hesaplayınız. rfm adında dataframe'e atalım.
# Analiz tarihi belirlememiz gerekmektedir. (Veri setinde yer alan son işlem gününden iki gün sonrayı seçelim)
df['last_order_date'].max() # 2021-05-30
analysis_date = dt.datetime(2021, 6, 1)

rfm = pd.DataFrame()
rfm['customer_id'] = df['master_id']
rfm['recency'] = (analysis_date - df['last_order_date']).astype('timedelta64[D]')
rfm['frequency'] = df['order_num_total']
rfm['monetary'] = df['customer_value_total']
rfm.head()

##############################################################################################
# Görev 3: RF Skorunun Hesaplanması
###############################################################################################

#Adım 1: Recency, Frequency ve Monetary metriklerini qcut yardımı ile 1-5 arasında skorlara çeviriniz.
#Bu skorları recency_score, frequency_score ve monetary_score olarak kaydediniz.

rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

rfm.head()

# Adım 3: recency_score ve frequency_score’u tek bir değişken olarak ifade ediniz ve RF_SCORE olarak kaydediniz.

rfm['RF_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
rfm.head()

##############################################################################################
# Görev 4: RF Skorunun Segment Olarak Tanımlanması
###############################################################################################
# Adım 1: Oluşturulan RF skorları için segment tanımlamaları yapınız. Aşağıdaki seg_map yardımı ile skorları segmentlere çeviriniz.

# regex : Bu bir yaklaşım. Frequency 5 olan ve recency si 5 olana campion yaz.
seg_map = {r'[1-2][1-2]': 'hibernating',
           r'[1-2][3-4]': 'at_Risk',
           r'[1-2]5': 'cant_loose',
           r'3[1-2]': 'about_to_sleep',
           r'33': 'need_attention',
           r'[3-4][4-5]': 'loyal_customers',
           r'41': 'promising',
           r'51': 'new_customers',
           r'[4-5][2-3]': 'potential_loyalist',
           r'5[4-5]': 'champions'
}

rfm['segment']= rfm['RF_score'].replace(seg_map, regex=True)
rfm.head()

##############################################################################################
# Görev 5: Aksiyon Zamanı
###############################################################################################

#Adım1: Segmentlerin recency, frequnecy ve monetary ortalamalarını inceleyiniz.

rfm[['segment', 'recency', 'frequency', 'monetary']].groupby('segment').agg(['mean', 'count'])

# Adım2: RFM analizi yardımıyla aşağıda verilen 2 case için ilgili profildeki müşterileri bulun ve müşteri id'lerini csv olarak kaydediniz.

# a. FLO bünyesine yeni bir kadın ayakkabı markası dahil ediyor. Dahil ettiği markanın ürün fiyatları genel müşteri
#tercihlerinin üstünde. Bu nedenle markanın tanıtımı ve ürün satışları için ilgilenecek profildeki müşterilerle özel olarak
#iletişime geçmek isteniliyor. Sadık müşterilerinden(champions, loyal_customers) ve kadın kategorisinden alışveriş
#yapan kişiler özel olarak iletişim kurulacak müşteriler. Bu müşterilerin id numaralarını csv dosyasına kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["champions","loyal_customers"])]["customer_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("KADIN"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

# b. Erkek ve Çocuk ürünlerinde %40'a yakın indirim planlanmaktadır. Bu indirimle ilgili kategorilerle ilgilenen geçmişte
# iyi müşteri olan ama uzun süredir alışveriş yapmayan kaybedilmemesi gereken müşteriler, uykuda olanlar ve yeni
# gelen müşteriler özel olarak hedef alınmak isteniyor. Uygun profildeki müşterilerin id'lerini csv dosyasına kaydediniz.

target_segments_customer_ids = rfm[rfm["segment"].isin(["cant_loose","hibernating", "new_customers"])]["customer_id"]

cust_ids = df[(df["master_id"].isin(target_segments_customer_ids)) & (df["interested_in_categories_12"].str.contains("ERKEK")) | (df["interested_in_categories_12"].str.contains("COCUK"))]["master_id"]
cust_ids.to_csv("yeni_marka_hedef_müşteri_id.csv", index=False)
cust_ids.shape

##############################################################################################
# BONUS : Çalışmanın Fonksiyonlaştırılması
###############################################################################################

def create_rfm(dataframe):
    # Veriyi Hazırlama
    dataframe['order_num_total'] = dataframe['order_num_total_ever_online'] + dataframe['order_num_total_ever_offline']
    dataframe['customer_value_total'] = dataframe['customer_value_total_ever_offline'] + dataframe['customer_value_total_ever_online']

    date_columns = dataframe.columns[dataframe.columns.str.contains("date")]
    dataframe[date_columns] = dataframe[date_columns].apply(pd.to_datetime)

    #RFM Metriklerinin Hesaplanması
    dataframe['last_order_date'].max()  # 2021-05-30
    analysis_date = dt.datetime(2021, 6, 1)
    rfm = pd.DataFrame()
    rfm['customer_id'] = dataframe['master_id']
    rfm['recency'] = (analysis_date - dataframe['last_order_date']).astype('timedelta64[D]')
    rfm['frequency'] = dataframe['order_num_total']
    rfm['monetary'] = dataframe['customer_value_total']

    #RF Skorlarının Hesaplanması
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])
    rfm['RF_score'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))

    seg_map = {r'[1-2][1-2]': 'hibernating',
               r'[1-2][3-4]': 'at_Risk',
               r'[1-2]5': 'cant_loose',
               r'3[1-2]': 'about_to_sleep',
               r'33': 'need_attention',
               r'[3-4][4-5]': 'loyal_customers',
               r'41': 'promising',
               r'51': 'new_customers',
               r'[4-5][2-3]': 'potential_loyalist',
               r'5[4-5]': 'champions'
               }

    rfm['segment'] = rfm['RF_score'].replace(seg_map, regex=True)

    return rfm[['customer_id', 'recency', 'frequency', 'monetary', 'RF_score', 'segment']]

df = df_.copy()

create_rfm(df)

